"""Moving-tokens-only transformers ("sparse token update", STREAMDIFFUSION_SPARSE_TOKENS).

Frame to frame, a webcam image changes on a small part of the picture. In the selected
transformer levels only a fixed budget of K tokens (those whose input moved most since they
were last computed) is recomputed; every other token keeps the self-attention K/V and the
transformer output it had when it was last computed (per denoising step). Self-attention
still sees the whole picture: fresh K/V of the active tokens + cached K/V of the others.
K is fixed, so the engine has static shapes and CUDA graphs work: the engine only gathers
(active / inactive token lists and an ``order`` permutation back to the spatial layout), the
caches are written back outside the engine (one ``index_copy_`` per cache kind and level).

Two exports of the same UNet share the caches (``SparseUNetPair``):
- dense: computes every token and writes full-size K/V/(StreamV2V output)/delta caches (first
  frame, scene cut, more motion than the budget, prompt change, periodic refresh);
- sparse: computes K tokens, reads the caches for the others, outputs the K new entries.
StreamV2V: the attention also reads the previous frames' ring cache, and the ring entry of the
current frame is rebuilt full size (fresh active tokens + cached ones).

Measured (RTX 5080, SDXL UNet 1024 fp16, K = 25 % of the 32x32 tokens): 45.1 -> 29.5 ms.
"""
import logging
import math
import os
import types
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- configuration

def sparse_tokens_config():
    """(budget fraction or 0, must-update threshold, refresh interval in frames)."""
    try:
        budget = float(os.environ.get("STREAMDIFFUSION_SPARSE_TOKENS", "0") or 0)
    except ValueError:
        budget = 0.0
    if not 0.0 < budget < 1.0:
        return 0.0, 0.0, 0
    try:
        thr = float(os.environ.get("STREAMDIFFUSION_SPARSE_THRESH", "0.15"))
    except ValueError:
        thr = 0.15
    try:
        refresh = int(os.environ.get("STREAMDIFFUSION_SPARSE_REFRESH", "0"))
    except ValueError:
        refresh = 0
    return budget, thr, max(0, refresh)


def budget_tokens(budget: float, tokens: int) -> int:
    """K for a level: multiple of 16 (GEMM-friendly), at least 16, below the token count."""
    k = int(round(budget * tokens / 16.0)) * 16
    return max(16, min(tokens - 16, k))


def default_sides(is_sdxl: bool, height: int, width: int) -> List[int]:
    """Levels made sparse: SDXL's 32x32-at-1024 stack (60 of its 70 transformer layers,
    ~70 % of the UNet time); SD 1.5's two highest-resolution levels."""
    side = min(height, width) // 8
    return [side // 4] if is_sdxl else [side, side // 2]


def sparse_suffix(budget: float, mode: str) -> str:
    """Engine filename tag: dense export (shared by every budget) or sparse at the budget."""
    return "--spd" if mode == "dense" else f"--sp{int(round(budget * 100))}"


def sparse_engine_paths(unet_path: str, budget: float):
    """(dense, sparse) engine paths next to the regular UNet engine path."""
    stem = unet_path[:-len(".engine")] if unet_path.endswith(".engine") else unet_path
    return stem + sparse_suffix(budget, "dense") + ".engine", stem + sparse_suffix(budget, "sparse") + ".engine"


def groups_from_engine(trt_engine):
    """Level groups read back from a dense sparse-token engine's output ports
    sp_{k,v,o,d}_out_{side}: (n, B, T, ch)."""
    import re
    pat = re.compile(r"^sp_([kvod])_out_(\d+)$")
    found = {}
    for i in range(trt_engine.num_io_tensors):
        name = trt_engine.get_tensor_name(i)
        m = pat.match(name)
        if m:
            found.setdefault(int(m.group(2)), {})[m.group(1)] = tuple(trt_engine.get_tensor_shape(name))
    groups = []
    for s in sorted(found, reverse=True):
        f = found[s]
        groups.append({"side": s, "T": s * s, "L": f["k"][0], "C": f["k"][3], "NT": f["d"][0], "c": f["d"][3],
                       "o": "o" in f})
    return groups


# ---------------------------------------------------------------- export side

class SparseTokenContext:
    """Per-call tensors in and out of the patched Transformer2DModel.forward (per level side)."""
    def __init__(self):
        self.mode = "off"          # off | dense | sparse
        self.idx: Dict[int, torch.Tensor] = {}     # (K,) active tokens
        self.inv: Dict[int, torch.Tensor] = {}     # (T-K,) inactive tokens
        self.order: Dict[int, torch.Tensor] = {}   # (T,) position of each token in cat([active, inactive])
        self.k_in: Dict[int, torch.Tensor] = {}    # (L, B, T, C)
        self.v_in: Dict[int, torch.Tensor] = {}
        self.o_in: Dict[int, torch.Tensor] = {}    # StreamV2V attn1 outputs (L, B, T, C)
        self.d_in: Dict[int, torch.Tensor] = {}    # transformer output deltas (NT, B, T, c)
        self.begin()

    def begin(self):
        self.k_out: Dict[int, List[torch.Tensor]] = {}
        self.v_out: Dict[int, List[torch.Tensor]] = {}
        self.o_out: Dict[int, List[torch.Tensor]] = {}
        self.d_out: Dict[int, List[torch.Tensor]] = {}


def _lin(mod, x):
    """proj_in / proj_out on (B, N, C) tokens: Linear, or a 1x1 Conv2d (SD 1.5) as a linear."""
    if isinstance(mod, torch.nn.Conv2d):
        return F.linear(x, mod.weight.view(mod.out_channels, mod.in_channels), mod.bias)
    return mod(x)


def _attend(attn, q_in, k, v):
    b = q_in.shape[0]
    q = attn.to_q(q_in)
    hd = q.shape[-1] // attn.heads
    q = q.view(b, -1, attn.heads, hd).transpose(1, 2)
    k = k.view(b, -1, attn.heads, hd).transpose(1, 2)
    v = v.view(b, -1, attn.heads, hd).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, -1, attn.heads * hd)
    return attn.to_out[1](attn.to_out[0](o))


def _sparse_forward(self, hidden_states, encoder_hidden_states=None, timestep=None, added_cond_kwargs=None,
                    class_labels=None, cross_attention_kwargs=None, attention_mask=None,
                    encoder_attention_mask=None, return_dict=True):
    ctx: SparseTokenContext = self._sp_ctx
    if ctx.mode == "off":
        return self._sp_orig_forward(hidden_states, encoder_hidden_states=encoder_hidden_states,
                                     timestep=timestep, added_cond_kwargs=added_cond_kwargs,
                                     class_labels=class_labels, cross_attention_kwargs=cross_attention_kwargs,
                                     attention_mask=attention_mask, encoder_attention_mask=encoder_attention_mask,
                                     return_dict=return_dict)
    from .attention_processors import _get_nn_feats
    sparse = ctx.mode == "sparse"
    s = self._sp_side
    b, c, h, w = hidden_states.shape
    residual = hidden_states
    x = self.norm(hidden_states).permute(0, 2, 3, 1).reshape(b, h * w, c)
    if sparse:
        idx, inv, order = ctx.idx[s], ctx.inv[s], ctx.order[s]
        x = torch.index_select(x, 1, idx)
    x = _lin(self.proj_in, x)
    k_out, v_out = ctx.k_out.setdefault(s, []), ctx.v_out.setdefault(s, [])
    for li, blk in enumerate(self.transformer_blocks):
        lid = self._sp_first_layer + li
        a = blk.attn1
        n = blk.norm1(x)
        k, v = a.to_k(n), a.to_v(n)
        k_out.append(k)
        v_out.append(v)
        if sparse:
            k = torch.cat([k, torch.index_select(ctx.k_in[s][lid], 1, inv)], 1)
            v = torch.cat([v, torch.index_select(ctx.v_in[s][lid], 1, inv)], 1)
        proc = a.processor
        ring = hasattr(proc, "_pool_cached")        # StreamV2V ring processor (TRT export)
        cache = getattr(proc, "_cache_in", None) if ring else None
        k_att, v_att = k, v
        if cache and proc.use_cache_attn:
            k_att = torch.cat([k] + [proc._pool_cached(cf[0]) for cf in cache], 1)
            v_att = torch.cat([v] + [proc._pool_cached(cf[1]) for cf in cache], 1)
        o = _attend(a, n, k_att, v_att)
        if (cache and proc.is_decoder_block and proc.use_feature_injection and proc.fi_strength > 0.0):
            fi_cache = cache[-1:] if proc.fi_last_frame_only else cache
            cached_output = torch.cat([cf[2] for cf in fi_cache], dim=1)
            o = o * (1.0 - proc.fi_strength) + proc.fi_strength * _get_nn_feats(o, cached_output, threshold=proc.fi_threshold)
        if ring:
            ctx.o_out.setdefault(s, []).append(o)
            if sparse:
                o_full = torch.index_select(torch.cat([o, torch.index_select(ctx.o_in[s][lid], 1, inv)], 1), 1, order)
                proc._cache_out = torch.stack([torch.index_select(k, 1, order), torch.index_select(v, 1, order), o_full], 0)
            else:
                proc._cache_out = torch.stack([k, v, o], 0)
        x = x + o
        x = x + blk.attn2(blk.norm2(x), encoder_hidden_states=encoder_hidden_states)
        x = x + blk.ff(blk.norm3(x))
    d = _lin(self.proj_out, x)
    ctx.d_out.setdefault(s, []).append(d)
    if sparse:
        d = torch.index_select(torch.cat([d, torch.index_select(ctx.d_in[s][self._sp_index], 1, inv)], 1), 1, order)
    out = residual + d.reshape(b, h, w, c).permute(0, 3, 1, 2)
    if not return_dict:
        return (out,)
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    return Transformer2DModelOutput(sample=out)


def transformer_levels(unet, height: int, width: int):
    """[(Transformer2DModel, side)] of every attention transformer, side = token grid side."""
    side = min(height, width) // 8
    plan = []
    for blk in unet.down_blocks:
        plan.append((blk, side))
        if getattr(blk, "downsamplers", None):
            side //= 2
    plan.append((unet.mid_block, side))
    for blk in unet.up_blocks:
        plan.append((blk, side))
        if getattr(blk, "upsamplers", None):
            side *= 2
    out = []
    for blk, sd in plan:
        for tr in getattr(blk, "attentions", None) or []:
            out.append((tr, sd))
    return out


def install_sparse_transformers(unet, height: int, width: int, sides, ctx: SparseTokenContext,
                                has_v2v: bool = False):
    """Patch the Transformer2DModels of the given levels. Returns the level groups
    [{side, T, L (layers), C (attn inner dim), NT (transformers), c (channels), o (v2v)}]."""
    groups = {}
    for tr, sd in transformer_levels(unet, height, width):
        if sd not in sides or height % 8 or width != height:
            continue
        g = groups.setdefault(sd, {"side": sd, "T": sd * sd, "L": 0, "NT": 0,
                                   "C": tr.transformer_blocks[0].attn1.to_k.out_features,
                                   "c": tr.norm.num_channels, "o": bool(has_v2v)})
        if not hasattr(tr, "_sp_orig_forward"):
            tr._sp_orig_forward = tr.forward
        tr.forward = types.MethodType(_sparse_forward, tr)
        tr._sp_ctx, tr._sp_side = ctx, sd
        tr._sp_first_layer, tr._sp_index = g["L"], g["NT"]
        g["L"] += len(tr.transformer_blocks)
        g["NT"] += 1
    return [groups[s] for s in sorted(groups, reverse=True)]


def uninstall_sparse_transformers(unet):
    for tr, _ in transformer_levels(unet, 64, 64):
        if hasattr(tr, "_sp_orig_forward"):
            tr.forward = tr._sp_orig_forward
            del tr._sp_orig_forward


def _kinds(g):
    return ("k", "v", "o", "d") if g["o"] else ("k", "v", "d")


def _cache_dims(g, kind):
    """(leading count, channels) of a cache kind."""
    return (g["NT"], g["c"]) if kind == "d" else (g["L"], g["C"])


class SparseTokenSpec:
    """Engine I/O spec = the base UNet spec + the sparse-token ports (delegates the rest)."""
    def __init__(self, base, groups, mode: str, budget: float):
        self.base, self.groups, self.mode, self.budget = base, groups, mode, budget
        self.name = f"{base.name}-sparse-{mode}"

    def __getattr__(self, name):
        return getattr(self.__dict__["base"], name)

    def k_of(self, g):
        return g["T"] if self.mode == "dense" else budget_tokens(self.budget, g["T"])

    def sparse_input_names(self):
        if self.mode != "sparse":
            return []
        names = []
        for g in self.groups:
            s = g["side"]
            names += [f"sp_idx_{s}", f"sp_inv_{s}", f"sp_order_{s}"] + [f"sp_{k}_{s}" for k in _kinds(g)]
        return names

    def sparse_output_names(self):
        return [f"sp_{k}_out_{g['side']}" for g in self.groups for k in _kinds(g)]

    def get_input_names(self):
        return self.base.get_input_names() + self.sparse_input_names()

    def get_output_names(self):
        return self.base.get_output_names() + self.sparse_output_names()

    def get_dynamic_axes(self):
        axes = dict(self.base.get_dynamic_axes() or {})
        for n in self.sparse_input_names() + self.sparse_output_names():
            if not any(n.startswith(p) for p in ("sp_idx_", "sp_inv_", "sp_order_")):
                axes[n] = {1: "2B"}
        return axes

    def _shapes(self, batch):
        d = {}
        for g in self.groups:
            s, T, K = g["side"], g["T"], self.k_of(g)
            if self.mode == "sparse":
                d[f"sp_idx_{s}"], d[f"sp_inv_{s}"], d[f"sp_order_{s}"] = (K,), (T - K,), (T,)
                for kind in _kinds(g):
                    n, ch = _cache_dims(g, kind)
                    d[f"sp_{kind}_{s}"] = (n, batch, T, ch)
            for kind in _kinds(g):
                n, ch = _cache_dims(g, kind)
                d[f"sp_{kind}_out_{s}"] = (n, batch, K, ch)
        return d

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        profile = self.base.get_input_profile(batch_size, image_height, image_width, static_batch, static_shape)
        min_b, max_b = self.base.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)[:2]
        for n, shp in self._shapes(batch_size).items():
            if n not in self.sparse_input_names():
                continue
            if len(shp) == 1:
                profile[n] = [shp, shp, shp]
            else:
                profile[n] = [(shp[0], min_b) + shp[2:], shp, (shp[0], max_b) + shp[2:]]
        return profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        d = self.base.get_shape_dict(batch_size, image_height, image_width)
        for n, shp in self._shapes(2 * batch_size).items():
            d[n] = shp
        return d

    def get_sample_input(self, batch_size, image_height, image_width):
        base = list(self.base.get_sample_input(batch_size, image_height, image_width))
        if self.mode != "sparse":
            return tuple(base)
        dtype = torch.float16 if self.fp16 else torch.float32
        for g in self.groups:
            T, K = g["T"], self.k_of(g)
            perm = torch.arange(T, device=self.device)
            base += [perm[:K].clone(), perm[K:].clone(), perm.clone()]
            for kind in _kinds(g):
                n, ch = _cache_dims(g, kind)
                base.append(torch.zeros(n, 2 * batch_size, T, ch, dtype=dtype, device=self.device))
        return tuple(base)


class TorchSparseWrapper(torch.nn.Module):
    """Wraps the UNet ONNX wrapper (plain CN or StreamV2V ring): sets the sparse context from
    the extra inputs, runs the base wrapper, appends the stacked cache outputs."""
    def __init__(self, base_wrapper, spec: SparseTokenSpec, ctx: SparseTokenContext):
        super().__init__()
        self.base_wrapper = base_wrapper
        self._spec, self._ctx = spec, ctx
        self._n_base = len(spec.base.get_input_names())

    def forward(self, *args):
        ctx, spec = self._ctx, self._spec
        ctx.mode = spec.mode
        ctx.begin()
        extra = args[self._n_base:]
        if spec.mode == "sparse":
            i = 0
            for g in spec.groups:
                s = g["side"]
                ctx.idx[s], ctx.inv[s], ctx.order[s] = extra[i], extra[i + 1], extra[i + 2]
                i += 3
                for kind in _kinds(g):
                    getattr(ctx, f"{kind}_in")[s] = extra[i]
                    i += 1
        try:
            out = self.base_wrapper(*args[:self._n_base])
        finally:
            ctx.mode = "off"
        out = out if isinstance(out, tuple) else (out,)
        stacks = []
        for g in spec.groups:
            s = g["side"]
            for kind in _kinds(g):
                stacks.append(torch.stack(getattr(ctx, f"{kind}_out")[s], 0))
        return out + tuple(stacks)


# ---------------------------------------------------------------- runtime side

class TokenSelector:
    """Per-token motion of the input since the token was last computed; top-K tokens per level.
    The budget left by the moving tokens goes to the longest-unchanged ones (round robin on a
    still image): a token is never frozen on a state that later frames would have changed,
    such as the first StreamV2V frames (empty cache, flat image).
    ``select`` returns False when the frame must be dense (warm-up after a reset, refresh, more
    must-update tokens than the budget on any level)."""
    AGE_WEIGHT = 0.005     # score of one frame without update (10 frames ~ faint motion 0.05)

    def __init__(self, groups, budget, thr, refresh, warmup=1):
        self.groups, self.budget, self.thr, self.refresh = groups, budget, thr, refresh
        self.warmup = max(1, int(warmup))
        self.ref = {}
        self.age = {}
        self.frames = 0
        self.dense_left = self.warmup

    def reset(self):
        self.ref = {}
        self.dense_left = self.warmup

    def select(self, x, bufs):
        """``x``: input (B, 3, H, W) in [-1, 1]; ``bufs``: side -> (idx, inv, order) buffers."""
        luma = ((0.299 * x[0:1, 0:1] + 0.587 * x[0:1, 1:2] + 0.114 * x[0:1, 2:3]) * 0.5 + 0.5).float()
        self.frames += 1
        dense = self.dense_left > 0 or not self.ref or (self.refresh and self.frames % self.refresh == 0)
        cells = {}
        plans = {}
        for g in self.groups:
            s, T = g["side"], g["T"]
            cell = F.adaptive_avg_pool2d(luma, s)
            fine = F.adaptive_avg_pool2d(luma, s * 4)
            cells[s] = (cell, fine)
            if dense or s not in self.ref:
                dense = True
                continue
            rc, rf = self.ref[s]
            score = (cell - rc).abs() + F.avg_pool2d((fine - rf).abs(), 4)
            score = F.max_pool2d(score, 3, stride=1, padding=1).flatten()
            plans[s] = score
        if not dense:
            # one host sync per frame: must-update tokens vs the budget on every level
            over = torch.stack([(plans[g["side"]] > self.thr).sum() - budget_tokens(self.budget, g["T"])
                                for g in self.groups]).max()
            dense = bool(over > 0)
        if dense:
            for g in self.groups:
                s = g["side"]
                self.ref[s] = (cells[s][0].clone(), cells[s][1].clone())
                self.age[s] = torch.zeros(g["T"], dtype=torch.float32, device=x.device)
            self.dense_left = max(0, self.dense_left - 1)
            return False
        for g in self.groups:
            s, T = g["side"], g["T"]
            K = budget_tokens(self.budget, T)
            age = self.age[s]
            age += 1.0
            active = torch.zeros(T, dtype=torch.bool, device=x.device)
            active[(plans[s] + self.AGE_WEIGHT * age).topk(K).indices] = True
            age.masked_fill_(active, 0.0)
            perm = torch.argsort((~active).to(torch.int8), stable=True)   # active first, ascending
            idx_b, inv_b, order_b = bufs[s]
            idx_b.copy_(perm[:K])
            inv_b.copy_(perm[K:])
            order_b.scatter_(0, perm, torch.arange(T, device=x.device))
            rc, rf = self.ref[s]
            m = active.view(1, 1, s, s)
            rc.copy_(torch.where(m, cells[s][0], rc))
            rf.copy_(torch.where(F.interpolate(m.float(), scale_factor=4) > 0, cells[s][1], rf))
        return True


class _SparseIO:
    """Engine hook (UNet2DConditionModelEngine.extra_io): external port names, shapes and
    per-step binding of the caches; cache write-back after a sparse run."""
    def __init__(self, rt, mode):
        self.rt, self.mode = rt, mode
        spec_names = []
        for g in rt.groups:
            s = g["side"]
            if mode == "sparse":
                spec_names += [f"sp_idx_{s}", f"sp_inv_{s}", f"sp_order_{s}"] + [f"sp_{k}_{s}" for k in _kinds(g)]
            spec_names += [f"sp_{k}_out_{s}" for k in _kinds(g)]
        self._names = tuple(spec_names)

    def names(self):
        return self._names

    def shapes(self, batch):
        self.rt.batch = batch
        d = {}
        for g in self.rt.groups:
            s, T = g["side"], g["T"]
            K = T if self.mode == "dense" else self.rt.K[s]
            for kind in _kinds(g):
                n, ch = _cache_dims(g, kind)
                if self.mode == "sparse":
                    d[f"sp_{kind}_{s}"] = (n, batch, T, ch)
                    d[f"sp_idx_{s}"], d[f"sp_inv_{s}"], d[f"sp_order_{s}"] = (K,), (T - K,), (T,)
                d[f"sp_{kind}_out_{s}"] = (n, batch, K, ch)
        return d

    def bind(self, engine, step):
        caches = self.rt.caches(step)
        for g in self.rt.groups:
            s = g["side"]
            for kind in _kinds(g):
                if self.mode == "dense":
                    engine.bind_external(f"sp_{kind}_out_{s}", caches[s][kind])
                else:
                    engine.bind_external(f"sp_{kind}_{s}", caches[s][kind])
                    engine.bind_external(f"sp_{kind}_out_{s}", self.rt.scratch[s][kind])
            if self.mode == "sparse":
                idx, inv, order = self.rt.index_bufs[s]
                engine.bind_external(f"sp_idx_{s}", idx)
                engine.bind_external(f"sp_inv_{s}", inv)
                engine.bind_external(f"sp_order_{s}", order)

    def after(self, step):
        if self.mode == "dense":
            self.rt.valid.add(step)
            return
        caches = self.rt.caches(step)
        for g in self.rt.groups:
            s = g["side"]
            idx = self.rt.index_bufs[s][0]
            for kind in _kinds(g):
                caches[s][kind].index_copy_(2, idx, self.rt.scratch[s][kind])


class SparseUNetPair:
    """Dense + sparse UNet engines sharing the StreamV2V rings and the token caches; stands in
    for UNet2DConditionModelEngine in the pipeline (v2v_slot, prune_v2v_slots, engine)."""
    def __init__(self, dense, sparse, groups, budget, thr, refresh, device, dtype, warmup=1):
        self.dense, self.sparse = dense, sparse
        self.groups, self.budget = groups, budget
        self.K = {g["side"]: budget_tokens(budget, g["T"]) for g in groups}
        self.device, self.dtype = device, dtype
        self.batch = None
        self._caches = {}              # step -> side -> kind -> (n, B, T, ch)
        self.valid = set()             # steps whose caches were filled by a dense run
        self.scratch = {}              # side -> kind -> (n, B, K, ch)
        self.index_bufs = {g["side"]: (torch.zeros(self.K[g["side"]], dtype=torch.long, device=device),
                                       torch.zeros(g["T"] - self.K[g["side"]], dtype=torch.long, device=device),
                                       torch.zeros(g["T"], dtype=torch.long, device=device))
                           for g in groups}
        # warmup: dense frames after a reset (StreamV2V: until its frame cache is full)
        self.selector = TokenSelector(groups, budget, thr, refresh, warmup)
        self.frame_sparse = False
        self.n_sparse = self.n_dense = 0
        # One StreamV2V history for both engines: share the ring dicts (never reassigned).
        sparse._rings = dense._rings
        sparse._ring_phases = dense._ring_phases
        dense.extra_io = _SparseIO(self, "dense")
        sparse.extra_io = _SparseIO(self, "sparse")

    # -- cache storage
    def caches(self, step):
        c = self._caches.get(step)
        if c is None:
            c = {}
            for g in self.groups:
                s, T = g["side"], g["T"]
                c[s] = {}
                for kind in _kinds(g):
                    n, ch = _cache_dims(g, kind)
                    c[s][kind] = torch.zeros(n, self.batch, T, ch, dtype=self.dtype, device=self.device)
                    if s not in self.scratch or kind not in self.scratch[s]:
                        self.scratch.setdefault(s, {})[kind] = torch.zeros(
                            n, self.batch, self.K[s], ch, dtype=self.dtype, device=self.device)
            self._caches[step] = c
        return c

    # -- per frame
    def begin_frame(self, x):
        """Select this frame's tokens from the input image (None = dense)."""
        if x is None:
            self.selector.dense_left = max(self.selector.dense_left, 1)
            self.frame_sparse = False
        else:
            self.frame_sparse = self.selector.select(x, self.index_bufs)
        if self.frame_sparse and self.valid:
            self.n_sparse += 1
        else:
            self.frame_sparse = False
            self.n_dense += 1

    def reset(self):
        self.selector.reset()
        self.valid.clear()

    # -- engine facade
    @property
    def engine(self):
        return (self.sparse if self.frame_sparse else self.dense).engine

    @property
    def stream(self):
        # CUDA stream of both engines (the TensorRT ControlNets are built/run on it)
        return self.dense.stream

    @property
    def _is_v2v(self):
        return self.dense._is_v2v

    @property
    def v2v_slot(self):
        return self.dense.v2v_slot

    @v2v_slot.setter
    def v2v_slot(self, slot):
        self.dense.v2v_slot = slot
        self.sparse.v2v_slot = slot

    def prune_v2v_slots(self, n_slots):
        self.dense.prune_v2v_slots(n_slots)
        self.sparse.prune_v2v_slots(n_slots)
        for s in [s for s in self._caches if s >= max(1, int(n_slots))]:
            del self._caches[s]
            self.valid.discard(s)

    def __call__(self, *args, **kwargs):
        eng = self.sparse if (self.frame_sparse and self.dense.v2v_slot in self.valid) else self.dense
        return eng(*args, **kwargs)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass
