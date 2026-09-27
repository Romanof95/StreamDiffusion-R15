"""Step cache (DeepCache-style) for multi-step SD 1.5: the deep UNet levels change little from one
denoising step to the next, so only every Nth step runs the full UNet; the others run a shallow UNet
(the two highest-resolution levels: conv_in, down_blocks[0:2], up_blocks[-2:], conv_out) on the
input of up_blocks[-2] kept from the last full step.

Engines: the full one is the regular UNet with one more output ``dc_feat`` (that deep feature), the
shallow one takes the same inputs plus ``dc_feat``. Its StreamV2V ports of the skipped layers output
zeros: those layers never run at a shallow step, so their ring slots for that step are never read.
Unused inputs (the ControlNet residuals of the skipped levels, the cache inputs of the skipped layers)
are pruned by the build; the runtime skips them.

Webcam test (kohaku + Hyper-SD15 4 steps @512): full every 2 steps looks the same as every step,
shallow UNet = 62% of the full one under TensorRT. Recomputing the 64x64 level only (37%) broke the
4-step image (colors bleeding, smeared faces)."""
import logging
import os

import torch

BRANCH = 1          # deep levels skipped below up_blocks[-1 - BRANCH]


def step_cache_config() -> int:
    """Full UNet every N steps (STREAMDIFFUSION_STEP_CACHE, N >= 2), 0 = off."""
    try:
        n = int(os.environ.get("STREAMDIFFUSION_STEP_CACHE", "0") or 0)
    except ValueError:
        n = 0
    return n if n >= 2 else 0


def step_cache_engine_paths(unet_path: str):
    """(full, shallow) engine paths next to the regular UNet engine path."""
    stem = unet_path[:-len(".engine")] if unet_path.endswith(".engine") else unet_path
    return stem + "--dcf.engine", stem + "--dcs.engine"


def feat_shape(batch: int, latent_h: int, latent_w: int, unet_config=None, channels: int = None):
    """Shape of ``dc_feat`` (input of up_blocks[-2], SD 1.5: 1280 channels at half the latent)."""
    if channels is None:
        channels = list(unet_config.block_out_channels)[-1 - BRANCH]
    return (batch, channels, latent_h // 2 ** BRANCH, latent_w // 2 ** BRANCH)


class StepCacheSpec:
    """Engine I/O spec = the base UNet spec (plain or StreamV2V ring) + ``dc_feat``."""
    def __init__(self, base, mode: str, channels: int):
        self.base, self.mode, self.channels = base, mode, channels
        self.name = f"{base.name}-stepcache-{mode}"

    def __getattr__(self, name):
        return getattr(self.__dict__["base"], name)

    def get_input_names(self):
        return self.base.get_input_names() + (["dc_feat"] if self.mode == "shallow" else [])

    def get_output_names(self):
        return self.base.get_output_names() + (["dc_feat"] if self.mode == "full" else [])

    def get_dynamic_axes(self):
        axes = dict(self.base.get_dynamic_axes() or {})
        axes["dc_feat"] = {0: "2B"}
        return axes

    def _feat(self, batch_size, image_height, image_width, batch=None):
        lh, lw = self.base.check_dims(batch_size, image_height, image_width)
        return feat_shape(batch_size if batch is None else batch, lh, lw, channels=self.channels)

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        profile = self.base.get_input_profile(batch_size, image_height, image_width, static_batch, static_shape)
        if self.mode == "shallow":
            min_b, max_b = self.base.get_minmax_dims(batch_size, image_height, image_width,
                                                     static_batch, static_shape)[:2]
            shp = self._feat(batch_size, image_height, image_width)
            profile["dc_feat"] = [(min_b,) + shp[1:], shp, (max_b,) + shp[1:]]
        return profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        d = self.base.get_shape_dict(batch_size, image_height, image_width)
        d["dc_feat"] = self._feat(batch_size, image_height, image_width, 2 * batch_size)
        return d

    def get_sample_input(self, batch_size, image_height, image_width):
        base = tuple(self.base.get_sample_input(batch_size, image_height, image_width))
        if self.mode != "shallow":
            return base
        dtype = torch.float16 if self.fp16 else torch.float32
        return base + (torch.zeros(self._feat(batch_size, image_height, image_width, 2 * batch_size),
                                   dtype=dtype, device=self.device),)


class TorchStepCacheWrapper(torch.nn.Module):
    """Wraps the UNet ONNX wrapper (TorchUNetWrapper or TorchUNetV2VRingWrapper: sample, timestep,
    encoder_hidden_states, 12 down + 1 mid ControlNet residuals, then the ring cache inputs).
    ``full``: the base forward + the captured deep feature. ``shallow``: the two outer levels on
    ``dc_feat``, StreamV2V outputs of the skipped layers as zeros."""
    def __init__(self, base_wrapper, spec: StepCacheSpec):
        super().__init__()
        self.base_wrapper, self._spec = base_wrapper, spec
        self.unet = base_wrapper.unet
        self._procs = list(getattr(base_wrapper, "_kvo_processors", None) or [])
        self._frames = int(getattr(base_wrapper, "_max_frames", 1) or 1)
        self._n_base = len(spec.base.get_input_names())
        self._feat = None
        if spec.mode == "full":
            self.unet.up_blocks[-1 - BRANCH].register_forward_pre_hook(self._grab, with_kwargs=True)

    def _grab(self, mod, args, kwargs):
        self._feat = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]

    def forward(self, *args):
        if self._spec.mode == "full":
            out = self.base_wrapper(*args[:self._n_base])
            out = out if isinstance(out, tuple) else (out,)
            return out + (self._feat,)
        u = self.unet
        sample, timestep, ehs = args[0], args[1], args[2]
        cn = args[3:15]
        kvo_in = args[16:self._n_base]
        feat = args[self._n_base]
        n = self._frames
        for k, proc in enumerate(self._procs):
            proc._cache_in = tuple(kvo_in[k * n:(k + 1) * n])
            proc._cache_out = None
        sample = sample.to(u.conv_in.weight.dtype)
        t = timestep.expand(sample.shape[0]) if timestep.dim() else timestep.view(1).expand(sample.shape[0])
        emb = u.time_embedding(u.time_proj(t).to(sample.dtype))
        h = u.conv_in(sample)
        res_all = (h,)
        for db in u.down_blocks[:BRANCH + 1]:
            if getattr(db, "has_cross_attention", False):
                h, res = db(hidden_states=h, temb=emb, encoder_hidden_states=ehs)
            else:
                h, res = db(hidden_states=h, temb=emb)
            res_all += tuple(res)
        res_all = tuple(r + c for r, c in zip(res_all, cn[:len(res_all)]))
        h = feat.to(sample.dtype)
        up = u.up_blocks[len(u.up_blocks) - BRANCH - 1:]
        for j, ub in enumerate(up):
            lo = 3 * (BRANCH - j)
            kw = dict(hidden_states=h, temb=emb, res_hidden_states_tuple=res_all[lo:lo + 3])
            if getattr(ub, "has_cross_attention", False):
                kw["encoder_hidden_states"] = ehs
            h = ub(**kw)
        h = u.conv_out(u.conv_act(u.conv_norm_out(h)))
        outs = [h]
        for i, proc in enumerate(self._procs):
            if proc._cache_out is not None:
                outs.append(proc._cache_out)
            else:
                seq, dim = self._spec.base.kvo_cache_shapes[i]
                outs.append(h.new_zeros(3, sample.shape[0], seq, dim))
        return tuple(outs) if len(outs) > 1 else h


class _StepCacheIO:
    """Engine hook (UNet2DConditionModelEngine.extra_io): the ``dc_feat`` port, bound to the
    pair's buffer (output of the full engine, input of the shallow one)."""
    def __init__(self, rt):
        self.rt = rt

    def names(self):
        return ("dc_feat",)

    def shapes(self, batch):
        return {"dc_feat": self.rt.feat_buffer(batch).shape}

    def bind(self, engine, step):
        engine.bind_external("dc_feat", self.rt.feat)

    def after(self, step):
        pass


class StepCacheUNetPair:
    """Full + shallow UNet engines sharing the StreamV2V rings; stands in for
    UNet2DConditionModelEngine in the pipeline (v2v_slot = denoising step, set before each step)."""
    def __init__(self, full, shallow, interval: int, latent_hw, channels: int, device, dtype):
        self.full, self.shallow, self.interval = full, shallow, int(interval)
        self.latent_hw, self.channels = latent_hw, channels
        self.device, self.dtype = device, dtype
        self.feat = None
        self.n_full = self.n_shallow = 0
        shallow._rings = full._rings
        shallow._ring_phases = full._ring_phases
        full.extra_io = _StepCacheIO(self)
        shallow.extra_io = _StepCacheIO(self)

    def feat_buffer(self, batch):
        shp = feat_shape(batch, *self.latent_hw, channels=self.channels)
        if self.feat is None or tuple(self.feat.shape) != shp:
            self.feat = torch.zeros(shp, dtype=self.dtype, device=self.device)
        return self.feat

    @property
    def engine(self):
        return self.full.engine

    @property
    def stream(self):
        # CUDA stream of both engines (the TensorRT ControlNets are built/run on it)
        return self.full.stream

    @property
    def _is_v2v(self):
        return self.full._is_v2v

    @property
    def v2v_slot(self):
        return self.full.v2v_slot

    @v2v_slot.setter
    def v2v_slot(self, slot):
        self.full.v2v_slot = slot
        self.shallow.v2v_slot = slot

    def prune_v2v_slots(self, n_slots):
        self.full.prune_v2v_slots(n_slots)
        self.shallow.prune_v2v_slots(n_slots)

    def __call__(self, *args, **kwargs):
        if self.full.v2v_slot % self.interval == 0:
            self.n_full += 1
            return self.full(*args, **kwargs)
        self.n_shallow += 1
        return self.shallow(*args, **kwargs)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


def step_cache_log(interval: int, steps: int) -> None:
    plan = "".join("F" if i % interval == 0 else "s" for i in range(steps))
    logging.info(f"[StepCache] On: full UNet every {interval} steps ({plan}, s = levels 64/32 only)")
