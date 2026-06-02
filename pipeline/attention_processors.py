"""
StreamV2V temporal consistency.

Two paths:
- torch.compile: K/V cache lives in pre-allocated registered buffers on each
  attn1; cache writes run in eager mode (disabled from compilation) to avoid
  CUDA graph errors.
- TensorRT: cache flows through the engine as named I/O tensors, set per-frame
  via side-channel attributes on the processor.

Based on https://github.com/Jeff-LiangF/streamv2v (arxiv 2405.15757).
"""

import logging
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0


def _get_nn_feats(x, y, threshold=0.95):
    """Nearest-neighbor feature matching via cosine similarity."""
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    cosine_sim = torch.bmm(x_norm, y_norm.transpose(1, 2))
    max_vals, nn_indices = torch.max(cosine_sim, dim=-1)
    nn_feats = torch.gather(y, 1, nn_indices.unsqueeze(-1).expand(-1, -1, y.shape[-1]))
    mask = (max_vals < threshold).unsqueeze(-1)
    return torch.where(mask, x, nn_feats)


@torch.compiler.disable
def _write_cache(dst, src):
    """Write to registered buffer outside the compiled graph (avoids CUDA graph errors)."""
    dst.copy_(src.clone())


class StreamV2VAttnProcessor2_0:
    """Stateless attention processor; cache lives in registered buffers on the Attention module."""

    def __init__(self, cache_enabled=True, is_decoder_block=False,
                 use_feature_fusion=True, ff_strength=0.95, ff_threshold=0.95):
        self._cache_enabled = cache_enabled
        self.is_decoder_block = is_decoder_block
        self.use_feature_fusion = use_feature_fusion
        self.ff_strength = ff_strength
        self.ff_threshold = ff_threshold

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        is_self_attn = encoder_hidden_states is None

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Extended Attention
        if is_self_attn and self._cache_enabled:
            _write_cache(attn._sv2v_new_key, key)
            _write_cache(attn._sv2v_new_value, value)
            key = torch.cat([key, attn._sv2v_cached_key], dim=1)
            value = torch.cat([value, attn._sv2v_cached_value], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Feature Fusion (decoder blocks only)
        if (is_self_attn and self._cache_enabled and self.use_feature_fusion
                and self.is_decoder_block and self.ff_strength > 0.0):
            _write_cache(attn._sv2v_new_output, hidden_states)
            nn_feats = _get_nn_feats(
                hidden_states, attn._sv2v_cached_output, threshold=self.ff_threshold
            )
            hidden_states = hidden_states * (1.0 - self.ff_strength) + self.ff_strength * nn_feats

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def _get_attn1_modules(unet):
    """Find all self-attention (attn1) modules and flag decoder-side ones."""
    modules = []
    for name, module in unet.named_modules():
        if name.endswith('.attn1') and hasattr(module, 'to_q'):
            is_decoder = any(b in name for b in ('up_blocks.0', 'up_blocks.1', 'mid_block'))
            modules.append((name, module, is_decoder))
    return modules


def _get_seq_len_for_module(unet, module_name, latent_h, latent_w):
    """Compute seq_len for a specific attn1 module by walking to its position."""
    # Strip _orig_mod. prefix added by torch.compile
    name = module_name.replace('_orig_mod.', '')
    h, w = latent_h, latent_w

    for i, block in enumerate(unet.down_blocks):
        prefix = f"down_blocks.{i}."
        if name.startswith(prefix):
            return h * w
        if hasattr(block, 'downsamplers') and block.downsamplers is not None:
            h //= 2
            w //= 2

    if name.startswith("mid_block."):
        return h * w

    for i, block in enumerate(unet.up_blocks):
        prefix = f"up_blocks.{i}."
        if name.startswith(prefix):
            return h * w
        if hasattr(block, 'upsamplers') and block.upsamplers is not None:
            h *= 2
            w *= 2

    logging.warning(f"[StreamV2V] Could not determine seq_len for {module_name} (stripped: {name}), using {h*w}")
    return h * w


def enable_cached_attention(unet, cache_maxframes=1, cache_interval=1,
                            use_feature_fusion=True, ff_strength=0.95, ff_threshold=0.95,
                            height=512, width=512, batch_size=1, device='cuda', dtype=torch.float16):
    """Install StreamV2V on a UNet with pre-allocated buffers (must run before torch.compile)."""
    attn1_modules = _get_attn1_modules(unet)

    vae_scale_factor = 8
    latent_h = height // vae_scale_factor
    latent_w = width // vae_scale_factor

    for name, module, is_decoder in attn1_modules:
        dim = module.to_k.in_features
        seq_len = _get_seq_len_for_module(unet, name, latent_h, latent_w)
        cache_seq = seq_len * cache_maxframes
        logging.debug(f"[StreamV2V] {name}: seq_len={seq_len}, dim={dim}")

        module.register_buffer('_sv2v_cached_key',
            torch.zeros(batch_size, cache_seq, dim, dtype=dtype, device=device))
        module.register_buffer('_sv2v_cached_value',
            torch.zeros(batch_size, cache_seq, dim, dtype=dtype, device=device))
        module.register_buffer('_sv2v_new_key',
            torch.zeros(batch_size, seq_len, dim, dtype=dtype, device=device))
        module.register_buffer('_sv2v_new_value',
            torch.zeros(batch_size, seq_len, dim, dtype=dtype, device=device))

        if is_decoder:
            out_dim = module.to_out[0].out_features
            module.register_buffer('_sv2v_cached_output',
                torch.zeros(batch_size, seq_len, out_dim, dtype=dtype, device=device))
            module.register_buffer('_sv2v_new_output',
                torch.zeros(batch_size, seq_len, out_dim, dtype=dtype, device=device))

    processors = unet.attn_processors
    new_processors = {}
    for proc_name, proc in processors.items():
        if "attn1" in proc_name:
            is_decoder = any(
                name in proc_name for name, _, dec in attn1_modules if dec
            )
            new_processors[proc_name] = StreamV2VAttnProcessor2_0(
                cache_enabled=True,
                is_decoder_block=is_decoder,
                use_feature_fusion=use_feature_fusion,
                ff_strength=ff_strength,
                ff_threshold=ff_threshold,
            )
        else:
            new_processors[proc_name] = proc

    unet.set_attn_processor(new_processors)

    unet._sv2v_config = {
        'cache_maxframes': cache_maxframes,
        'cache_interval': cache_interval,
        'frame_count': 0,
        'attn1_modules': attn1_modules,
    }

    logging.info(f"[StreamV2V] Installed on {len(attn1_modules)} layers with pre-allocated buffers "
                 f"(latent={latent_h}x{latent_w}, maxframes={cache_maxframes})")
    return len(attn1_modules)


def update_cache_after_unet(unet):
    """Copy new -> cached for each attn1. Call after each UNet forward (outside CUDA graph)."""
    config = getattr(unet, '_sv2v_config', None)
    if config is None:
        return

    config['frame_count'] += 1
    if config['frame_count'] % config['cache_interval'] != 0:
        return

    for name, module, is_decoder in config['attn1_modules']:
        if not hasattr(module, '_sv2v_cached_key'):
            continue

        if config['cache_maxframes'] == 1:
            module._sv2v_cached_key.copy_(module._sv2v_new_key)
            module._sv2v_cached_value.copy_(module._sv2v_new_value)
        else:
            seq_len = module._sv2v_new_key.shape[1]
            module._sv2v_cached_key[:, :-seq_len].copy_(module._sv2v_cached_key[:, seq_len:].clone())
            module._sv2v_cached_key[:, -seq_len:].copy_(module._sv2v_new_key)
            module._sv2v_cached_value[:, :-seq_len].copy_(module._sv2v_cached_value[:, seq_len:].clone())
            module._sv2v_cached_value[:, -seq_len:].copy_(module._sv2v_new_value)

        if is_decoder and hasattr(module, '_sv2v_cached_output'):
            module._sv2v_cached_output.copy_(module._sv2v_new_output)


def disable_cached_attention(unet):
    """Remove StreamV2V processors and buffers from the UNet."""
    processors = unet.attn_processors
    new_processors = {}
    for name, proc in processors.items():
        if isinstance(proc, StreamV2VAttnProcessor2_0):
            new_processors[name] = AttnProcessor2_0()
        else:
            new_processors[name] = proc
    unet.set_attn_processor(new_processors)

    for name, module, _ in _get_attn1_modules(unet):
        for buf_name in ['_sv2v_cached_key', '_sv2v_cached_value',
                         '_sv2v_new_key', '_sv2v_new_value',
                         '_sv2v_cached_output', '_sv2v_new_output']:
            if hasattr(module, buf_name):
                delattr(module, buf_name)

    if hasattr(unet, '_sv2v_config'):
        delattr(unet, '_sv2v_config')
    logging.info("[StreamV2V] Disabled")


def reset_attention_cache(unet):
    """Zero all caches."""
    config = getattr(unet, '_sv2v_config', None)
    if config is None:
        return
    config['frame_count'] = 0
    for name, module, _ in config['attn1_modules']:
        for buf_name in ['_sv2v_cached_key', '_sv2v_cached_value', '_sv2v_cached_output']:
            buf = getattr(module, buf_name, None)
            if buf is not None:
                buf.zero_()


# ============================================================================
# TensorRT path: kvo-cache passthrough utilities
# ============================================================================
# TRT needs the cache to flow through the engine as I/O tensors. The functions
# below port the upstream StreamV2V approach (Jeff-LiangF/streamv2v) where each
# attn1 layer's [key, value, output] is stacked into one
# (3, max_frames, B, seq, dim) tensor, exposed as named engine I/O.


def get_kvo_cache_info(unet, height=512, width=512):
    """Walk the UNet and record each attn1's (seq, dim).

    Returns: (kvo_cache_shapes, kvo_cache_structure, kvo_cache_count).
    Works for SD 1.5 and SDXL — no architecture-specific hardcoding.
    """
    latent_height = height // 8
    latent_width = width // 8

    kvo_cache_shapes = []
    kvo_cache_structure = []
    current_h, current_w = latent_height, latent_width

    for block in unet.down_blocks:
        if hasattr(block, 'attentions') and block.attentions is not None:
            attn_count = 0
            for attn_block in block.attentions:
                for transformer in attn_block.transformer_blocks:
                    attn = transformer.attn1
                    hidden_dim = attn.to_k.out_features
                    seq_length = current_h * current_w
                    kvo_cache_shapes.append((seq_length, hidden_dim))
                    attn_count += 1
            kvo_cache_structure.append(attn_count)

        if hasattr(block, 'downsamplers') and block.downsamplers is not None:
            current_h //= 2
            current_w //= 2

    if hasattr(unet.mid_block, 'attentions') and unet.mid_block.attentions is not None:
        # Upstream resets attn_count per inner attention block; only the last
        # block's count makes it into structure. SD 1.5 / SDXL mid_block has
        # exactly one attention block, so this matches — kept for parity.
        for attn_block in unet.mid_block.attentions:
            attn_count = 0
            for transformer in attn_block.transformer_blocks:
                attn = transformer.attn1
                hidden_dim = attn.to_k.out_features
                seq_length = current_h * current_w
                kvo_cache_shapes.append((seq_length, hidden_dim))
                attn_count += 1
            kvo_cache_structure.append(attn_count)

    for block in unet.up_blocks:
        if hasattr(block, 'attentions') and block.attentions is not None:
            attn_count = 0
            for attn_block in block.attentions:
                for transformer in attn_block.transformer_blocks:
                    attn = transformer.attn1
                    hidden_dim = attn.to_k.out_features
                    seq_length = current_h * current_w
                    kvo_cache_shapes.append((seq_length, hidden_dim))
                    attn_count += 1
            kvo_cache_structure.append(attn_count)

        if hasattr(block, 'upsamplers') and block.upsamplers is not None:
            current_h *= 2
            current_w *= 2

    kvo_cache_count = sum(kvo_cache_structure)
    return kvo_cache_shapes, kvo_cache_structure, kvo_cache_count


def create_kvo_cache(unet, batch_size, cache_maxframes, height=512, width=512,
                     device='cuda', dtype=torch.float16):
    """Allocate one (3, max_frames, B, seq, dim) tensor per attn1 layer."""
    kvo_cache_shapes, kvo_cache_structure, _ = get_kvo_cache_info(unet, height, width)
    kvo_cache = [
        torch.zeros(3, cache_maxframes, batch_size, seq, dim, dtype=dtype, device=device)
        for seq, dim in kvo_cache_shapes
    ]
    return kvo_cache, kvo_cache_structure


class KvoPassthroughAttnProcessor2_0:
    """StreamV2V self-attention with cache as engine I/O (TRT-compatible).

    Cache is passed in/out via side-channel attributes (_cache_in / _cache_out)
    set by the wrapper around the UNet forward. ONNX-traceable because the
    tensor ops between read and write are recorded by the tracer.

    Cache shape per attn1: (3, max_frames, B, seq, dim) — leading 3 stacks
    [key, value, output].
    """

    def __init__(self, name=None, is_decoder_block=False,
                 use_feature_injection=True,
                 feature_injection_strength=0.95,
                 feature_similarity_threshold=0.95,
                 max_frames=1):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("KvoPassthroughAttnProcessor2_0 requires PyTorch 2.0+.")
        self.name = name
        self.is_decoder_block = is_decoder_block
        self.use_feature_injection = use_feature_injection
        self.fi_strength = feature_injection_strength
        self.fi_threshold = feature_similarity_threshold
        self.max_frames = max_frames
        self._cache_in = None
        self._cache_out = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_selfattn = encoder_hidden_states is None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # _cache_in: (3, max_frames, B, seq, dim). Index 0=keys, 1=values, 2=outputs.
        cached_key = self._cache_in[0] if (is_selfattn and self._cache_in is not None) else None
        cached_value = self._cache_in[1] if (is_selfattn and self._cache_in is not None) else None
        cached_output = self._cache_in[2] if (is_selfattn and self._cache_in is not None) else None

        if is_selfattn:
            curr_key = key.clone()
            curr_value = value.clone()
            if cached_key is not None:
                # (max_frames, B, seq, dim) -> (B, max_frames*seq, dim) for concat on seq axis.
                cached_key_reshaped = cached_key.transpose(0, 1).contiguous().flatten(1, 2)
                cached_value_reshaped = cached_value.transpose(0, 1).contiguous().flatten(1, 2)
                key = torch.cat([curr_key, cached_key_reshaped], dim=1)
                value = torch.cat([curr_value, cached_value_reshaped], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Feature Injection (decoder blocks only).
        if (is_selfattn and self.is_decoder_block and self.use_feature_injection
                and cached_output is not None and self.fi_strength > 0.0):
            curr_output_for_fi = hidden_states.clone() if input_ndim != 4 else hidden_states
            cached_output_reshaped = cached_output.transpose(0, 1).contiguous().flatten(1, 2)
            nn_feats = _get_nn_feats(curr_output_for_fi, cached_output_reshaped, threshold=self.fi_threshold)
            hidden_states = hidden_states * (1.0 - self.fi_strength) + self.fi_strength * nn_feats

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor

        # Build cache_out for self-attention (cross-attention has no useful cache).
        if is_selfattn:
            if input_ndim == 4:
                curr_output = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            else:
                curr_output = hidden_states
            if self.max_frames == 1 or cached_key is None:
                new_cached_key = curr_key.unsqueeze(0)
                new_cached_value = curr_value.unsqueeze(0)
                new_cached_output = curr_output.unsqueeze(0)
            else:
                new_cached_key = torch.cat([cached_key[1:], curr_key.unsqueeze(0)], dim=0)
                new_cached_value = torch.cat([cached_value[1:], curr_value.unsqueeze(0)], dim=0)
                new_cached_output = torch.cat([cached_output[1:], curr_output.unsqueeze(0)], dim=0)
            self._cache_out = torch.stack([new_cached_key, new_cached_value, new_cached_output], dim=0)

        return hidden_states


def _get_unet_attn1_modules_in_order(unet):
    """Walk the UNet down/mid/up and return attn1 modules in the same order as get_kvo_cache_info."""
    modules = []
    for block in unet.down_blocks:
        if hasattr(block, 'attentions') and block.attentions is not None:
            for attn_block in block.attentions:
                for transformer in attn_block.transformer_blocks:
                    modules.append((transformer.attn1, False))
    if hasattr(unet.mid_block, 'attentions') and unet.mid_block.attentions is not None:
        for attn_block in unet.mid_block.attentions:
            for transformer in attn_block.transformer_blocks:
                modules.append((transformer.attn1, True))  # mid is decoder-like for FI
    for i, block in enumerate(unet.up_blocks):
        if hasattr(block, 'attentions') and block.attentions is not None:
            is_decoder = i < 2  # upstream gating
            for attn_block in block.attentions:
                for transformer in attn_block.transformer_blocks:
                    modules.append((transformer.attn1, is_decoder))
    return modules


def install_kvo_processors(unet, max_frames=1, use_feature_injection=True,
                           fi_strength=0.95, fi_threshold=0.95):
    """Install a KvoPassthroughAttnProcessor2_0 on each attn1; return the ordered list."""
    attn1_modules = _get_unet_attn1_modules_in_order(unet)
    processors = []
    for i, (attn_module, is_decoder) in enumerate(attn1_modules):
        proc = KvoPassthroughAttnProcessor2_0(
            name=f"attn1_{i}",
            is_decoder_block=is_decoder,
            use_feature_injection=use_feature_injection,
            feature_injection_strength=fi_strength,
            feature_similarity_threshold=fi_threshold,
            max_frames=max_frames,
        )
        attn_module.set_processor(proc)
        processors.append(proc)
    return processors


class TorchUNetKvoWrapper(torch.nn.Module):
    """ONNX-export wrapper that threads kvo cache as engine I/O.

    forward takes ``*kvo_cache_in`` (one tensor per attn1) and returns
    ``(model_pred, *kvo_cache_out)``. Tensor ops between bind (set _cache_in)
    and collect (read _cache_out) make the trace fully connected.
    """

    def __init__(self, unet: torch.nn.Module, processors_in_order):
        super().__init__()
        self.unet = unet
        # Plain Python list — processors aren't nn.Modules; the modules they're
        # attached to live inside self.unet and are already tracked.
        self._kvo_processors = processors_in_order

    def forward(self, sample, timestep, encoder_hidden_states, *kvo_cache_in):
        for proc, c in zip(self._kvo_processors, kvo_cache_in):
            proc._cache_in = c
        model_pred = self.unet(
            sample, timestep, encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]
        kvo_cache_out = tuple(proc._cache_out for proc in self._kvo_processors)
        return (model_pred,) + kvo_cache_out
