from typing import *

import torch
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTinyOutput
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.autoencoders.vae import DecoderOutput
from polygraphy import cuda

from .utilities import Engine


class UNet2DConditionModelEngine:
    def __init__(self, filepath: str, stream: cuda.Stream, use_cuda_graph: bool = False,
                 v2v_cache_maxframes: int = 1):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph

        self.engine.load()
        self.engine.activate()

        self._buffers_allocated = False
        # Cheap scalar signature for stable-shape skip of the full shape_dict rebuild.
        self._cached_shapes_sig = None

        # Cached zero CN residuals keyed by (batch, h, w, dtype). Reused
        # across frames so input pointers stay stable — required for CUDA
        # Graph replay on the CN-enabled path.
        self._zero_residuals_cache = None
        self._zero_residuals_key = None
        # Latch: once the zero DOWN residuals are copied into the engine's
        # input buffers, they persist there between calls (the buffers are
        # only reallocated on shape change). So on steady-state CN-disabled
        # frames we can skip re-copying ~10 MB of zero->zero down residuals.
        # Reset whenever the buffers/cache are (re)built or a real-residual
        # frame overwrites the down buffers (see __call__).
        self._zeros_staged = False

        # StreamV2V auto-detection. ``self.engine.tensors`` is empty until
        # ``allocate_buffers()`` runs, so probe binding names off the
        # underlying TRT engine at __init__ time.
        binding_names = {
            self.engine.engine.get_tensor_name(i)
            for i in range(self.engine.engine.num_io_tensors)
        }
        # Two cache layouts: legacy (kvo_in_i / kvo_out_i, (3, F, B, seq, dim), shifted
        # inside the engine and copied in/out every frame) and ring (kvo_in_i_j per cached
        # frame + kvo_out_i for the current frame, (3, B, seq, dim); the runtime keeps F+1
        # slot buffers and rotates the engine's addresses, nothing is copied).
        import re as _re
        _ring_in = _re.compile(r"^kvo_in_(\d+)_(\d+)$")
        ring_inputs = {n: _ring_in.match(n) for n in binding_names if _ring_in.match(n)}
        self._is_v2v_ring = bool(ring_inputs)
        self._is_v2v = self._is_v2v_ring or "kvo_in_0" in binding_names
        self._n_kvo = 0
        self._kvo_shapes_baked = None
        self._cache_maxframes = v2v_cache_maxframes
        # Sequential multi-step: one StreamV2V cache per denoising step (v2v_slot = step
        # index, set by the pipeline before each step), so every step attends to the
        # previous frame at its own noise level, as the stream batch does per batch slot.
        # Slot 0 is the only one used by single-step and stream-batch runs.
        self.v2v_slot = 0
        self._kvo_caches = {}      # legacy layout: slot -> [per-port cache]
        self._rings = {}           # ring layout: slot -> [F+1 ring slots][per-port buffer]
        self._ring_phases = {}     # ring layout: slot -> phase
        self._kvo_names = ()
        if self._is_v2v_ring:
            while f"kvo_out_{self._n_kvo}" in binding_names:
                self._n_kvo += 1
            # Blocks that never read the cache (attn_decoder_only) have no kvo_in ports:
            # the frame count comes from the highest cached-frame index present.
            self._cache_maxframes = 1 + max(int(m.group(2)) for m in ring_inputs.values())
            self._kvo_shapes_baked = []
            for i in range(self._n_kvo):
                shp = tuple(self.engine.engine.get_tensor_shape(f"kvo_out_{i}"))
                self._kvo_shapes_baked.append((shp[2], shp[3]))
            self._kvo_names = tuple(
                [n for n in (f"kvo_in_{i}_{j}" for i in range(self._n_kvo) for j in range(self._cache_maxframes))
                 if n in binding_names]
                + [f"kvo_out_{i}" for i in range(self._n_kvo)]
            )
            self._kvo_present = set(self._kvo_names)
            import logging
            logging.info(
                f"[TensorRT Engine] StreamV2V (v2v ring) mode: {self._n_kvo} attn1 ports x "
                f"{self._cache_maxframes} cached frames, kvo_shapes={self._kvo_shapes_baked}"
            )
        elif self._is_v2v:
            while f"kvo_in_{self._n_kvo}" in binding_names:
                self._n_kvo += 1
            # Read baked (seq, dim) from each port. Axes 1/2 (cache_maxframes,
            # batch) are dynamic; axes 0/3/4 (3, seq, dim) are fixed.
            self._kvo_shapes_baked = []
            for i in range(self._n_kvo):
                shp = tuple(self.engine.engine.get_tensor_shape(f"kvo_in_{i}"))
                self._kvo_shapes_baked.append((shp[3], shp[4]))
            import logging
            logging.info(
                f"[TensorRT Engine] StreamV2V (v2v) mode: {self._n_kvo} kvo ports, "
                f"cache_maxframes={self._cache_maxframes}, "
                f"kvo_shapes={self._kvo_shapes_baked}"
            )

        # Variant detection + per-variant CN residual layout tables, computed
        # ONCE from the engine bindings (constants of the loaded engine)
        # instead of every __call__. binding_names is already probed above and
        # is equivalent to self.engine.tensors for membership of these ports.
        self._has_controlnet = "down_block_0" in binding_names
        self._is_sdxl_engine = "text_embeds" in binding_names
        if self._has_controlnet:
            if self._is_sdxl_engine:
                from .models import (
                    SDXL_CN_DOWN_CHANNELS,
                    SDXL_CN_DOWN_SPATIAL_DIVS,
                    SDXL_CN_NUM_DOWN,
                    SDXL_CN_MID_CHANNELS,
                    SDXL_CN_MID_SPATIAL_DIV,
                )
                self._cn_down_channels = SDXL_CN_DOWN_CHANNELS
                self._cn_down_spatial_divs = SDXL_CN_DOWN_SPATIAL_DIVS
                self._cn_num_down = SDXL_CN_NUM_DOWN
                self._cn_mid_channels = SDXL_CN_MID_CHANNELS
                self._cn_mid_div = SDXL_CN_MID_SPATIAL_DIV
            else:
                self._cn_down_channels = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
                self._cn_down_spatial_divs = [1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8]
                self._cn_num_down = 12
                self._cn_mid_channels = 1280
                self._cn_mid_div = 8

    def __call__(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
        **kwargs,
    ) -> Any:
        if timestep.dtype != torch.float32:
            timestep = timestep.float()

        sample_shape = latent_model_input.shape
        text_batch = encoder_hidden_states.shape[0]
        sig = (sample_shape, text_batch)

        if not self._buffers_allocated or self._cached_shapes_sig != sig:
            current_shapes = {
                "sample": sample_shape,
                "timestep": timestep.shape,
                "encoder_hidden_states": encoder_hidden_states.shape,
                "latent": sample_shape,
            }

            if self._is_v2v_ring:
                batch = sample_shape[0]
                for i, (seq, dim) in enumerate(self._kvo_shapes_baked):
                    for j in range(self._cache_maxframes):
                        current_shapes[f"kvo_in_{i}_{j}"] = (3, batch, seq, dim)
                    current_shapes[f"kvo_out_{i}"] = (3, batch, seq, dim)
            elif self._is_v2v:
                batch = sample_shape[0]
                for i, (seq, dim) in enumerate(self._kvo_shapes_baked):
                    kvo_shape = (3, self._cache_maxframes, batch, seq, dim)
                    current_shapes[f"kvo_in_{i}"] = kvo_shape
                    current_shapes[f"kvo_out_{i}"] = kvo_shape

            self.engine.allocate_buffers(
                shape_dict=current_shapes,
                device=latent_model_input.device,
                external=self._kvo_names,
            )
            self._buffers_allocated = True
            self._cached_shapes_sig = sig
            # Fresh buffers → previously-staged zero residuals are gone.
            self._zeros_staged = False
            # Batch changed → drop kvo cache so it re-inits at the new shape.
            if self._is_v2v:
                self._kvo_caches = {}
                self._rings = {}
                self._ring_phases = {}

        inputs = {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        # Detect engine variant from actual binding names.
        engine_has_controlnet = self._has_controlnet
        engine_is_sdxl = self._is_sdxl_engine

        if engine_has_controlnet:
            # Residual layout (constants of the loaded engine) cached in __init__.
            down_block_channels = self._cn_down_channels
            down_block_spatial_divs = self._cn_down_spatial_divs
            num_down_blocks = self._cn_num_down
            mid_channels = self._cn_mid_channels
            mid_div = self._cn_mid_div

            batch_size = latent_model_input.shape[0]
            latent_h = latent_model_input.shape[2]
            latent_w = latent_model_input.shape[3]

            if down_block_additional_residuals is not None:
                for i, residual in enumerate(down_block_additional_residuals):
                    inputs[f"down_block_{i}"] = residual
                if mid_block_additional_residual is not None:
                    inputs["mid_block"] = mid_block_additional_residual
                else:
                    inputs["mid_block"] = torch.zeros(
                        batch_size, mid_channels, latent_h // mid_div, latent_w // mid_div,
                        dtype=latent_model_input.dtype,
                        device=latent_model_input.device
                    )
                # Real residuals overwrote the engine's down buffers — the
                # next CN-disabled frame must re-stage zeros into them.
                self._zeros_staged = False
            else:
                zero_key = (
                    batch_size, latent_h, latent_w,
                    latent_model_input.dtype,
                )
                if self._zero_residuals_cache is None or self._zero_residuals_key != zero_key:
                    cache = {}
                    for i in range(num_down_blocks):
                        h = latent_h // down_block_spatial_divs[i]
                        w = latent_w // down_block_spatial_divs[i]
                        cache[f"down_block_{i}"] = torch.zeros(
                            batch_size, down_block_channels[i], h, w,
                            dtype=latent_model_input.dtype,
                            device=latent_model_input.device,
                        )
                    cache["mid_block"] = torch.zeros(
                        batch_size, mid_channels,
                        latent_h // mid_div, latent_w // mid_div,
                        dtype=latent_model_input.dtype,
                        device=latent_model_input.device,
                    )
                    self._zero_residuals_cache = cache
                    self._zero_residuals_key = zero_key
                    # New zero tensors → not yet copied into engine buffers.
                    self._zeros_staged = False

                # The zero DOWN residuals are identical every frame; once
                # copied into the engine buffers they persist, so re-feed them
                # only when not already staged (saves ~10 MB of zero->zero D2D
                # copies/frame on steady-state CN-disabled frames). The MID
                # port is a single cheap tensor and may occasionally carry a
                # real residual, so always feed it for correctness.
                if not self._zeros_staged:
                    for i in range(num_down_blocks):
                        inputs[f"down_block_{i}"] = self._zero_residuals_cache[f"down_block_{i}"]
                    self._zeros_staged = True

                if mid_block_additional_residual is not None:
                    inputs["mid_block"] = mid_block_additional_residual
                else:
                    inputs["mid_block"] = self._zero_residuals_cache["mid_block"]

        if engine_is_sdxl:
            added_cond_kwargs = kwargs.get("added_cond_kwargs", {})
            if added_cond_kwargs:
                if "text_embeds" in added_cond_kwargs:
                    inputs["text_embeds"] = added_cond_kwargs["text_embeds"]
                if "time_ids" in added_cond_kwargs:
                    inputs["time_ids"] = added_cond_kwargs["time_ids"]

        graph_key = None
        kvo_cache = None
        if self._is_v2v_ring:
            n_slots = self._cache_maxframes + 1
            step = self.v2v_slot
            ring = self._rings.get(step)
            if ring is None:
                batch = latent_model_input.shape[0]
                ring = [
                    [
                        torch.zeros(3, batch, seq, dim, dtype=latent_model_input.dtype,
                                    device=latent_model_input.device)
                        for (seq, dim) in self._kvo_shapes_baked
                    ]
                    for _ in range(n_slots)
                ]
                self._rings[step] = ring
                self._ring_phases[step] = 0
            # Phase p: cached frames read from slots p..p+F-1 (oldest first), the current
            # frame written to slot p+F; next phase the oldest slot becomes the free one.
            # Addresses are baked into one CUDA graph per (step, phase), so binding only
            # happens until each one has been captured (or every call without graphs).
            phase = self._ring_phases[step]
            graph_key = phase if step == 0 else (step, phase)
            if not (self.use_cuda_graph and graph_key in self.engine.graphs):
                for i in range(self._n_kvo):
                    for j in range(self._cache_maxframes):
                        name = f"kvo_in_{i}_{j}"
                        if name in self._kvo_present:
                            self.engine.bind_external(name, ring[(phase + j) % n_slots][i])
                    self.engine.bind_external(f"kvo_out_{i}", ring[(phase + self._cache_maxframes) % n_slots][i])
            self._ring_phases[step] = (phase + 1) % n_slots
        elif self._is_v2v:
            kvo_cache = self._kvo_caches.get(self.v2v_slot)
            if kvo_cache is None:
                batch = latent_model_input.shape[0]
                kvo_cache = [
                    torch.zeros(
                        3, self._cache_maxframes, batch, seq, dim,
                        dtype=latent_model_input.dtype,
                        device=latent_model_input.device,
                    )
                    for (seq, dim) in self._kvo_shapes_baked
                ]
                self._kvo_caches[self.v2v_slot] = kvo_cache
            for i in range(self._n_kvo):
                inputs[f"kvo_in_{i}"] = kvo_cache[i]

        engine_outputs = self.engine.infer(
            inputs,
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
            graph_key=graph_key,
        )
        noise_pred = engine_outputs["latent"]

        # Legacy layout: copy kvo outputs into the local cache — the engine reuses its
        # output buffers, so without a copy the next call would race.
        if kvo_cache is not None:
            for i in range(self._n_kvo):
                kvo_cache[i].copy_(engine_outputs[f"kvo_out_{i}"])

        return UNet2DConditionOutput(sample=noise_pred)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class ControlNetEngine:
    """TensorRT engine for ControlNet inference (12 down + 1 mid outputs)."""
    def __init__(self, filepath: str, stream: cuda.Stream, use_cuda_graph: bool = False):
        self.engine = Engine(filepath)
        self.stream = stream
        self.use_cuda_graph = use_cuda_graph

        self.engine.load()
        self.engine.activate()

        self._buffers_allocated = False
        self._cached_shapes_sig = None

    def __call__(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        **kwargs,
    ):
        if timestep.dtype != torch.float32:
            timestep = timestep.float()

        sig = (latent_model_input.shape, encoder_hidden_states.shape[0],
               controlnet_cond.shape)
        if not self._buffers_allocated or self._cached_shapes_sig != sig:
            current_shapes = {
                "sample": latent_model_input.shape,
                "timestep": timestep.shape,
                "encoder_hidden_states": encoder_hidden_states.shape,
                "controlnet_cond": controlnet_cond.shape,
            }
            self.engine.allocate_buffers(
                shape_dict=current_shapes,
                device=latent_model_input.device,
            )
            self._buffers_allocated = True
            self._cached_shapes_sig = sig

        outputs = self.engine.infer(
            {
                "sample": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "controlnet_cond": controlnet_cond,
            },
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )

        down_block_res_samples = tuple(outputs[f"down_block_{i}"] for i in range(12))
        mid_block_res_sample = outputs["mid_block"]

        # Apply conditioning scale. The pipeline passes the scale as a cached
        # 0-d CUDA tensor (to avoid torch.compile recompiles); evaluating
        # `tensor != 1.0` inside an `if` would force a per-CN/frame GPU->CPU
        # sync. Short-circuit on isinstance so a tensor scale never hits
        # bool() — when it's a tensor we always scale (a redundant *1.0 is
        # harmless and far cheaper than a host stall). Multiply stays
        # out-of-place: the engine outputs are graph-owned buffers.
        if isinstance(conditioning_scale, torch.Tensor) or conditioning_scale != 1.0:
            down_block_res_samples = tuple(sample * conditioning_scale for sample in down_block_res_samples)
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        return down_block_res_samples, mid_block_res_sample

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class ControlNetUnionEngine:
    """TensorRT engine for the SDXL Union ControlNet (9 down + 1 mid outputs).

    One engine per ordered control-type tuple (baked at export). Conditioning
    scales are a runtime (N,) input; timestep is float32 like ControlNetEngine.
    Call signature mirrors what ``UnionControlNetWrapper`` computes per frame.
    """
    NUM_DOWN = 9

    def __init__(self, filepath: str, stream: cuda.Stream, control_type_idx, use_cuda_graph: bool = False):
        self.engine = Engine(filepath)
        self.stream = stream
        self.control_type_idx = tuple(int(i) for i in control_type_idx)
        self.num_conds = len(self.control_type_idx)
        self.use_cuda_graph = use_cuda_graph

        self.engine.load()
        self.engine.activate()

        self._buffers_allocated = False
        self._cached_shapes_sig = None

    def __call__(
        self,
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond,
        control_type: torch.Tensor,
        conditioning_scale,
        text_embeds: torch.Tensor,
        time_ids: torch.Tensor,
        **kwargs,
    ):
        batch = sample.shape[0]
        dev = sample.device

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([float(timestep)] * batch, device=dev)
        if timestep.dim() == 0:
            timestep = timestep.reshape(1)
        if timestep.dtype != torch.float32:
            timestep = timestep.float()
        if timestep.shape[0] != batch:
            timestep = timestep.expand(batch)

        if not isinstance(controlnet_cond, (list, tuple)):
            controlnet_cond = [controlnet_cond]
        if len(controlnet_cond) != self.num_conds:
            raise ValueError(
                f"ControlNetUnionEngine: got {len(controlnet_cond)} conditioning images "
                f"but engine was built for {self.num_conds} control types {self.control_type_idx}."
            )
        conds = []
        for c in controlnet_cond:
            if c.dim() == 3:
                c = c.unsqueeze(0)
            if c.shape[0] != batch:
                c = c.expand(batch, -1, -1, -1)
            conds.append(c)

        n = self.num_conds
        if isinstance(conditioning_scale, torch.Tensor):
            scales = conditioning_scale.reshape(-1)
            if scales.numel() == 1 and n > 1:
                scales = scales.expand(n)
        else:
            if not isinstance(conditioning_scale, (list, tuple)):
                conditioning_scale = [conditioning_scale] * n
            scales = torch.tensor(
                [float(s) for s in conditioning_scale], device=dev, dtype=torch.float16
            )

        if control_type.dim() == 1:
            control_type = control_type.unsqueeze(0)
        if control_type.shape[0] != batch:
            control_type = control_type.expand(batch, -1)
        if text_embeds.shape[0] != batch:
            text_embeds = text_embeds.expand(batch, -1)
        if time_ids.shape[0] != batch:
            time_ids = time_ids.expand(batch, -1)

        feed = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "text_embeds": text_embeds,
            "time_ids": time_ids,
            "control_type": control_type,
            "conditioning_scale": scales,
        }
        for i, c in enumerate(conds):
            feed[f"cond_{i}"] = c

        sig = (
            tuple(sample.shape), tuple(encoder_hidden_states.shape),
            tuple(tuple(c.shape) for c in conds),
        )
        if not self._buffers_allocated or self._cached_shapes_sig != sig:
            self.engine.allocate_buffers(
                shape_dict={k: tuple(v.shape) for k, v in feed.items()},
                device=dev,
            )
            self._buffers_allocated = True
            self._cached_shapes_sig = sig

        outputs = self.engine.infer(feed, self.stream, use_cuda_graph=self.use_cuda_graph)

        # Scales are applied inside the engine (baked forward). Outputs are
        # graph-owned buffers: consumers must not hold them across frames.
        down_block_res_samples = [outputs[f"down_block_{i}"] for i in range(self.NUM_DOWN)]
        mid_block_res_sample = outputs["mid_block"]
        return down_block_res_samples, mid_block_res_sample

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class _DepthEstimatorOutput:
    """Minimal stand-in for HF ``DepthEstimatorOutput`` (only ``predicted_depth`` is used)."""
    __slots__ = ("predicted_depth",)

    def __init__(self, predicted_depth):
        self.predicted_depth = predicted_depth


class DepthAnythingEngine:
    """TensorRT runtime for Depth-Anything V2 (HF-API polymorphic)."""
    def __init__(self, filepath: str, stream: cuda.Stream, image_size: int, use_cuda_graph: bool = False):
        self.engine = Engine(filepath)
        self.stream = stream
        self.image_size = image_size
        self.use_cuda_graph = use_cuda_graph

        self.engine.load()
        self.engine.activate()

        self._buffers_allocated = False
        self._cached_shapes_sig = None
        # Lazy-init: avoid paying event allocation if engine is never called.
        self._pre_event = None
        self._post_event = None

    def __call__(self, pixel_values=None, **kwargs):
        if pixel_values is None:
            raise ValueError(
                "DepthAnythingEngine requires `pixel_values` (positional or keyword)."
            )

        sig = pixel_values.shape
        if not self._buffers_allocated or self._cached_shapes_sig != sig:
            self.engine.allocate_buffers(
                shape_dict={"pixel_values": tuple(sig)},
                device=pixel_values.device,
            )
            self._buffers_allocated = True
            self._cached_shapes_sig = sig

        # Cross-stream coordination via CUDA events (host-non-blocking):
        # depth preprocessor writes pixel_values on its torch stream, TRT
        # reads on the polygraphy stream, downstream torch ops read the
        # output back on the torch stream.
        from .utilities import cudart, CUASSERT

        if self._pre_event is None:
            self._pre_event = CUASSERT(cudart.cudaEventCreateWithFlags(
                cudart.cudaEventDisableTiming
            ))
            self._post_event = CUASSERT(cudart.cudaEventCreateWithFlags(
                cudart.cudaEventDisableTiming
            ))

        torch_stream_handle = torch.cuda.current_stream().cuda_stream

        CUASSERT(cudart.cudaEventRecord(self._pre_event, torch_stream_handle))
        CUASSERT(cudart.cudaStreamWaitEvent(self.stream.ptr, self._pre_event, 0))

        outputs = self.engine.infer(
            {"pixel_values": pixel_values},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )

        CUASSERT(cudart.cudaEventRecord(self._post_event, self.stream.ptr))
        CUASSERT(cudart.cudaStreamWaitEvent(torch_stream_handle, self._post_event, 0))

        return _DepthEstimatorOutput(outputs["predicted_depth"])

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)


class AutoencoderKLEngine:
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        stream: cuda.Stream,
        scaling_factor: int,
        use_cuda_graph: bool = False,
    ):
        self.encoder = Engine(encoder_path)
        self.decoder = Engine(decoder_path)
        self.stream = stream
        self.vae_scale_factor = scaling_factor
        self.use_cuda_graph = use_cuda_graph

        self.encoder.load()
        self.decoder.load()
        self.encoder.activate()
        self.decoder.activate()

        self._encoder_buffers_allocated = False
        self._decoder_buffers_allocated = False
        self._encoder_cached_shapes_sig = None
        self._decoder_cached_shapes_sig = None

    def encode(self, images: torch.Tensor, **kwargs):
        sig = images.shape
        if not self._encoder_buffers_allocated or self._encoder_cached_shapes_sig != sig:
            current_shapes = {
                "images": images.shape,
                "latent": (
                    images.shape[0],
                    4,
                    images.shape[2] // self.vae_scale_factor,
                    images.shape[3] // self.vae_scale_factor,
                ),
            }
            self.encoder.allocate_buffers(
                shape_dict=current_shapes,
                device=images.device,
            )
            self._encoder_buffers_allocated = True
            self._encoder_cached_shapes_sig = sig
        latents = self.encoder.infer(
            {"images": images},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["latent"]
        return AutoencoderTinyOutput(latents=latents)

    def decode(self, latent: torch.Tensor, **kwargs):
        sig = latent.shape
        if not self._decoder_buffers_allocated or self._decoder_cached_shapes_sig != sig:
            current_shapes = {
                "latent": latent.shape,
                "images": (
                    latent.shape[0],
                    3,
                    latent.shape[2] * self.vae_scale_factor,
                    latent.shape[3] * self.vae_scale_factor,
                ),
            }
            self.decoder.allocate_buffers(
                shape_dict=current_shapes,
                device=latent.device,
            )
            self._decoder_buffers_allocated = True
            self._decoder_cached_shapes_sig = sig

        images = self.decoder.infer(
            {"latent": latent},
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )["images"]

        # TRT VAE decoder does NOT clamp to [-1, 1] like torch's tanh head.
        # Without this, denormalize() in downstream image_utils produces
        # oversaturated / corrupted output.
        # In-place: `images` is the engine's own output buffer, consumed by
        # denormalize() before the next decode overwrites it — equivalent to
        # the existing buffer-reuse semantics, minus a full-image alloc/frame.
        images.clamp_(-1.0, 1.0)

        return DecoderOutput(sample=images)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass
