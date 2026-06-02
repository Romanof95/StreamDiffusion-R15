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

        # StreamV2V auto-detection. ``self.engine.tensors`` is empty until
        # ``allocate_buffers()`` runs, so probe binding names off the
        # underlying TRT engine at __init__ time.
        binding_names = {
            self.engine.engine.get_tensor_name(i)
            for i in range(self.engine.engine.num_io_tensors)
        }
        self._is_v2v = "kvo_in_0" in binding_names
        self._kvo_cache = None
        self._n_kvo = 0
        self._kvo_shapes_baked = None
        self._cache_maxframes = v2v_cache_maxframes
        if self._is_v2v:
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

            if self._is_v2v:
                batch = sample_shape[0]
                for i, (seq, dim) in enumerate(self._kvo_shapes_baked):
                    kvo_shape = (3, self._cache_maxframes, batch, seq, dim)
                    current_shapes[f"kvo_in_{i}"] = kvo_shape
                    current_shapes[f"kvo_out_{i}"] = kvo_shape

            self.engine.allocate_buffers(
                shape_dict=current_shapes,
                device=latent_model_input.device,
            )
            self._buffers_allocated = True
            self._cached_shapes_sig = sig
            # Batch changed → drop kvo cache so it re-inits at the new shape.
            if self._is_v2v:
                self._kvo_cache = None

        inputs = {
            "sample": latent_model_input,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

        # Detect engine variant from actual binding names.
        engine_has_controlnet = "down_block_0" in self.engine.tensors
        engine_is_sdxl = "text_embeds" in self.engine.tensors

        if engine_has_controlnet:
            if engine_is_sdxl:
                from .models import (
                    SDXL_CN_DOWN_CHANNELS as down_block_channels,
                    SDXL_CN_DOWN_SPATIAL_DIVS as down_block_spatial_divs,
                    SDXL_CN_NUM_DOWN as num_down_blocks,
                    SDXL_CN_MID_CHANNELS as mid_channels,
                    SDXL_CN_MID_SPATIAL_DIV as mid_div,
                )
            else:
                down_block_channels = [320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
                down_block_spatial_divs = [1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8]
                num_down_blocks = 12
                mid_channels = 1280
                mid_div = 8

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

                for i in range(num_down_blocks):
                    inputs[f"down_block_{i}"] = self._zero_residuals_cache[f"down_block_{i}"]

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

        if self._is_v2v:
            if self._kvo_cache is None:
                batch = latent_model_input.shape[0]
                self._kvo_cache = [
                    torch.zeros(
                        3, self._cache_maxframes, batch, seq, dim,
                        dtype=latent_model_input.dtype,
                        device=latent_model_input.device,
                    )
                    for (seq, dim) in self._kvo_shapes_baked
                ]
            for i in range(self._n_kvo):
                inputs[f"kvo_in_{i}"] = self._kvo_cache[i]

        engine_outputs = self.engine.infer(
            inputs,
            self.stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        noise_pred = engine_outputs["latent"]

        # Copy kvo outputs into the local cache — the engine reuses its
        # output buffers, so without a copy the next call would race.
        if self._is_v2v:
            for i in range(self._n_kvo):
                self._kvo_cache[i].copy_(engine_outputs[f"kvo_out_{i}"])

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

        if conditioning_scale != 1.0:
            down_block_res_samples = tuple(sample * conditioning_scale for sample in down_block_res_samples)
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

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
        images = images.clamp(-1.0, 1.0)

        return DecoderOutput(sample=images)

    def to(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass
