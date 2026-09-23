import gc
import os

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from polygraphy import cuda

from ...pipeline import StreamDiffusion
from ..engine_cache import engine_ready
from .builder import EngineBuilder, create_onnx_path
from .engine import AutoencoderKLEngine, UNet2DConditionModelEngine, ControlNetEngine, ControlNetUnionEngine, DepthAnythingEngine
from .models import VAE, BaseModel, UNet, UNetXL, UNetSimple, UNetXLSimple, VAEEncoder, ControlNet, ControlNetUnion, DepthAnything


class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor):
        return retrieve_latents(self.vae.encode(x))


class TorchControlNetWrapper(torch.nn.Module):
    """Wraps ControlNet so its tuple output becomes 12 down + 1 mid named tensors (TRT-friendly)."""
    def __init__(self, controlnet: ControlNetModel):
        super().__init__()
        self.controlnet = controlnet

    def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond):
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )
        return (*down_block_res_samples, mid_block_res_sample)



class TorchControlNetUnionWrapper(torch.nn.Module):
    """SDXL Union ControlNet export wrapper: N conds as separate named inputs,
    control_type_idx baked at export, 9 down + 1 mid named outputs."""
    def __init__(self, controlnet_union, control_type_idx):
        super().__init__()
        self.controlnet = controlnet_union
        self.control_type_idx = [int(i) for i in control_type_idx]

    def forward(
        self, sample, timestep, encoder_hidden_states, text_embeds, time_ids,
        control_type, conditioning_scale,
        cond_0=None, cond_1=None, cond_2=None, cond_3=None, cond_4=None, cond_5=None,
    ):
        conds = [c for c in (cond_0, cond_1, cond_2, cond_3, cond_4, cond_5) if c is not None]
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=conds,
            control_type=control_type,
            control_type_idx=self.control_type_idx,
            conditioning_scale=conditioning_scale,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )
        return (*down_block_res_samples, mid_block_res_sample)


class TorchUNetWrapper(torch.nn.Module):
    """SD 1.5/2.1 UNet wrapper: accepts 12 down + 1 mid CN residuals as separate inputs."""
    def __init__(self, unet: UNet2DConditionModel, is_sdxl: bool = False):
        super().__init__()
        self.unet = unet
        self.is_sdxl = is_sdxl

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        down_block_0=None,
        down_block_1=None,
        down_block_2=None,
        down_block_3=None,
        down_block_4=None,
        down_block_5=None,
        down_block_6=None,
        down_block_7=None,
        down_block_8=None,
        down_block_9=None,
        down_block_10=None,
        down_block_11=None,
        mid_block=None,
        text_embeds=None,
        time_ids=None,
    ):
        down_block_additional_residuals = None
        if down_block_0 is not None:
            down_block_additional_residuals = (
                down_block_0,
                down_block_1,
                down_block_2,
                down_block_3,
                down_block_4,
                down_block_5,
                down_block_6,
                down_block_7,
                down_block_8,
                down_block_9,
                down_block_10,
                down_block_11,
            )

        added_cond_kwargs = None
        if self.is_sdxl and text_embeds is not None and time_ids is not None:
            added_cond_kwargs = {
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            }

        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]


class TorchUNetXLControlNetWrapper(torch.nn.Module):
    """SDXL UNet wrapper: 9 CN down residuals + mid + SDXL added_cond_kwargs."""
    def __init__(self, unet: UNet2DConditionModel):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        down_block_0=None,
        down_block_1=None,
        down_block_2=None,
        down_block_3=None,
        down_block_4=None,
        down_block_5=None,
        down_block_6=None,
        down_block_7=None,
        down_block_8=None,
        mid_block=None,
        text_embeds=None,
        time_ids=None,
    ):
        down_block_additional_residuals = None
        if down_block_0 is not None:
            down_block_additional_residuals = (
                down_block_0, down_block_1, down_block_2,
                down_block_3, down_block_4, down_block_5,
                down_block_6, down_block_7, down_block_8,
            )
        added_cond_kwargs = None
        if text_embeds is not None and time_ids is not None:
            added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]


class TorchUNetV2VWrapper(torch.nn.Module):
    """SD 1.5 UNet ONNX wrapper: 12 CN residuals + StreamV2V kvo cache as engine I/O.

    kvo cache is threaded via side-channel attributes (``proc._cache_in`` /
    ``proc._cache_out``) on each attn1 processor — the wrapper binds inputs
    before forward, collects outputs after. ``kvo_processors`` MUST be the
    ordered list from ``install_kvo_processors(unet, ...)``.
    """
    def __init__(self, unet: UNet2DConditionModel, kvo_processors):
        super().__init__()
        self.unet = unet
        self._kvo_processors = list(kvo_processors)

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        down_block_0, down_block_1, down_block_2, down_block_3,
        down_block_4, down_block_5, down_block_6, down_block_7,
        down_block_8, down_block_9, down_block_10, down_block_11,
        mid_block,
        *kvo_cache_in,
    ):
        # Bind cache_in BEFORE forward — processors read it during attention.
        for proc, c in zip(self._kvo_processors, kvo_cache_in):
            proc._cache_in = c

        down_block_additional_residuals = (
            down_block_0, down_block_1, down_block_2,
            down_block_3, down_block_4, down_block_5,
            down_block_6, down_block_7, down_block_8,
            down_block_9, down_block_10, down_block_11,
        )

        model_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block,
            return_dict=False,
        )[0]

        kvo_cache_out = tuple(proc._cache_out for proc in self._kvo_processors)
        return (model_pred,) + kvo_cache_out


class TorchUNetV2VRingWrapper(torch.nn.Module):
    """SD 1.5 UNet ONNX wrapper: 12 CN residuals + StreamV2V ring cache (``max_frames``
    inputs per attn1, one output per attn1; see KvoRingAttnProcessor2_0)."""
    def __init__(self, unet: UNet2DConditionModel, kvo_processors, max_frames: int):
        super().__init__()
        self.unet = unet
        self._kvo_processors = list(kvo_processors)
        self._max_frames = int(max_frames)

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        down_block_0, down_block_1, down_block_2, down_block_3,
        down_block_4, down_block_5, down_block_6, down_block_7,
        down_block_8, down_block_9, down_block_10, down_block_11,
        mid_block,
        *kvo_cache_in,
    ):
        n = self._max_frames
        for k, proc in enumerate(self._kvo_processors):
            proc._cache_in = tuple(kvo_cache_in[k * n:(k + 1) * n])

        down_block_additional_residuals = (
            down_block_0, down_block_1, down_block_2,
            down_block_3, down_block_4, down_block_5,
            down_block_6, down_block_7, down_block_8,
            down_block_9, down_block_10, down_block_11,
        )

        model_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block,
            return_dict=False,
        )[0]

        return (model_pred,) + tuple(proc._cache_out for proc in self._kvo_processors)


class TorchUNetXLV2VWrapper(torch.nn.Module):
    """SDXL UNet ONNX wrapper: 9 CN residuals + SDXL conditioning + StreamV2V kvo cache."""
    def __init__(self, unet: UNet2DConditionModel, kvo_processors):
        super().__init__()
        self.unet = unet
        self._kvo_processors = list(kvo_processors)

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        down_block_0, down_block_1, down_block_2,
        down_block_3, down_block_4, down_block_5,
        down_block_6, down_block_7, down_block_8,
        mid_block,
        text_embeds,
        time_ids,
        *kvo_cache_in,
    ):
        for proc, c in zip(self._kvo_processors, kvo_cache_in):
            proc._cache_in = c

        down_block_additional_residuals = (
            down_block_0, down_block_1, down_block_2,
            down_block_3, down_block_4, down_block_5,
            down_block_6, down_block_7, down_block_8,
        )
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}

        model_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        kvo_cache_out = tuple(proc._cache_out for proc in self._kvo_processors)
        return (model_pred,) + kvo_cache_out


class TorchUNetXLV2VRingWrapper(torch.nn.Module):
    """SDXL UNet ONNX wrapper: 9 CN residuals + SDXL conditioning + StreamV2V ring cache
    (``max_frames`` inputs per attn1, one output per attn1; see KvoRingAttnProcessor2_0)."""
    def __init__(self, unet: UNet2DConditionModel, kvo_processors, max_frames: int):
        super().__init__()
        self.unet = unet
        self._kvo_processors = list(kvo_processors)
        self._max_frames = int(max_frames)

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        down_block_0, down_block_1, down_block_2,
        down_block_3, down_block_4, down_block_5,
        down_block_6, down_block_7, down_block_8,
        mid_block,
        text_embeds,
        time_ids,
        *kvo_cache_in,
    ):
        n = self._max_frames
        for k, proc in enumerate(self._kvo_processors):
            proc._cache_in = tuple(kvo_cache_in[k * n:(k + 1) * n])

        down_block_additional_residuals = (
            down_block_0, down_block_1, down_block_2,
            down_block_3, down_block_4, down_block_5,
            down_block_6, down_block_7, down_block_8,
        )
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}

        model_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        return (model_pred,) + tuple(proc._cache_out for proc in self._kvo_processors)


class TorchUNetWrapperSimple(torch.nn.Module):
    """UNet wrapper without ControlNet (SD or SDXL)."""
    def __init__(self, unet: UNet2DConditionModel, is_sdxl: bool = False):
        super().__init__()
        self.unet = unet
        self.is_sdxl = is_sdxl

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        text_embeds=None,
        time_ids=None,
    ):
        added_cond_kwargs = None
        if self.is_sdxl and text_embeds is not None and time_ids is not None:
            added_cond_kwargs = {
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            }

        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]


def compile_vae_encoder(
    vae: TorchVAEEncoder,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    opt_image_height: int = 512,
    opt_image_width: int = 512,
    engine_build_options: dict = {},
):
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        opt_image_height=opt_image_height,
        opt_image_width=opt_image_width,
        **engine_build_options,
    )


def compile_vae_decoder(
    vae: AutoencoderKL,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    opt_image_height: int = 512,
    opt_image_width: int = 512,
    engine_build_options: dict = {},
):
    vae = vae.to(torch.device("cuda"))
    builder = EngineBuilder(model_data, vae, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        opt_image_height=opt_image_height,
        opt_image_width=opt_image_width,
        **engine_build_options,
    )


def compile_unet(
    unet: UNet2DConditionModel,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    opt_image_height: int = 512,
    opt_image_width: int = 512,
    engine_build_options: dict = {},
    is_sdxl: bool = False,
    use_simple_wrapper: bool = False,
    kvo_processors=None,
    kvo_ring_frames=None,
):
    if kvo_processors is not None:
        if is_sdxl and kvo_ring_frames:
            unet_wrapper = TorchUNetXLV2VRingWrapper(unet, kvo_processors, kvo_ring_frames).to(
                torch.device("cuda"), dtype=torch.float16
            )
        elif is_sdxl:
            unet_wrapper = TorchUNetXLV2VWrapper(unet, kvo_processors).to(
                torch.device("cuda"), dtype=torch.float16
            )
        elif kvo_ring_frames:
            unet_wrapper = TorchUNetV2VRingWrapper(unet, kvo_processors, kvo_ring_frames).to(
                torch.device("cuda"), dtype=torch.float16
            )
        else:
            unet_wrapper = TorchUNetV2VWrapper(unet, kvo_processors).to(
                torch.device("cuda"), dtype=torch.float16
            )
    elif use_simple_wrapper:
        unet_wrapper = TorchUNetWrapperSimple(unet, is_sdxl=is_sdxl).to(torch.device("cuda"), dtype=torch.float16)
    elif is_sdxl:
        unet_wrapper = TorchUNetXLControlNetWrapper(unet).to(torch.device("cuda"), dtype=torch.float16)
    else:
        unet_wrapper = TorchUNetWrapper(unet, is_sdxl=is_sdxl).to(torch.device("cuda"), dtype=torch.float16)

    builder = EngineBuilder(model_data, unet_wrapper, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        opt_image_height=opt_image_height,
        opt_image_width=opt_image_width,
        **engine_build_options,
    )


def compile_controlnet(
    controlnet: ControlNetModel,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    opt_image_height: int = 512,
    opt_image_width: int = 512,
    engine_build_options: dict = {},
):
    """Compile a ControlNet to a TRT engine (tuple output unpacked to named tensors).

    opt_image_height/width bake the static shape profile (build_dynamic_shape is
    off), so they MUST match the stream resolution — a 512-baked engine cannot
    serve a 1024 session. Mirrors compile_unet.
    """
    controlnet_wrapper = TorchControlNetWrapper(controlnet).to(torch.device("cuda"), dtype=torch.float16)

    builder = EngineBuilder(model_data, controlnet_wrapper, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        opt_image_height=opt_image_height,
        opt_image_width=opt_image_width,
        **engine_build_options,
    )


def compile_controlnet_union(
    controlnet_union,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    opt_batch_size: int = 1,
    opt_image_height: int = 1024,
    opt_image_width: int = 1024,
    engine_build_options: dict = {},
):
    """Compile the SDXL Union ControlNet to a TRT engine for the control-type
    tuple baked in ``model_data`` (a ``ControlNetUnion`` spec). Static shape
    profile: resolution must match the stream. Mirrors compile_controlnet.
    """
    wrapper = TorchControlNetUnionWrapper(controlnet_union, model_data.control_type_idx).to(
        torch.device("cuda"), dtype=torch.float16
    )

    builder = EngineBuilder(model_data, wrapper, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=opt_batch_size,
        opt_image_height=opt_image_height,
        opt_image_width=opt_image_width,
        **engine_build_options,
    )


def compile_unet_quantized(
    unet: UNet2DConditionModel,
    model_data: BaseModel,
    onnx_path: str,
    engine_path: str,
    opt_batch_size: int,
    opt_image_height: int,
    opt_image_width: int,
    precision: str,
    kvo_processors=None,
    kvo_ring_frames=None,
):
    """SDXL UNet (already quantized in place by quantization.quantize_model) -> static
    ONNX with TensorRT block-scaled quant ops -> strongly-typed engine. Same I/O as the
    fp16 ``unet_cn`` engine (ControlNet residual ports), so the runtime class is unchanged.
    With ``kvo_processors`` (StreamV2V, ``model_data`` = UNetXLV2V) the attention cache
    ports are exported too, same I/O as the fp16 ``unet_v2v_xl_mfN`` engine.
    """
    from ..quantization import export_quantized_onnx, build_strongly_typed_engine

    kvo_shapes = getattr(model_data, "kvo_cache_shapes", None)
    if os.path.exists(onnx_path) and os.path.exists(onnx_path + ".data"):
        import logging as _logging
        _logging.info(f"[Quant] reusing exported ONNX: {onnx_path}")
    else:
        if kvo_processors is not None and kvo_ring_frames:
            wrapper = TorchUNetXLV2VRingWrapper(unet, kvo_processors, kvo_ring_frames)
        elif kvo_processors is not None:
            wrapper = TorchUNetXLV2VWrapper(unet, kvo_processors)
        else:
            wrapper = TorchUNetXLControlNetWrapper(unet)
        # The UNetXL spec emits 2*batch rows ("2B" axis: dim 0; dim 1 for the ring cache
        # (3, 2B, seq, dim); dim 2 for the legacy 5-D cache); the static graph is built at batch.
        def _at_batch(name, t):
            if t.dim() == 5:
                return t[:, :, :opt_batch_size]
            if name.startswith("kvo_in"):
                return t[:, :opt_batch_size]
            return t[:opt_batch_size]
        inputs = tuple(
            _at_batch(n, t) for n, t in zip(
                model_data.get_input_names(),
                model_data.get_sample_input(opt_batch_size, opt_image_height, opt_image_width),
            )
        )
        export_quantized_onnx(
            wrapper, unet, inputs, model_data.get_input_names(), model_data.get_output_names(),
            onnx_path, precision,
        )
        del wrapper, inputs
    try:
        unet.to("cpu")
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    profile = model_data.get_input_profile(
        opt_batch_size, opt_image_height, opt_image_width, static_batch=False, static_shape=True,
    )
    if kvo_shapes is not None and kvo_ring_frames:
        for i, (seq, dim) in enumerate(kvo_shapes):
            for j in range(kvo_ring_frames):
                shape = (3, opt_batch_size, seq, dim)
                profile[f"kvo_in_{i}_{j}"] = [shape, shape, shape]
    elif kvo_shapes is not None:
        # The graph is static: pin the kvo ports (the spec profile allows 1..max_cache_frames).
        for i, (seq, dim) in enumerate(kvo_shapes):
            shape = (3, model_data.max_cache_frames, opt_batch_size, seq, dim)
            profile[f"kvo_in_{i}"] = [shape, shape, shape]
    build_strongly_typed_engine(onnx_path, engine_path, profile)


def compile_controlnet_union_quantized(
    controlnet_union,
    model_data: BaseModel,
    onnx_path: str,
    engine_path: str,
    opt_batch_size: int,
    opt_image_height: int,
    opt_image_width: int,
    precision: str,
):
    """SDXL Union ControlNet (quantized in place) -> static ONNX -> strongly-typed engine."""
    from ..quantization import export_quantized_onnx, build_strongly_typed_engine

    wrapper = TorchControlNetUnionWrapper(controlnet_union, model_data.control_type_idx)
    inputs = tuple(model_data.get_sample_input(opt_batch_size, opt_image_height, opt_image_width))
    export_quantized_onnx(
        wrapper, controlnet_union, inputs, model_data.get_input_names(), model_data.get_output_names(),
        onnx_path, precision,
    )
    del wrapper, inputs
    gc.collect()
    torch.cuda.empty_cache()
    profile = model_data.get_input_profile(
        opt_batch_size, opt_image_height, opt_image_width, static_batch=False, static_shape=True,
    )
    build_strongly_typed_engine(onnx_path, engine_path, profile)


class TorchDepthAnythingWrapper(torch.nn.Module):
    """Wraps HF DepthAnythingForDepthEstimation so ONNX gets a single named output."""
    def __init__(self, depth_model):
        super().__init__()
        self.depth_model = depth_model

    def forward(self, pixel_values):
        outputs = self.depth_model(pixel_values=pixel_values)
        return outputs.predicted_depth


def compile_depth_anything(
    depth_model,
    model_data: BaseModel,
    onnx_path: str,
    onnx_opt_path: str,
    engine_path: str,
    image_size: int = 378,
    engine_build_options: dict = {},
):
    """Compile Depth-Anything V2 to a static-shape TRT engine."""
    wrapper = TorchDepthAnythingWrapper(depth_model).to(torch.device("cuda"), dtype=torch.float16)

    builder = EngineBuilder(model_data, wrapper, device=torch.device("cuda"))
    builder.build(
        onnx_path,
        onnx_opt_path,
        engine_path,
        opt_batch_size=1,
        opt_image_height=image_size,
        opt_image_width=image_size,
        build_static_batch=True,
        build_dynamic_shape=False,
        **engine_build_options,
    )


def accelerate_with_tensorrt(
    stream: StreamDiffusion,
    engine_dir: str,
    max_batch_size: int = 2,
    min_batch_size: int = 1,
    use_cuda_graph: bool = False,
    engine_build_options: dict = {},
):
    if "opt_batch_size" not in engine_build_options or engine_build_options["opt_batch_size"] is None:
        engine_build_options["opt_batch_size"] = max_batch_size
    text_encoder = stream.text_encoder
    unet = stream.unet
    vae = stream.vae

    del stream.unet, stream.vae, stream.pipe.unet, stream.pipe.vae

    vae_config = vae.config
    vae_dtype = vae.dtype

    unet.to(torch.device("cpu"))
    vae.to(torch.device("cpu"))

    gc.collect()
    torch.cuda.empty_cache()

    onnx_dir = os.path.join(engine_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    unet_engine_path = f"{engine_dir}/unet.engine"
    vae_encoder_engine_path = f"{engine_dir}/vae_encoder.engine"
    vae_decoder_engine_path = f"{engine_dir}/vae_decoder.engine"

    unet_model = UNet(
        fp16=True,
        device=stream.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        embedding_dim=text_encoder.config.hidden_size,
        unet_dim=unet.config.in_channels,
    )
    vae_decoder_model = VAE(
        device=stream.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
    )
    vae_encoder_model = VAEEncoder(
        device=stream.device,
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
    )

    if not engine_ready(unet_engine_path):
        compile_unet(
            unet,
            unet_model,
            create_onnx_path("unet", onnx_dir, opt=False),
            create_onnx_path("unet", onnx_dir, opt=True),
            unet_engine_path,
            **engine_build_options,
        )
    else:
        del unet

    if not engine_ready(vae_decoder_engine_path):
        vae.forward = vae.decode
        compile_vae_decoder(
            vae,
            vae_decoder_model,
            create_onnx_path("vae_decoder", onnx_dir, opt=False),
            create_onnx_path("vae_decoder", onnx_dir, opt=True),
            vae_decoder_engine_path,
            **engine_build_options,
        )

    if not engine_ready(vae_encoder_engine_path):
        vae_encoder = TorchVAEEncoder(vae).to(torch.device("cuda"))
        compile_vae_encoder(
            vae_encoder,
            vae_encoder_model,
            create_onnx_path("vae_encoder", onnx_dir, opt=False),
            create_onnx_path("vae_encoder", onnx_dir, opt=True),
            vae_encoder_engine_path,
            **engine_build_options,
        )

    del vae

    cuda_stream = cuda.Stream()

    stream.unet = UNet2DConditionModelEngine(unet_engine_path, cuda_stream, use_cuda_graph=use_cuda_graph)
    stream.vae = AutoencoderKLEngine(
        vae_encoder_engine_path,
        vae_decoder_engine_path,
        cuda_stream,
        stream.pipe.vae_scale_factor,
        use_cuda_graph=use_cuda_graph,
    )
    setattr(stream.vae, "config", vae_config)
    setattr(stream.vae, "dtype", vae_dtype)

    gc.collect()
    torch.cuda.empty_cache()

    import logging as _logging
    try:
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        reserv_gb = torch.cuda.memory_reserved() / 1e9
        _logging.info(
            f"[VRAM] after TRT init: allocated={alloc_gb:.2f} GB, "
            f"reserved={reserv_gb:.2f} GB (gap={reserv_gb - alloc_gb:.2f} GB)"
        )
    except Exception:
        pass

    return stream
