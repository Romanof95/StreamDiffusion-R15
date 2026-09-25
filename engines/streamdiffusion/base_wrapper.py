"""Base wrapper class shared by SD 1.5 and SDXL StreamDiffusion wrappers."""
import gc
import hashlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import logging
import numpy as np
import torch
from PIL import Image

from pipeline.image_utils import postprocess_image

# Package root — TRT engines and torch.compile caches live alongside the code.
PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent


def from_pretrained_any_variant(load: Callable, *args, **kwargs):
    """``load(*args, **kwargs)`` (a ``from_pretrained``), retried with ``variant="fp16"``: some
    repos ship only ``*.fp16.safetensors`` (RunDiffusion/Juggernaut-XL-v9), the default load then
    fetches the configs alone and fails on the first model ("no file named model.safetensors").
    Raises the first error when the fp16 retry fails too."""
    try:
        return load(*args, **kwargs)
    except Exception as first:
        try:
            result = load(*args, variant="fp16", **kwargs)
        except Exception:
            raise first
        logging.info(f"[Model] {args[0] if args else ''}: no default weights, fp16 variant loaded")
        return result


def lora_signature(lora_dict: Optional[Dict[str, float]], lcm_fused: bool = False) -> str:
    """Stable short hash of (lora_name, weight) pairs for TRT engine prefixes.

    Returns empty string when no custom LoRAs are set, so default engines
    keep their existing on-disk path. ``-f2`` marks engines where two or more adapters
    (custom LoRAs, or the LCM LoRA plus a custom one) are fused: before
    ``fuse_lora_adapters`` only the last one survived, so those engines must be rebuilt.
    """
    if not lora_dict:
        return ""
    items = sorted((k, round(float(v), 4)) for k, v in lora_dict.items())
    h = hashlib.md5(repr(items).encode("utf-8")).hexdigest()[:10]
    stacked = len(lora_dict) + (1 if lcm_fused else 0) >= 2
    return f"--lora-{h}" + ("-f2" if stacked else "")


def user_lcm_lora(lora_dict) -> bool:
    """An LCM LoRA listed with the user LoRAs (latent-consistency/lcm-lora-sdv1-5...) replaces
    the automatic one: both fused stacked LCM at 2x and burned the image."""
    return any("lcm" in str(k).lower() for k in (lora_dict or {}))


def lcm_lora_fused(model_id_or_path, use_lcm_lora, lora_dict, sdxl=False, use_hyper_unet=False) -> bool:
    """Whether the wrappers fuse the automatic LCM LoRA: requested, not already in the user
    LoRAs, and the model is not turbo / Hyper-SD (LoRA or, on SDXL, checkpoint) / SDXL Lightning."""
    m = str(model_id_or_path).lower()
    if not use_lcm_lora or "turbo" in m or "sdxs" in m or user_lcm_lora(lora_dict):
        return False
    if any("hyper" in str(k).lower() for k in (lora_dict or {})):
        return False
    if sdxl and ("lightning" in m or "hyper" in m or use_hyper_unet):
        return False
    return True


def fuse_lora_adapters(stream, adapters) -> None:
    """Load every LoRA as its own named adapter, fuse them together with their weights,
    then drop the PEFT wrappers (the fused weights stay).

    ``adapters``: list of ``(label, load_fn, args, kwargs, scale)``; ``load_fn`` is
    ``stream.load_lora`` / ``stream.load_lcm_lora``. Fusing after each load does not stack:
    loading the next adapter unmerges the fused ones ("Adapter cannot be set when the
    model is merged"), so only the last LoRA survived (Hyper-SDXL + a style LoRA ran on
    the base model plus the style LoRA)."""
    if not adapters:
        return
    names, weights = [], []
    for i, (label, load_fn, args, kwargs, scale) in enumerate(adapters):
        name = f"smode_lora_{i}"
        load_fn(*args, adapter_name=name, **kwargs)
        names.append(name)
        weights.append(float(scale))
        logging.info(f"[LoRA] Loaded {label} with weight {scale}")
    stream.pipe.set_adapters(names, adapter_weights=weights)
    stream.pipe.fuse_lora(adapter_names=names, lora_scale=1.0)
    stream.pipe.unload_lora_weights()
    logging.info(f"[LoRA] {len(names)} adapter(s) fused into the model weights")


torch.set_grad_enabled(False)
torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cuda.matmul.fp32_precision = 'tf32'


class BaseStreamDiffusionWrapper(ABC):
    """Shared logic for SD 1.5 and SDXL wrappers."""

    # CLIP Vision encoder used by FaceID Plus / PlusV2 (ViT-H/14, 1280-dim).
    FACEID_CLIP_ENCODER = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    def __init__(
        self,
        model_id_or_path: str,
        t_index_list: List[int],
        lora_dict: Optional[Dict[str, float]] = None,
        mode: Literal["img2img", "txt2img"] = "img2img",
        output_type: Literal["pil", "pt", "np", "latent"] = "pil",
        lcm_lora_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        device: Literal["cpu", "cuda"] = "cuda",
        dtype: torch.dtype = torch.float16,
        frame_buffer_size: int = 1,
        width: int = 512,
        height: int = 512,
        warmup: int = 10,
        acceleration: Literal["none", "tensorrt"] = "none",
        do_add_noise: bool = True,
        device_ids: Optional[List[int]] = None,
        use_lcm_lora: bool = True,
        use_tiny_vae: bool = True,
        enable_similar_image_filter: bool = False,
        similar_image_filter_threshold: float = 0.98,
        similar_image_filter_max_skip_frame: int = 10,
        use_denoising_batch: bool = True,
        cfg_type: Literal["none", "full", "self", "initialize"] = "self",
        seed: int = 2,
        use_safety_checker: bool = False,
        engine_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        torch_compile_enabled: bool = True,
        precision: str = "fp16",
        torch_compile_mode: str = "reduce-overhead",
        torch_compile_fullgraph: bool = False,
        faceid_config: Optional[Dict] = None,
        streamv2v_enabled: bool = False,
        streamv2v_cache_maxframes: int = 1,
        streamv2v_options: Optional[Dict] = None,
        warning_callback: Optional[Callable[[bool, str], None]] = None,
    ):
        self.sd_turbo = "turbo" in model_id_or_path or "sdxs" in model_id_or_path.lower()
        self.warning_callback = warning_callback

        if mode == "txt2img":
            if cfg_type != "none":
                raise ValueError(
                    f"txt2img mode accepts only cfg_type = 'none', but got {cfg_type}"
                )
            if use_denoising_batch and frame_buffer_size > 1:
                if not self.sd_turbo:
                    raise ValueError(
                        "txt2img mode cannot use denoising batch with frame_buffer_size > 1."
                    )

        # img2img without the denoising batch = sequential multi-step on the batch-1
        # engines (predict_x0_batch of both pipelines, STREAMDIFFUSION_SEQUENTIAL=1).

        self.device = device
        self.dtype = dtype
        self.width = width
        self.height = height
        self.mode = mode
        self.output_type = output_type
        self.frame_buffer_size = frame_buffer_size
        self.batch_size = (
            len(t_index_list) * frame_buffer_size
            if use_denoising_batch
            else frame_buffer_size
        )
        self.use_denoising_batch = use_denoising_batch
        self.use_safety_checker = use_safety_checker
        self.torch_compile_enabled = torch_compile_enabled
        self.precision = precision  # TensorRT engine precision (fp16 | mxfp8 | nvfp4)
        self.torch_compile_mode = torch_compile_mode
        self.faceid_config = faceid_config
        self._faceid_loaded = False
        self.streamv2v_enabled = streamv2v_enabled
        self.streamv2v_cache_maxframes = streamv2v_cache_maxframes
        self.streamv2v_options = dict(streamv2v_options or {})

        if engine_dir is None:
            engine_dir = self._get_default_engine_dir()

        self.stream = self._load_model(
            model_id_or_path=model_id_or_path,
            lora_dict=lora_dict,
            lcm_lora_id=lcm_lora_id,
            vae_id=vae_id,
            t_index_list=t_index_list,
            acceleration=acceleration,
            warmup=warmup,
            do_add_noise=do_add_noise,
            use_lcm_lora=use_lcm_lora,
            use_tiny_vae=use_tiny_vae,
            cfg_type=cfg_type,
            seed=seed,
            engine_dir=engine_dir,
            cache_dir=cache_dir,
        )

        if device_ids is not None:
            self.stream.unet = torch.nn.DataParallel(
                self.stream.unet, device_ids=device_ids
            )

        if enable_similar_image_filter:
            self.stream.enable_similar_image_filter(
                similar_image_filter_threshold, similar_image_filter_max_skip_frame
            )
        else:
            # The pipeline starts with the filter ON: honour an explicit "disabled".
            self.stream.disable_similar_image_filter()

    @abstractmethod
    def _load_model(self, **kwargs):
        ...

    @abstractmethod
    def _get_default_engine_dir(self) -> Path:
        ...

    @abstractmethod
    def enable_tensorrt_acceleration(
        self, stream, model_id_or_path, use_lcm_lora, use_tiny_vae,
        engine_dir=None, lora_dict=None,
    ):
        """``lora_dict`` participates in the engine cache key so swapping LoRAs
        forces a fresh compile rather than silently reusing a stale engine."""
        ...

    def load_ip_adapter_faceid(self, pipe, faceid_config, is_sdxl: bool = False):
        """Load IP-Adapter FaceID weights (Base / Plus / PlusV2) onto the pipeline.

        Must run AFTER LoRA loading and BEFORE torch.compile / TensorRT.
        """
        self._faceid_loaded = False
        self._faceid_plus_v2 = False
        self._faceid_pipe_ref = None

        if faceid_config is None:
            return

        if hasattr(faceid_config, 'enabled'):
            enabled = faceid_config.enabled
            scale = faceid_config.scale
            plus_v2 = getattr(faceid_config, 'plus_v2', False)
        else:
            enabled = faceid_config.get('faceid_enabled', False)
            scale = faceid_config.get('faceid_scale', 0.6)
            plus_v2 = faceid_config.get('faceid_plus_v2', False)

        if not enabled:
            return

        model_repo = "h94/IP-Adapter-FaceID"
        if is_sdxl:
            weight_name = "ip-adapter-faceid-plusv2_sdxl.bin" if plus_v2 else "ip-adapter-faceid_sdxl.bin"
        else:
            weight_name = "ip-adapter-faceid-plusv2_sd15.bin" if plus_v2 else "ip-adapter-faceid_sd15.bin"

        variant_name = "PlusV2" if plus_v2 else "Base"
        model_type = "SDXL" if is_sdxl else "SD 1.5"
        logging.info(f"[FaceID] Loading {variant_name} variant for {model_type}: {weight_name}")

        try:
            if plus_v2:
                from transformers import CLIPVisionModelWithProjection
                logging.info(f"[FaceID] Loading CLIP Vision encoder: {self.FACEID_CLIP_ENCODER}")
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    self.FACEID_CLIP_ENCODER,
                    torch_dtype=self.dtype,
                ).to(self.device)
                pipe.image_encoder = image_encoder

            # image_encoder_folder=None because the face embedding comes from
            # InsightFace, not a CLIP encoder inside the IP-Adapter repo.
            pipe.load_ip_adapter(
                model_repo,
                subfolder=None,
                weight_name=weight_name,
                image_encoder_folder=None,
            )

            pipe.set_ip_adapter_scale(scale)

            if plus_v2:
                try:
                    proj_layer = pipe.unet.encoder_hid_proj.image_projection_layers[0]
                    proj_layer.shortcut = False
                    # Pre-inject zero CLIP embeddings: diffusers' forward reads
                    # clip_embeds.dtype + reshapes to 4D before any real face
                    # extraction has injected an embedding.
                    proj_layer.clip_embeds = torch.zeros(
                        1, 1, 257, 1280, dtype=self.dtype, device=self.device
                    )
                except Exception as e:
                    logging.warning(f"[FaceID] Could not configure PlusV2 projection layer: {e}")

            self._faceid_loaded = True
            self._faceid_plus_v2 = plus_v2
            self._faceid_pipe_ref = pipe
            logging.info(f"[FaceID] IP-Adapter FaceID {variant_name} loaded successfully (scale={scale})")

        except Exception as e:
            logging.error(f"[FaceID] Failed to load IP-Adapter {variant_name}: {e}")
            import traceback
            traceback.print_exc()
            self._faceid_loaded = False
            self._faceid_plus_v2 = False

    def set_faceid_scale(self, scale: float) -> bool:
        """Live update of IP-Adapter FaceID scale. Returns True on success."""
        if not self._faceid_loaded or self._faceid_pipe_ref is None:
            return False
        try:
            self._faceid_pipe_ref.set_ip_adapter_scale(scale)
            return True
        except Exception as e:
            logging.warning(f"[FaceID] Failed to update scale via set_ip_adapter_scale: {e}")
            return False

    def prepare(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.2,
        delta: float = 1.0,
    ) -> None:
        self.stream.prepare(
            prompt,
            negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            delta=delta,
        )

    def __call__(
        self,
        image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
        prompt: Optional[str] = None,
        controlnet_image: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        controlnet_model: Optional[Union[Any, List[Any]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        if self.mode == "img2img":
            return self.img2img(
                image, prompt,
                controlnet_image=controlnet_image,
                controlnet_model=controlnet_model,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
            )
        else:
            return self.txt2img(prompt)

    def txt2img(
        self, prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if self.sd_turbo:
            image_tensor = self.stream.txt2img_sd_turbo(self.batch_size)
        else:
            image_tensor = self.stream.txt2img(self.frame_buffer_size)
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            image = self._check_safety(image, image_tensor)

        return image

    def img2img(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        prompt: Optional[str] = None,
        controlnet_image: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        controlnet_model: Optional[Union[Any, List[Any]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        if prompt is not None:
            self.stream.update_prompt(prompt)

        if isinstance(image, str) or isinstance(image, Image.Image):
            image = self.preprocess_image(image)

        image_tensor = self.stream(
            image,
            controlnet_image=controlnet_image,
            controlnet_model=controlnet_model,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
        )
        image = self.postprocess_image(image_tensor, output_type=self.output_type)

        if self.use_safety_checker:
            image = self._check_safety(image, image_tensor)

        return image

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(image, str):
            with Image.open(image) as img:
                image = img.convert("RGB").resize((self.width, self.height))
        elif isinstance(image, Image.Image):
            image = image.convert("RGB").resize((self.width, self.height))

        return self.stream.image_processor.preprocess(
            image, self.height, self.width
        ).to(device=self.device, dtype=self.dtype)

    def postprocess_image(
        self, image_tensor: torch.Tensor, output_type: str = "pil"
    ) -> Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray]:
        if self.frame_buffer_size > 1:
            return postprocess_image(image_tensor, output_type=output_type)
        else:
            return postprocess_image(image_tensor, output_type=output_type)[0]

    def _check_safety(self, image, image_tensor):
        safety_checker_input = self.feature_extractor(
            image, return_tensors="pt"
        ).to(self.device)
        _, has_nsfw_concept = self.safety_checker(
            images=image_tensor.to(self.dtype),
            clip_input=safety_checker_input.pixel_values.to(self.dtype),
        )
        return self.nsfw_fallback_img if has_nsfw_concept[0] else image

    def _emit_warning(self, active: bool, message: str = "") -> None:
        if self.warning_callback is None:
            return
        try:
            self.warning_callback(active, message)
        except Exception as e:
            logging.warning(f"[Warning] warning_callback failed: {e}")

    def setup_torch_compile(self, stream, acceleration: str, cache_subdir: str):
        """Apply torch.compile() to UNet and VAE (shared SD/SDXL logic)."""
        if not self.torch_compile_enabled:
            logging.info("Skipping torch.compile() - disabled by config")
            return
        if acceleration == "tensorrt":
            logging.info("Skipping torch.compile() - TensorRT already provides maximum optimization")
            return
        if not hasattr(torch, 'compile') or torch.__version__ < '2.0':
            logging.warning("torch.compile() not available (PyTorch < 2.0)")
            return

        try:
            cache_dir = PACKAGE_DIR / f"torch_compile_cache/{cache_subdir}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache_dir)
            logging.info(f"Compiling U-Net with torch.compile() (cache: {cache_dir})...")

            self._emit_warning(True, "Compiling U-Net with torch.compile() - first run can take 1-2 minutes")
            stream.unet = torch.compile(
                stream.unet,
                mode=self.torch_compile_mode,
                fullgraph=False,
                dynamic=False
            )

            if hasattr(stream.vae, 'encoder'):
                self._emit_warning(True, "Compiling VAE encoder with torch.compile()")
                self._compile_vae_component(
                    stream.vae, 'encoder',
                    PACKAGE_DIR / f"torch_compile_cache/{cache_subdir}_vae_encoder",
                    dummy_shape=(1, 3, 512, 512),
                )

            if hasattr(stream.vae, 'decoder'):
                self._emit_warning(True, "Compiling VAE decoder with torch.compile()")
                self._compile_vae_component(
                    stream.vae, 'decoder',
                    PACKAGE_DIR / f"torch_compile_cache/{cache_subdir}_vae_decoder",
                    dummy_shape=(1, 4, 64, 64),
                )

        except Exception as e:
            logging.warning(f"torch.compile() failed: {e}. Continuing without compilation.")
        finally:
            self._emit_warning(False)

    def _compile_vae_component(self, vae, component_name: str, cache_dir: Path, dummy_shape: tuple):
        """Compile and warmup a VAE component (encoder or decoder)."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        old_cache = os.environ.get('TORCHINDUCTOR_CACHE_DIR', '')
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache_dir)
        os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'

        try:
            component = getattr(vae, component_name)
            compiled = torch.compile(component, mode='reduce-overhead', fullgraph=False, dynamic=False)
            setattr(vae, component_name, compiled)

            dummy = torch.randn(*dummy_shape, dtype=vae.dtype, device=vae.device)
            with torch.no_grad():
                _ = compiled(dummy)
        finally:
            if old_cache:
                os.environ['TORCHINDUCTOR_CACHE_DIR'] = old_cache
            else:
                os.environ.pop('TORCHINDUCTOR_CACHE_DIR', None)

    def cleanup(self):
        """Release GPU resources. Prevents leaks on wrapper recreation."""
        try:
            if hasattr(self, 'stream') and self.stream is not None:
                for buf in ['x_t_latent_buffer', 'init_noise', 'stock_noise', 'prev_image_result']:
                    if hasattr(self.stream, buf):
                        delattr(self.stream, buf)

                if hasattr(self.stream, '_scheduler_coeffs_cache'):
                    for coeffs in self.stream._scheduler_coeffs_cache.values():
                        for tensor in coeffs.values():
                            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                                del tensor
                    self.stream._scheduler_coeffs_cache.clear()

                if hasattr(self.stream, 'similar_filter') and self.stream.similar_filter is not None:
                    self.stream.similar_filter.prev_tensor = None

                # TensorRT: release the engines now (graphs, contexts, buffers) rather than
                # whenever GC reaches them, then the CUDA stream they ran on (polygraphy
                # streams have no finalizer). ControlNet / depth engines that borrow this
                # stream are released by the app before this cleanup.
                trt_stream = None
                for comp in (getattr(self.stream, 'unet', None), getattr(self.stream, 'vae', None)):
                    if comp is None:
                        continue
                    trt_stream = trt_stream or getattr(comp, 'stream', None)
                    for attr in ('engine', 'encoder', 'decoder'):
                        eng = getattr(comp, attr, None)
                        if eng is not None and hasattr(eng, 'release'):
                            try:
                                eng.release()
                            except Exception:
                                pass
                if trt_stream is not None and hasattr(trt_stream, 'free'):
                    try:
                        trt_stream.synchronize()
                        trt_stream.free()
                    except Exception:
                        pass

                del self.stream
                self.stream = None

            if hasattr(self, 'safety_checker') and self.safety_checker is not None:
                del self.safety_checker
                self.safety_checker = None
            if hasattr(self, 'feature_extractor') and self.feature_extractor is not None:
                del self.feature_extractor
                self.feature_extractor = None

            # Release the FaceID pipe. FaceID forces acceleration='none', so this
            # ref holds the FULL PyTorch UNet/VAE/TEs (~5 GB) on GPU; without
            # nulling it here, `del self.stream` cannot break the chain and the
            # VRAM survives cleanup() — OOM risk across wrapper recreation on a
            # 16 GB card. image_encoder is freed transitively with the pipe.
            if getattr(self, '_faceid_pipe_ref', None) is not None:
                del self._faceid_pipe_ref
            self._faceid_pipe_ref = None
            self._faceid_loaded = False

            # gc.collect() BEFORE empty_cache(): cyclic refs (pipe<->components,
            # KvoCacheWrapper<->unet, faceid back-refs) must be collected first
            # or their tensors stay alive through the flush. Matches the rest of
            # the codebase's teardown order.
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            logging.warning(f"Error during cleanup: {e}")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
