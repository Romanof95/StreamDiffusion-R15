"""Typed configuration schema for StreamDiffusion (SD 1.5 / SDXL)."""
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

@dataclass
class CNConfig:
    enabled: bool = True
    guidance_strength: float = 0.58
    # >1: run the preprocessors AND the ControlNet forward every N frames only; in between,
    # the cached control maps and the previous ControlNet residuals are reused (single-step
    # streams). Saves the whole ControlNet cost on skipped frames at the price of a control
    # signal refreshed at fps/N.
    skip_frames: int = 1
    preview_mode: str = "normal"


@dataclass
class CannyConfig:
    enabled: bool = False
    scale: float = 1.0
    resolution: int = 384
    low_threshold: int = 100
    high_threshold: int = 255
    aperture_size: int = 3
    l2_gradient: bool = False
    # GPU Canny (PyTorch + CUDA graph, ~0.4 ms @384 / ~1.5 ms @1024); False = original OpenCV CPU path.
    gpu: bool = True


@dataclass
class DepthConfig:
    enabled: bool = False
    scale: float = 0.6
    method: str = "grayscale"
    model_size: str = "small"
    resolution: int = 384
    blur_kernel: int = 1
    contrast: float = 1.0
    brightness: int = 0
    near_threshold: int = 0
    far_threshold: int = 255
    invert: bool = False

    def __post_init__(self):
        if self.blur_kernel < 1:
            self.blur_kernel = 1
        elif self.blur_kernel % 2 == 0:
            self.blur_kernel += 1


@dataclass
class OpenPoseConfig:
    enabled: bool = False
    scale: float = 1.0
    detect_resolution: int = 512
    # DWPose (YOLOX-S + DW-LL) as TensorRT engines (~5 ms/frame) instead of onnxruntime (~22 ms).
    # Engines built once into tensorrt_cache/dwpose/; onnxruntime is the automatic fallback.
    tensorrt: bool = True


@dataclass
class FaceIDConfig:
    enabled: bool = False
    model: str = "h94/IP-Adapter-FaceID"
    weight_name: str = "ip-adapter-faceid_sd15.bin"
    scale: float = 0.6
    skip_frames: int = 10
    plus_v2: bool = False


@dataclass
class StreamV2VConfig:
    enabled: bool = False
    cache_maxframes: int = 4
    cache_interval: int = 1
    # Speed/quality trade-offs of the SDXL TensorRT (ring) engine, all off by default.
    # Not in the Smode packet yet: set the environment variables before launch.
    # attn_cache_pool: cached keys/values are average-pooled NxN before the extended
    # attention (2 -> attention over 1.5x tokens instead of 3x with 2 cached frames).
    attn_cache_pool: int = int(os.environ.get("STREAMDIFFUSION_V2V_ATTN_POOL", "1"))
    # fi_last_frame_only: feature injection matches the newest cached frame only.
    fi_last_frame_only: bool = os.environ.get("STREAMDIFFUSION_V2V_FI_LAST", "0") == "1"
    # attn_decoder_only: extended attention in mid/up blocks only.
    attn_decoder_only: bool = os.environ.get("STREAMDIFFUSION_V2V_ATTN_DECODER", "0") == "1"


@dataclass
class SimilarImageFilterConfig:
    enabled: bool = True
    threshold: float = 0.95
    max_skip: int = 5

@dataclass
class ControlNetConfig:
    """Top-level configuration for ControlNet"""
    controlnet: CNConfig = field(default_factory=CNConfig)

    # Individual ControlNet configs
    canny: CannyConfig = field(default_factory=CannyConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    openpose: OpenPoseConfig = field(default_factory=OpenPoseConfig)

    # IP-Adapter FaceID
    faceid: FaceIDConfig = field(default_factory=FaceIDConfig)

    # Temporal consistency
    streamv2v: StreamV2VConfig = field(default_factory=StreamV2VConfig)
    latent_feedback_strength: float = 0.0

    # Acceleration
    use_tiny_vae: bool = True
    torch_compile_enabled: bool = True
    # TensorRT engine precision: "fp16" | "mxfp8" | "nvfp4" (RTX 50 / Blackwell block-scaled
    # formats). Applied to the SDXL UNet and the SDXL Union ControlNet at engine-build time
    # (ModelOpt quantization, one-time per model+LoRA+steps+resolution, cached like fp16
    # engines). Ignored outside TensorRT. Measured on an RTX 5080 @1024: mxfp8 -12% UNet time
    # with an image visually identical to fp16; nvfp4 -29% but the 1-step image drifts.
    # Not in the Smode packet yet: set STREAMDIFFUSION_PRECISION in the environment.
    precision: str = os.environ.get("STREAMDIFFUSION_PRECISION", "fp16")

    # Profiling: [PERF] line every 60 frames (GPU breakdown, frame-time spread, time spent
    # waiting for the caller). Not in the Smode packet: set STREAMDIFFUSION_PROFILING=1.
    profiling_enabled: bool = os.environ.get("STREAMDIFFUSION_PROFILING", "0") == "1"

    # Low-latency mode (controlled GC + HIGH process priority)
    low_latency_mode: bool = False

    def get(self, key, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        
        # Nested controlnet settings (CNConfig keeps full field names)
        if hasattr(self.controlnet, key):
            return getattr(self.controlnet, key)

        # Nested FaceID settings
        if key.startswith("faceid_"):
            return getattr(self.faceid, key.removeprefix("faceid_"), default)

        # Nested StreamV2V settings
        if key.startswith("streamv2v_"):
            return getattr(self.streamv2v, key.removeprefix("streamv2v_"), default)

        # Nested preprocessors
        if key.startswith("canny_"):
            return getattr(self.canny, key.removeprefix("canny_"), default)

        if key.startswith("depth_"):
            return getattr(self.depth, key.removeprefix("depth_"), default)

        if key.startswith("openpose_"):
            return getattr(self.openpose, key.removeprefix("openpose_"), default)

        return default

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
            return
        if hasattr(self.controlnet, key):
            setattr(self.controlnet, key, value)
            return
        for prefix, sub in (
            ("canny_", self.canny),
            ("depth_", self.depth),
            ("openpose_", self.openpose),
            ("faceid_", self.faceid),
            ("streamv2v_", self.streamv2v),
        ):
            if key.startswith(prefix):
                setattr(sub, key.removeprefix(prefix), value)
                return
