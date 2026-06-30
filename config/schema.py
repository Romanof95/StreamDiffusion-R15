"""Typed configuration schema for StreamDiffusion (SD 1.5 / SDXL)."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional

@dataclass
class CNConfig:
    controlnet_enabled: bool = True
    controlnet_guidance_strength: float = 0.58
    controlnet_skip_frames: int = 1
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
    motion_aware_noise: bool = True
    motion_aware_noise_sensitivity: float = 1.0

    # Acceleration
    use_tiny_vae: bool = True
    torch_compile_enabled: bool = True

    # Profiling
    profiling_enabled: bool = False

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
