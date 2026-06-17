"""Typed configuration schema for StreamDiffusion (SD 1.5 / SDXL)."""
from dataclasses import dataclass, field, asdict
from typing import List, Optional


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
    enabled: bool = True
    cache_maxframes: int = 4
    cache_interval: int = 1


@dataclass
class SimilarImageFilterConfig:
    enabled: bool = True
    threshold: float = 0.95
    max_skip: int = 5


@dataclass
class StreamDiffusionConfig:
    """Top-level configuration for StreamDiffusion (SD 1.5 / SDXL)."""

    # Global ControlNet settings
    controlnet_enabled: bool = True
    controlnet_guidance_strength: float = 0.58
    controlnet_skip_frames: int = 1
    preview_mode: str = "normal"

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

    # Similar image filter
    similar_image_filter: SimilarImageFilterConfig = field(
        default_factory=SimilarImageFilterConfig
    )

    # Acceleration
    use_tiny_vae: bool = True
    torch_compile_enabled: bool = True

    # Profiling
    profiling_enabled: bool = False

    # Low-latency mode (controlled GC + HIGH process priority)
    low_latency_mode: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "StreamDiffusionConfig":
        """Create config from a flat dictionary (controlnet_config.json format)."""
        config = cls()

        config.controlnet_enabled = data.get("controlnet_enabled", config.controlnet_enabled)
        config.controlnet_guidance_strength = data.get(
            "controlnet_guidance_strength", config.controlnet_guidance_strength
        )
        config.controlnet_skip_frames = data.get(
            "controlnet_skip_frames", config.controlnet_skip_frames
        )
        config.preview_mode = data.get("preview_mode", config.preview_mode)

        # Canny
        config.canny.enabled = data.get("canny_enabled", config.canny.enabled)
        config.canny.scale = data.get("canny_scale", config.canny.scale)
        config.canny.resolution = data.get("canny_resolution", config.canny.resolution)
        config.canny.low_threshold = data.get("canny_low_threshold", config.canny.low_threshold)
        config.canny.high_threshold = data.get("canny_high_threshold", config.canny.high_threshold)
        config.canny.aperture_size = data.get("canny_aperture_size", config.canny.aperture_size)
        config.canny.l2_gradient = data.get("canny_l2_gradient", config.canny.l2_gradient)

        # Depth
        config.depth.enabled = data.get("depth_enabled", config.depth.enabled)
        config.depth.scale = data.get("depth_scale", config.depth.scale)
        config.depth.method = data.get("depth_method", config.depth.method)
        config.depth.model_size = data.get("depth_model_size", config.depth.model_size)
        config.depth.resolution = data.get("depth_resolution", config.depth.resolution)
        config.depth.blur_kernel = data.get("depth_blur_kernel", config.depth.blur_kernel)
        config.depth.contrast = data.get("depth_contrast", config.depth.contrast)
        config.depth.brightness = data.get("depth_brightness", config.depth.brightness)
        config.depth.near_threshold = data.get("depth_near_threshold", config.depth.near_threshold)
        config.depth.far_threshold = data.get("depth_far_threshold", config.depth.far_threshold)
        config.depth.invert = data.get("depth_invert", config.depth.invert)

        # OpenPose
        config.openpose.enabled = data.get("openpose_enabled", config.openpose.enabled)
        config.openpose.scale = data.get("openpose_scale", config.openpose.scale)
        config.openpose.detect_resolution = data.get(
            "openpose_detect_resolution", config.openpose.detect_resolution
        )

        # FaceID
        config.faceid.enabled = data.get("faceid_enabled", config.faceid.enabled)
        config.faceid.model = data.get("faceid_model", config.faceid.model)
        config.faceid.weight_name = data.get("faceid_weight_name", config.faceid.weight_name)
        config.faceid.scale = data.get("faceid_scale", config.faceid.scale)
        config.faceid.skip_frames = data.get("faceid_skip_frames", config.faceid.skip_frames)
        config.faceid.plus_v2 = data.get("faceid_plus_v2", config.faceid.plus_v2)

        # StreamV2V
        config.streamv2v.enabled = data.get("streamv2v_enabled", config.streamv2v.enabled)
        config.streamv2v.cache_maxframes = data.get(
            "streamv2v_cache_maxframes", config.streamv2v.cache_maxframes
        )
        config.streamv2v.cache_interval = data.get(
            "streamv2v_cache_interval", config.streamv2v.cache_interval
        )

        # Similar image filter
        config.similar_image_filter.enabled = data.get(
            "similar_image_filter_enabled", config.similar_image_filter.enabled
        )
        config.similar_image_filter.threshold = data.get(
            "similar_image_filter_threshold", config.similar_image_filter.threshold
        )
        config.similar_image_filter.max_skip = data.get(
            "similar_image_filter_max_skip", config.similar_image_filter.max_skip
        )

        # Acceleration & misc
        config.use_tiny_vae = data.get("use_tiny_vae", config.use_tiny_vae)
        config.torch_compile_enabled = data.get(
            "torch_compile_enabled", config.torch_compile_enabled
        )
        config.latent_feedback_strength = data.get(
            "latent_feedback_strength", config.latent_feedback_strength
        )
        config.motion_aware_noise = data.get("motion_aware_noise", config.motion_aware_noise)
        config.motion_aware_noise_sensitivity = data.get(
            "motion_aware_noise_sensitivity", config.motion_aware_noise_sensitivity
        )
        config.profiling_enabled = data.get("profiling_enabled", config.profiling_enabled)
        config.low_latency_mode = data.get("low_latency_mode", config.low_latency_mode)

        return config

    def to_dict(self) -> dict:
        """Export config back to flat dictionary format (for JSON serialization)."""
        data = {}

        data["controlnet_enabled"] = self.controlnet_enabled
        data["controlnet_guidance_strength"] = self.controlnet_guidance_strength
        data["controlnet_skip_frames"] = self.controlnet_skip_frames
        data["preview_mode"] = self.preview_mode

        # Canny
        data["canny_enabled"] = self.canny.enabled
        data["canny_scale"] = self.canny.scale
        data["canny_resolution"] = self.canny.resolution
        data["canny_low_threshold"] = self.canny.low_threshold
        data["canny_high_threshold"] = self.canny.high_threshold
        data["canny_aperture_size"] = self.canny.aperture_size
        data["canny_l2_gradient"] = self.canny.l2_gradient

        # Depth
        data["depth_enabled"] = self.depth.enabled
        data["depth_scale"] = self.depth.scale
        data["depth_method"] = self.depth.method
        data["depth_model_size"] = self.depth.model_size
        data["depth_resolution"] = self.depth.resolution
        data["depth_blur_kernel"] = self.depth.blur_kernel
        data["depth_contrast"] = self.depth.contrast
        data["depth_brightness"] = self.depth.brightness
        data["depth_near_threshold"] = self.depth.near_threshold
        data["depth_far_threshold"] = self.depth.far_threshold
        data["depth_invert"] = self.depth.invert

        # OpenPose
        data["openpose_enabled"] = self.openpose.enabled
        data["openpose_scale"] = self.openpose.scale
        data["openpose_detect_resolution"] = self.openpose.detect_resolution

        # FaceID
        data["faceid_enabled"] = self.faceid.enabled
        data["faceid_model"] = self.faceid.model
        data["faceid_weight_name"] = self.faceid.weight_name
        data["faceid_scale"] = self.faceid.scale
        data["faceid_skip_frames"] = self.faceid.skip_frames
        data["faceid_plus_v2"] = self.faceid.plus_v2

        # StreamV2V
        data["streamv2v_enabled"] = self.streamv2v.enabled
        data["streamv2v_cache_maxframes"] = self.streamv2v.cache_maxframes
        data["streamv2v_cache_interval"] = self.streamv2v.cache_interval

        # Similar image filter
        data["similar_image_filter_enabled"] = self.similar_image_filter.enabled
        data["similar_image_filter_threshold"] = self.similar_image_filter.threshold
        data["similar_image_filter_max_skip"] = self.similar_image_filter.max_skip

        # Acceleration & misc
        data["use_tiny_vae"] = self.use_tiny_vae
        data["torch_compile_enabled"] = self.torch_compile_enabled
        data["latent_feedback_strength"] = self.latent_feedback_strength
        data["motion_aware_noise"] = self.motion_aware_noise
        data["motion_aware_noise_sensitivity"] = self.motion_aware_noise_sensitivity
        data["profiling_enabled"] = self.profiling_enabled
        data["low_latency_mode"] = self.low_latency_mode

        return data
