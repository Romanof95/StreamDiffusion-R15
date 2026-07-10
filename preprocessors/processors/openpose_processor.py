"""OpenPose / DWPose preprocessor for ControlNet."""
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from ..base import BasePreprocessor

# processors/ → preprocessors/ → package root.
PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent


class OptimizedDWposeDetector:
    """Custom DWPose detector wrapping easy_dwpose ``Wholebody`` with the
    standard DWposeDetector-compatible interface (YOLOX-S + DWPose-M)."""

    def __init__(self, pose_estimation_model):
        self.pose_estimation = pose_estimation_model

    @torch.inference_mode()
    def __call__(self, image, detect_resolution=512, draw_pose=None, output_type="pil", **kwargs):
        from easy_dwpose.body_estimation import resize_image
        from easy_dwpose.draw import draw_openpose

        if draw_pose is None:
            draw_pose = draw_openpose

        if type(image) != np.ndarray:
            image = np.array(image.convert("RGB"))

        image = image.copy()
        original_height, original_width, _ = image.shape

        image = resize_image(image, target_resolution=detect_resolution)
        height, width, _ = image.shape

        candidates, scores = self.pose_estimation(image)

        num_candidates, _, locs = candidates.shape
        candidates[..., 0] /= float(width)
        candidates[..., 1] /= float(height)

        bodies = candidates[:, :18].copy()
        bodies = bodies.reshape(num_candidates * 18, locs)

        body_scores = scores[:, :18]
        for i in range(len(body_scores)):
            for j in range(len(body_scores[i])):
                if body_scores[i][j] > 0.3:
                    body_scores[i][j] = int(18 * i + j)
                else:
                    body_scores[i][j] = -1

        faces = candidates[:, 24:92]
        faces_scores = scores[:, 24:92]

        hands = np.vstack([candidates[:, 92:113], candidates[:, 113:]])
        hands_scores = np.vstack([scores[:, 92:113], scores[:, 113:]])

        pose = dict(
            bodies=bodies, body_scores=body_scores,
            hands=hands, hands_scores=hands_scores,
            faces=faces, faces_scores=faces_scores,
        )

        if not draw_pose:
            return pose

        import PIL.Image
        pose_image = draw_pose(pose, height=height, width=width, **kwargs)
        pose_image = cv2.resize(pose_image, (original_width, original_height), cv2.INTER_LANCZOS4)

        if output_type == "pil":
            pose_image = PIL.Image.fromarray(pose_image)
        elif output_type == "np":
            pass
        else:
            raise ValueError("output_type should be 'pil' or 'np'")

        return pose_image


class OpenPoseProcessor(BasePreprocessor):
    """DWPose human pose detection preprocessor (YOLOX-S + DWPose-M)."""

    def __init__(self, device: torch.device, torch_dtype: torch.dtype, max_buffer_size: int = 1024):
        super().__init__(device, torch_dtype, max_buffer_size)
        self._detector: Optional[OptimizedDWposeDetector] = None
        self._input_buffer_max: Optional[np.ndarray] = None
        self._output_buffer_max: Optional[torch.Tensor] = None
        self._pose_cache: Optional[torch.Tensor] = None

    @property
    def name(self) -> str:
        return "openpose"

    def load_model(self, config) -> None:
        """Load DWPose detector with optimized ONNX models."""
        if self._detector is not None:
            return

        try:
            from huggingface_hub import hf_hub_download
            from easy_dwpose.body_estimation import Wholebody

            logging.info("Loading DWPose preprocessor (YOLOX-S + DWPose-M, GPU-accelerated)...")

            model_det_path = hf_hub_download(
                "hr16/yolox-onnx", "yolox_s.onnx",
                local_dir=str(PACKAGE_DIR / "checkpoints")
            )
            model_pose_path = hf_hub_download(
                "hr16/UnJIT-DWPose", "dw-mm_ucoco.onnx",
                local_dir=str(PACKAGE_DIR / "checkpoints")
            )

            pose_estimation = Wholebody(
                device=self.device,
                model_det=model_det_path,
                model_pose=model_pose_path
            )

            self._detector = OptimizedDWposeDetector(pose_estimation)
            self._loaded = True
            logging.info("DWPose loaded successfully (YOLOX-S + DWPose-M)")
        except Exception as e:
            logging.error(f"Failed to load DWPose: {e}")
            self._detector = None
            self._loaded = False
            torch.cuda.empty_cache()

    def unload_model(self) -> None:
        if self._detector is not None:
            if hasattr(self._detector.pose_estimation, 'cleanup'):
                self._detector.pose_estimation.cleanup()
            del self._detector
            self._detector = None
        self._input_buffer_max = None
        self._output_buffer_max = None
        self._pose_cache = None
        self._loaded = False
        torch.cuda.empty_cache()
        logging.info("[OpenPoseProcessor] Unloaded")

    def process(self, image_tensor: torch.Tensor, config) -> Optional[torch.Tensor]:
        """Run DWPose detection. Input/output: CHW [0,1] on GPU."""
        if self._detector is None:
            logging.warning("DWPose processor not loaded, returning original image")
            return image_tensor

        try:
            h, w = image_tensor.shape[1], image_tensor.shape[2]

            if hasattr(config, 'detect_resolution'):
                detect_resolution = config.detect_resolution
            else:
                detect_resolution = config.get('openpose_detect_resolution', 512)

            if self._input_buffer_max is None:
                self._input_buffer_max = np.empty(
                    (self.max_buffer_size, self.max_buffer_size, 3), dtype=np.uint8
                )

            input_buffer = self._input_buffer_max[:h, :w, :]

            if self._output_buffer_max is None:
                self._output_buffer_max = torch.empty(
                    (3, self.max_buffer_size, self.max_buffer_size),
                    device=self.device, dtype=self.torch_dtype
                )

            output_buffer = self._output_buffer_max[:, :h, :w].contiguous()

            # Convert GPU tensor to numpy for DWPose. Scale+cast on the GPU and
            # do a single uint8 D2H into the pre-allocated buffer, instead of
            # copying fp16/fp32 down then doing two full-res CPU passes
            # (multiply + astype) per frame. Blocking copy: the CPU detector
            # reads input_buffer immediately after.
            gpu_u8 = (image_tensor.permute(1, 2, 0) * 255).clamp_(0, 255).to(torch.uint8)
            torch.from_numpy(input_buffer).copy_(gpu_u8)
            del gpu_u8

            # Body-only for real-time performance.
            openpose_np = self._detector(
                input_buffer,
                detect_resolution=detect_resolution,
                output_type='np',
                include_hands=False,
                include_face=False,
            )

            openpose_temp = torch.from_numpy(openpose_np).permute(2, 0, 1).to(self.torch_dtype) * (1.0 / 255.0)
            del openpose_np

            output_buffer.copy_(openpose_temp)
            del openpose_temp

            self._cached_result = output_buffer
            return output_buffer

        except Exception as e:
            logging.error(f"DWPose processing failed: {e}")
            return image_tensor
