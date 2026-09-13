"""Canny edge detection preprocessor for ControlNet.

Two implementations:

* GPU (default): a PyTorch re-implementation of ``cv2.GaussianBlur(3x3)`` +
  ``cv2.Canny`` (per-channel Sobel with the max-magnitude channel, non-maximum
  suppression, hysteresis) + speck removal, captured in a CUDA graph per
  (shape, aperture, L2) and replayed every frame. Thresholds are device
  tensors updated in place, so slider changes never recapture. Measured on an
  RTX 5080: 0.4 ms at 384 px, 1.5 ms at 1024 px (OpenCV path: 1.6 / 10.4 ms),
  recall 98-99 % / precision ~88 % within 1 px of the OpenCV result.
* OpenCV (fallback, ``gpu: false`` or on any GPU-path error): the original
  CPU path (GPU->CPU copy, OpenCV, connected-components filter, CPU->GPU).
"""
import logging
import math
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..base import BasePreprocessor

_HYSTERESIS_ITERS = 48   # fixed (no host sync): weak chains further than 48 px from a strong seed are dropped
_SPECK_WINDOW = 15       # speck removal: keep an edge pixel if >= _SPECK_MIN edge pixels in its 15x15 window
_SPECK_MIN = 12


class _GpuCanny:
    """CUDA-graph captured Canny for one (shape, aperture, l2) configuration."""

    def __init__(self, shape, aperture: int, l2: bool, device: torch.device):
        self.shape = tuple(shape)          # (3, H, W)
        self.aperture = aperture
        self.l2 = l2
        self.device = device
        self.inp = torch.zeros(self.shape, device=device, dtype=torch.float32)
        self.low = torch.zeros((), device=device, dtype=torch.float32)
        self.high = torch.zeros((), device=device, dtype=torch.float32)
        g = torch.tensor([math.exp(-1 / (2 * 0.64)), 1.0, math.exp(-1 / (2 * 0.64))], device=device)
        g = g / g.sum()
        self.blur = (g[:, None] * g[None, :])[None, None].expand(3, 1, 3, 3).contiguous()
        kd, ks = cv2.getDerivKernels(1, 0, aperture, normalize=False)
        kx = np.outer(ks, kd).astype(np.float32)                     # d/dx: smooth rows, derive cols
        self.kx = torch.from_numpy(kx).to(device)[None, None].expand(3, 1, aperture, aperture).contiguous()
        self.ky = torch.from_numpy(kx.T.copy()).to(device)[None, None].expand(3, 1, aperture, aperture).contiguous()
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.out: Optional[torch.Tensor] = None

    def _forward(self) -> torch.Tensor:
        img = self.inp[None] * 255.0
        img = F.conv2d(F.pad(img, (1, 1, 1, 1), mode="reflect"), self.blur, groups=3).round()
        pad = self.aperture // 2
        p = F.pad(img, (pad, pad, pad, pad), mode="reflect")
        gx = F.conv2d(p, self.kx, groups=3)
        gy = F.conv2d(p, self.ky, groups=3)
        mag_c = (gx * gx + gy * gy).sqrt() if self.l2 else (gx.abs() + gy.abs())
        idx = mag_c.argmax(dim=1, keepdim=True)        # OpenCV multi-channel: strongest channel per pixel
        mag = mag_c.gather(1, idx)[:, 0]
        gx = gx.gather(1, idx)[:, 0]
        gy = gy.gather(1, idx)[:, 0]
        ax, ay = gx.abs(), gy.abs()
        horiz = ay < 0.4142135623730951 * ax
        vert = ay > 2.414213562373095 * ax
        same = (gx * gy) >= 0
        m = F.pad(mag, (1, 1, 1, 1))
        H, W = mag.shape[-2:]

        def c(dy, dx):
            return m[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]

        n1 = torch.where(horiz, c(0, -1), torch.where(vert, c(-1, 0), torch.where(same, c(-1, -1), c(-1, 1))))
        n2 = torch.where(horiz, c(0, 1), torch.where(vert, c(1, 0), torch.where(same, c(1, 1), c(1, -1))))
        keep = ((mag >= n1) & (mag > n2)) | ((mag > n1) & (mag >= n2))
        if self.l2:
            magc = mag * mag
            low, high = self.low * self.low, self.high * self.high
        else:
            magc, low, high = mag, self.low, self.high
        e = (keep & (magc > high)).half()[None]
        wk = (keep & (magc > low)).half()[None]
        for _ in range(_HYSTERESIS_ITERS):
            e = torch.maximum(F.max_pool2d(e, 3, 1, 1) * wk, e)
        edges = e[0, 0]
        cnt = F.avg_pool2d(edges[None, None], _SPECK_WINDOW, 1, _SPECK_WINDOW // 2) * float(_SPECK_WINDOW ** 2)
        return edges * (cnt[0, 0] >= _SPECK_MIN).half()

    def capture(self) -> None:
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            for _ in range(2):
                self._forward()
        torch.cuda.current_stream(self.device).wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._forward()

    def __call__(self, x: torch.Tensor, low: float, high: float) -> torch.Tensor:
        self.inp.copy_(x)
        self.low.fill_(float(low))
        self.high.fill_(float(high))
        self.graph.replay()
        return self.out


class CannyProcessor(BasePreprocessor):
    """Canny edge detection preprocessor (GPU CUDA-graph path, OpenCV fallback)."""

    def __init__(self, device: torch.device, torch_dtype: torch.dtype, max_buffer_size: int = 1024,
                 warning_callback=None):
        super().__init__(device, torch_dtype, max_buffer_size, warning_callback)
        self._input_buffer_max: Optional[np.ndarray] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._output_buffer_shape: Optional[tuple] = None
        self._in_ema: Optional[torch.Tensor] = None
        self._gpu: Optional[_GpuCanny] = None
        self._gpu_failed = False

    @property
    def name(self) -> str:
        return "canny"

    def load_model(self, config) -> None:
        """Canny has no model to load."""
        if self._loaded:
            return
        self._emit_warning(True, "Preparing Canny preprocessor...")
        try:
            self._loaded = True
            logging.info("[CannyProcessor] Ready (GPU CUDA-graph Canny, OpenCV fallback)")
        finally:
            self._emit_warning(False)

    def unload_model(self) -> None:
        self._input_buffer_max = None
        self._output_buffer = None
        self._output_buffer_shape = None
        self._in_ema = None
        self._gpu = None
        self._loaded = False
        logging.info("[CannyProcessor] Unloaded")

    def process(self, image_tensor: torch.Tensor, config) -> Optional[torch.Tensor]:
        """Run Canny edge detection. Input/output: CHW [0,1] on GPU."""
        if hasattr(config, 'low_threshold'):
            low_threshold = config.low_threshold
            high_threshold = config.high_threshold
            aperture_size = config.aperture_size
            l2_gradient = config.l2_gradient
            canny_resolution = config.resolution
            use_gpu = getattr(config, 'gpu', True)
        else:
            low_threshold = config.get('canny_low_threshold', 100)
            high_threshold = config.get('canny_high_threshold', 200)
            aperture_size = config.get('canny_aperture_size', 3)
            l2_gradient = config.get('canny_l2_gradient', False)
            canny_resolution = config.get('canny_resolution', 384)
            use_gpu = config.get('canny_gpu', True)

        # cv2.Canny requires an odd aperture in {3, 5, 7}.
        aperture_size = min(7, max(3, int(aperture_size) | 1))

        original_h, original_w = image_tensor.shape[1], image_tensor.shape[2]

        if canny_resolution < original_h:
            downscaled = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0),
                size=(canny_resolution, canny_resolution),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            process_h, process_w = canny_resolution, canny_resolution
        else:
            downscaled = image_tensor
            process_h, process_w = original_h, original_w

        # Temporal IIR denoise (motion-gated per pixel): sensor noise decorrelates
        # across frames while real detail persists -> SNR boost no threshold can give.
        if self._in_ema is None or self._in_ema.shape != downscaled.shape:
            self._in_ema = downscaled.detach().clone()
        else:
            w = (downscaled - self._in_ema).abs().mean(0, keepdim=True).mul_(8.0).clamp_(0.25, 1.0)
            self._in_ema.lerp_(downscaled, w)
        downscaled = self._in_ema

        edges_chw = None
        if use_gpu and not self._gpu_failed:
            try:
                edges_chw = self._edges_gpu(downscaled, low_threshold, high_threshold, aperture_size, bool(l2_gradient))
            except Exception as e:
                self._gpu_failed = True
                self._gpu = None
                logging.warning(f"[CannyProcessor] GPU path failed ({type(e).__name__}: {e}); using OpenCV")
        if edges_chw is None:
            edges_chw = self._edges_cv2(downscaled, process_h, process_w, low_threshold, high_threshold,
                                        aperture_size, l2_gradient)

        # NEAREST upscale preserves sharp binary edges (critical for ControlNet).
        if canny_resolution < original_h:
            edges_upscaled = torch.nn.functional.interpolate(
                edges_chw.unsqueeze(0),
                size=(original_h, original_w),
                mode='nearest'
            ).squeeze(0)
        else:
            edges_upscaled = edges_chw

        out_shape = (3, original_h, original_w)
        if self._output_buffer is None or self._output_buffer_shape != out_shape:
            self._output_buffer = torch.empty(
                out_shape, device=self.device, dtype=self.torch_dtype
            )
            self._output_buffer_shape = out_shape

        self._output_buffer.copy_(edges_upscaled, non_blocking=True)
        self._cached_result = self._output_buffer
        return self._output_buffer

    # ---- GPU path -----------------------------------------------------------
    def _edges_gpu(self, x: torch.Tensor, low, high, aperture: int, l2: bool) -> torch.Tensor:
        g = self._gpu
        if g is None or g.shape != tuple(x.shape) or g.aperture != aperture or g.l2 != l2:
            g = _GpuCanny(x.shape, aperture, l2, x.device)
            g.capture()
            self._gpu = g
            logging.info(f"[CannyProcessor] GPU Canny captured (CUDA graph) for {tuple(x.shape)}, "
                         f"aperture {aperture}, L2={l2}")
        edges = g(x, low, high)                      # (H, W) half in {0, 1}
        return edges.unsqueeze(0).expand(3, -1, -1).to(self.torch_dtype)

    # ---- OpenCV path (original implementation) --------------------------------
    def _edges_cv2(self, downscaled: torch.Tensor, process_h: int, process_w: int,
                   low_threshold, high_threshold, aperture_size: int, l2_gradient) -> torch.Tensor:
        if self._input_buffer_max is None:
            self._input_buffer_max = np.empty(
                (self.max_buffer_size, self.max_buffer_size, 3), dtype=np.uint8
            )
        input_buffer = self._input_buffer_max[:process_h, :process_w, :]

        # Scale+cast on the GPU and do one uint8 D2H into the pre-allocated buffer.
        gpu_u8 = (downscaled.permute(1, 2, 0) * 255).clamp_(0, 255).to(torch.uint8)
        torch.from_numpy(input_buffer).copy_(gpu_u8)
        del gpu_u8

        cv2.GaussianBlur(input_buffer, (3, 3), 0, dst=input_buffer)
        edges = cv2.Canny(
            input_buffer, low_threshold, high_threshold,
            apertureSize=aperture_size, L2gradient=l2_gradient
        )
        # Drop small isolated fragments (sensor-noise specks), keep real contour chains.
        ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
        if ncomp > 1:
            small = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] < 24) + 1
            if small.size:
                edges[np.isin(labels, small)] = 0

        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges_temp = torch.from_numpy(edges_rgb).float() / 255.0
        return edges_temp.permute(2, 0, 1).to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
