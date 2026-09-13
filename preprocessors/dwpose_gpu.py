"""GPU end-to-end DWPose: letterbox, person crops, SimCC decode and skeleton rasterization
in torch, around the two TensorRT engines of ``dwpose_trt``.

The onnxruntime/OpenCV path (easy_dwpose) downloads the full frame to the CPU, letterboxes
it in numpy, warps each person crop with cv2, decodes the SimCC maps in numpy and draws the
skeleton with cv2 primitives: ~13 ms of CPU per frame around ~4 ms of TensorRT. Here the
frame never leaves the GPU; only the keypoints (a few hundred floats) come back for the
temporal filtering, and the skeleton is rasterized as batched distance fields.
"""
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .dwpose_trt import TrtSession, TrtWholebody

_DET = 640
_POSE_W, _POSE_H = 288, 384
_MEAN = (123.675, 116.28, 103.53)
_STD = (58.395, 57.12, 57.375)
_EPS = 0.01

# easy_dwpose/draw/openpose.py tables (RGB)
_LIMB_SEQ = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12],
             [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
           [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
           [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
_HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
               [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18],
               [18, 19], [19, 20]]


def _hsv_to_rgb(h: float) -> List[float]:
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    q, t = 1.0 - f, f
    r, g, b = [(1, t, 0), (q, 1, 0), (0, 1, t), (0, q, 1), (t, 0, 1), (1, 0, q)][i]
    return [255.0 * r, 255.0 * g, 255.0 * b]


_HAND_COLORS = [_hsv_to_rgb(ie / float(len(_HAND_EDGES))) for ie in range(len(_HAND_EDGES))]


class GpuWholebody(TrtWholebody):
    """DWPose (YOLOX-S + DW-LL) on a GPU image. ``run(img)`` takes (3, H, W) float RGB in
    [0, 255] at the detect resolution and returns numpy keypoints (n, 134, 2) in pixels and
    scores (n, 134), OpenPose ordering (neck inserted), like TrtWholebody.__call__."""

    def __init__(self, session_det: TrtSession, session_pose: TrtSession):
        super().__init__(session_det, session_pose)
        dev = session_det.device
        self._mean = torch.tensor(_MEAN, device=dev).view(1, 3, 1, 1)
        self._std = torch.tensor(_STD, device=dev).view(1, 3, 1, 1)
        us = torch.arange(_POSE_W, device=dev, dtype=torch.float32) - _POSE_W * 0.5
        vs = torch.arange(_POSE_H, device=dev, dtype=torch.float32) - _POSE_H * 0.5
        self._crop_v, self._crop_u = torch.meshgrid(vs, us, indexing="ij")  # (384, 288)

    def _execute(self, session: TrtSession):
        """Run a static-shape session on the current torch stream, as a CUDA graph after the
        first call (the enqueue of ~200 layers costs ~2 ms of host time under WDDM)."""
        graph = getattr(session, "_graph", None)
        if graph is not None:
            graph.replay()
            return
        stream = torch.cuda.current_stream()
        if not session.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        if getattr(session, "_graph_failed", False):
            return
        try:
            stream.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):   # captures on a side stream; replay() runs on the current one
                if not session.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                    raise RuntimeError("TensorRT execution failed during capture")
            session._graph = g
        except Exception as e:
            import logging
            logging.warning(f"[DWPose GPU] CUDA graph capture failed ({type(e).__name__}: {e}); plain execution")
            session._graph_failed = True

    @torch.inference_mode()
    def detect_gpu(self, img: torch.Tensor) -> torch.Tensor:
        """(3, H, W) float [0, 255] -> person boxes (n, 4) xyxy in image pixels (GPU)."""
        from torchvision.ops import nms
        H, W = img.shape[-2:]
        r = min(_DET / H, _DET / W)
        nh, nw = int(H * r), int(W * r)
        resized = F.interpolate(img[None], size=(nh, nw), mode="bilinear", align_corners=False)
        s = self.session_det
        buf = s.buffers["images"]
        buf.fill_(114.0)
        buf[:, :, :nh, :nw] = resized.to(buf.dtype)
        self._execute(s)
        out = s.buffers["output"][0].float()                       # (8400, 85)
        xy = (out[:, :2] + self._grid) * self._stride
        wh = torch.exp(out[:, 2:4]) * self._stride
        score = out[:, 4] * out[:, 5]
        boxes = torch.cat((xy - wh / 2, xy + wh / 2), 1) / r
        m = score > 0.1
        boxes, score = boxes[m], score[m]
        keep = nms(boxes, score, 0.45)
        boxes, score = boxes[keep], score[keep]
        return boxes[score > 0.3]

    @torch.inference_mode()
    def pose_gpu(self, img: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Crops (affine, 288x384, aspect-fixed, padding 1.25) -> DW-LL -> SimCC decode.
        Returns keypoints (n, 133, 2) in image pixels and scores (n, 133), on the GPU."""
        H, W = img.shape[-2:]
        dev = img.device
        if boxes.shape[0] == 0:
            boxes = torch.tensor([[0.0, 0.0, float(W), float(H)]], device=dev)
        n = boxes.shape[0]
        center = (boxes[:, :2] + boxes[:, 2:]) * 0.5                 # (n, 2)
        scale = (boxes[:, 2:] - boxes[:, :2]) * 1.25                 # (n, 2) w, h
        aspect = _POSE_W / _POSE_H
        w, h = scale[:, 0], scale[:, 1]
        sw = torch.where(w > h * aspect, w, h * aspect)              # _fix_aspect_ratio
        k = sw / _POSE_W                                             # image px per crop px
        # crop pixel (u, v) samples image pixel center + (u - 144, v - 192) * k
        sx = center[:, 0].view(n, 1, 1) + self._crop_u[None] * k.view(n, 1, 1)
        sy = center[:, 1].view(n, 1, 1) + self._crop_v[None] * k.view(n, 1, 1)
        grid = torch.stack(((sx + 0.5) * (2.0 / W) - 1.0, (sy + 0.5) * (2.0 / H) - 1.0), dim=-1)
        crops = F.grid_sample(img[None].expand(n, -1, -1, -1), grid, mode="bilinear",
                              padding_mode="zeros", align_corners=False)
        crops = (crops - self._mean) / self._std                     # (n, 3, 384, 288)
        s = self.session_pose
        inp = s.buffers[s.input_names[0]]
        ox, oy = (s.buffers[name] for name in s.output_names[:2])
        kps, vals = [], []
        for i in range(n):
            inp.copy_(crops[i:i + 1].to(inp.dtype))
            self._execute(s)
            simcc_x, simcc_y = ox[0].float(), oy[0].float()          # (133, 576), (133, 768)
            mx, lx = simcc_x.max(dim=1)
            my, ly = simcc_y.max(dim=1)
            val = torch.minimum(mx, my)
            loc = torch.stack((lx, ly), dim=-1).float() * 0.5       # simcc_split_ratio 2
            kp = center[i] + (loc - torch.tensor([_POSE_W * 0.5, _POSE_H * 0.5], device=dev)) * k[i]
            kp = torch.where((val <= 0.0).unsqueeze(-1), torch.full_like(kp, -1.0), kp)
            kps.append(kp)
            vals.append(val)
        return torch.stack(kps), torch.stack(vals)

    @torch.inference_mode()
    def run(self, img: torch.Tensor):
        boxes = self.detect_gpu(img)
        keypoints, scores = self.pose_gpu(img, boxes)
        kp_s = torch.cat((keypoints, scores.unsqueeze(-1)), dim=-1).cpu().numpy()   # single small D2H
        keypoints_info = kp_s
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        neck[:, 2:4] = np.logical_and(keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        return new_keypoints_info[..., :2], new_keypoints_info[..., 2]


def make_gpu_wholebody(model_det_path: str, model_pose_path: str, device, warn=None) -> GpuWholebody:
    from .dwpose_trt import make_trt_sessions
    sd, sp = make_trt_sessions(model_det_path, model_pose_path, device, warn=warn)
    return GpuWholebody(sd, sp)


# --------------------------------------------------------------------------- rasterizer
def _paint(canvas: torch.Tensor, masks: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
    """Overwrite canvas (H, W, 3) where masks (N, H, W) hold, later primitives winning
    (cv2 draw order)."""
    if masks.shape[0] == 0:
        return canvas
    n = masks.shape[0]
    order = torch.arange(1, n + 1, device=canvas.device, dtype=torch.int32).view(n, 1, 1)
    idx = (masks.to(torch.int32) * order).amax(0)                   # 0 = untouched
    palette = torch.cat((torch.zeros(1, 3, device=canvas.device), colors), 0)
    return torch.where((idx > 0).unsqueeze(-1), palette[idx.long()], canvas)


def _ellipses(p0: torch.Tensor, p1: torch.Tensor, half_width: float, ys: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """Filled rotated ellipses with semi-axes (|p1-p0|/2, half_width): cv2.ellipse2Poly limbs."""
    n = p0.shape[0]
    m = (p0 + p1) * 0.5
    d = p1 - p0
    length = d.norm(dim=1)
    a = torch.floor(length * 0.5).clamp(min=0.5)                    # int(length / 2) in cv2
    direction = d / length.clamp(min=1e-6).unsqueeze(1)
    dx = xs - m[:, 0].view(n, 1, 1)
    dy = ys - m[:, 1].view(n, 1, 1)
    u = dx * direction[:, 0].view(n, 1, 1) + dy * direction[:, 1].view(n, 1, 1)
    v = -dx * direction[:, 1].view(n, 1, 1) + dy * direction[:, 0].view(n, 1, 1)
    return (u / a.view(n, 1, 1)) ** 2 + (v / half_width) ** 2 <= 1.0


def _capsules(p0: torch.Tensor, p1: torch.Tensor, radius: float, ys: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """Thick line segments (cv2.line with thickness 2*radius)."""
    n = p0.shape[0]
    d = p1 - p0
    l2 = (d * d).sum(1).clamp(min=1e-6)
    dx = xs - p0[:, 0].view(n, 1, 1)
    dy = ys - p0[:, 1].view(n, 1, 1)
    t = ((dx * d[:, 0].view(n, 1, 1) + dy * d[:, 1].view(n, 1, 1)) / l2.view(n, 1, 1)).clamp(0.0, 1.0)
    ex = dx - t * d[:, 0].view(n, 1, 1)
    ey = dy - t * d[:, 1].view(n, 1, 1)
    return ex * ex + ey * ey <= radius * radius


def _discs(p: torch.Tensor, radius: float, ys: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    n = p.shape[0]
    dx = xs - p[:, 0].view(n, 1, 1)
    dy = ys - p[:, 1].view(n, 1, 1)
    return dx * dx + dy * dy <= radius * radius


@torch.inference_mode()
def draw_pose_gpu(pose: dict, height: int, width: int, device, include_face: bool = True,
                  include_hands: bool = True) -> torch.Tensor:
    """easy_dwpose.draw.openpose.draw_pose on the GPU. ``pose`` is the numpy dict built by
    OptimizedDWposeDetector (normalized coordinates, -1 = filtered). Returns (3, H, W)
    float RGB in [0, 255]."""
    ys = torch.arange(height, device=device, dtype=torch.float32).view(1, height, 1)
    xs = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, width)
    canvas = torch.zeros(height, width, 3, device=device)
    f32 = lambda a: torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=device)

    bodies = np.asarray(pose["bodies"], dtype=np.float32)          # (n*18, 2) normalized
    subset = np.asarray(pose["body_scores"])                        # (n, 18) index or -1
    px = bodies * np.array([width, height], dtype=np.float32)
    # limbs
    p0, p1, col = [], [], []
    for i, (a, b) in enumerate(_LIMB_SEQ):
        for n in range(subset.shape[0]):
            ia, ib = int(subset[n][a - 1]), int(subset[n][b - 1])
            if ia == -1 or ib == -1:
                continue
            xa, ya = px[ia]; xb, yb = px[ib]
            p0.append((math.floor(xa), math.floor(ya))); p1.append((math.floor(xb), math.floor(yb)))
            col.append(_COLORS[i])
    if p0:
        canvas = _paint(canvas, _ellipses(f32(p0), f32(p1), 4.0, ys, xs), f32(col))
    canvas = canvas * 0.6
    # body joints
    pts, col = [], []
    for i in range(18):
        for n in range(subset.shape[0]):
            idx = int(subset[n][i])
            if idx == -1:
                continue
            pts.append((math.floor(px[idx][0]), math.floor(px[idx][1]))); col.append(_COLORS[i])
    if pts:
        canvas = _paint(canvas, _discs(f32(pts), 4.0, ys, xs), f32(col))
    # face
    if include_face:
        faces = np.asarray(pose["faces"], dtype=np.float32).reshape(-1, 2)
        ok = (faces[:, 0] > _EPS) & (faces[:, 1] > _EPS)
        if ok.any():
            pts = np.floor(faces[ok] * np.array([width, height], dtype=np.float32))
            canvas = _paint(canvas, _discs(f32(pts), 3.0, ys, xs), f32([[255.0, 255.0, 255.0]] * len(pts)))
    # hands
    if include_hands:
        hands = np.asarray(pose["hands"], dtype=np.float32)          # (2n, 21, 2)
        hpx = np.floor(hands * np.array([width, height], dtype=np.float32))
        p0, p1, col, pts = [], [], [], []
        for hand in range(hands.shape[0]):
            for ie, (a, b) in enumerate(_HAND_EDGES):
                xa, ya = hpx[hand][a]; xb, yb = hpx[hand][b]
                if xa > _EPS and ya > _EPS and xb > _EPS and yb > _EPS:
                    p0.append((xa, ya)); p1.append((xb, yb)); col.append(_HAND_COLORS[ie])
            for x, y in hpx[hand]:
                if x > _EPS and y > _EPS:
                    pts.append((x, y))
        if p0:
            canvas = _paint(canvas, _capsules(f32(p0), f32(p1), 1.0, ys, xs), f32(col))
        if pts:
            canvas = _paint(canvas, _discs(f32(pts), 4.0, ys, xs), f32([[0.0, 0.0, 255.0]] * len(pts)))
    return canvas.permute(2, 0, 1)
