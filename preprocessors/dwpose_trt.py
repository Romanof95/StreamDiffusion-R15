"""TensorRT backend for the DWPose (OpenPose) preprocessor.

easy_dwpose's ``Wholebody`` only touches its two onnxruntime sessions through
``get_inputs()[0].name``, ``get_outputs()`` and ``run(output_names, feed)``.
``TrtSession`` reproduces that interface on top of a TensorRT engine, so the
detector (YOLOX-S, 640x640) and the pose model (DW-LL, 384x288, one person per
call) run as fp16 TensorRT engines while easy_dwpose keeps its own numpy
pre/post-processing. Engines are built once from the ONNX files in
``checkpoints/`` and cached under ``tensorrt_cache/dwpose/`` with the usual
``.trtver`` marker. Measured on an RTX 5080 @512 detect resolution:
onnxruntime CUDA 22 ms/frame -> TensorRT ~5 ms/frame.
"""
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from pipeline.acceleration.engine_cache import engine_ready, mark_engine

PACKAGE_DIR = Path(__file__).resolve().parent.parent
ENGINE_DIR = PACKAGE_DIR / "tensorrt_cache" / "dwpose"


class _NodeArg:
    """Minimal onnxruntime NodeArg look-alike (name, shape, type)."""
    __slots__ = ("name", "shape", "type")

    def __init__(self, name: str, shape=None, type_=None):
        self.name = name
        self.shape = list(shape) if shape is not None else None
        self.type = type_


class TrtSession:
    """onnxruntime.InferenceSession look-alike backed by a static-shape TensorRT engine."""

    def __init__(self, engine_path: str, device):
        import tensorrt as trt

        self._trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"cannot deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.device = torch.device(device if device is not None else "cuda")
        self.stream = torch.cuda.Stream(device=self.device)
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.buffers: Dict[str, torch.Tensor] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        for name in self.input_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                shape = tuple(self.engine.get_tensor_profile_shape(name, 0)[1])  # opt shape
            self.context.set_input_shape(name, shape)
            self.buffers[name] = torch.empty(shape, dtype=self._dtype(name), device=self.device)
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            self.buffers[name] = torch.empty(shape, dtype=self._dtype(name), device=self.device)
        for name, t in self.buffers.items():
            self.context.set_tensor_address(name, t.data_ptr())

    def _dtype(self, name: str) -> torch.dtype:
        trt = self._trt
        table = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
                 trt.DataType.INT32: torch.int32, trt.DataType.BOOL: torch.bool}
        if hasattr(trt.DataType, "INT64"):
            table[trt.DataType.INT64] = torch.int64
        return table[self.engine.get_tensor_dtype(name)]

    # onnxruntime-compatible surface -------------------------------------------------
    def _nodearg(self, n: str) -> _NodeArg:
        t = self.buffers[n]
        return _NodeArg(n, tuple(t.shape), "tensor(float16)" if t.dtype == torch.float16 else "tensor(float)")

    def get_inputs(self) -> List[_NodeArg]:
        return [self._nodearg(n) for n in self.input_names]

    def get_outputs(self) -> List[_NodeArg]:
        return [self._nodearg(n) for n in self.output_names]

    def run(self, output_names: Optional[Sequence[str]], feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        names = list(output_names) if output_names else self.output_names
        with torch.cuda.stream(self.stream):
            for name, arr in feed.items():
                buf = self.buffers[name]
                src = torch.as_tensor(np.ascontiguousarray(arr)).reshape(buf.shape)
                buf.copy_(src.to(buf.dtype))
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            outs = [self.buffers[n].float().cpu().numpy() for n in names]  # .cpu() syncs this stream
        return outs


def _build_engine(onnx_path: str, engine_path: str, profile_shapes: Optional[Dict[str, tuple]] = None) -> None:
    import tensorrt as trt
    from polygraphy.backend.trt import CreateConfig, Profile, engine_from_network, network_from_onnx_path, save_engine

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    t0 = time.time()
    network = network_from_onnx_path(onnx_path)
    profiles = None
    if profile_shapes:
        p = Profile()
        for name, shape in profile_shapes.items():
            p.add(name, min=shape, opt=shape, max=shape)
        profiles = [p]
    engine = engine_from_network(network, config=CreateConfig(fp16=True, profiles=profiles, tactic_sources=[]))
    save_engine(engine, path=engine_path)
    mark_engine(engine_path)
    logging.info(f"[DWPose TRT] built {os.path.basename(engine_path)} in {time.time() - t0:.0f}s")


class TrtWholebody:
    """easy_dwpose ``Wholebody`` equivalent with TensorRT sessions and a GPU YOLOX post-process.

    The person detector's decode + NMS runs in torch on the raw (1, 8400, 85) engine output
    (only class 0 / person is ever used downstream), so no 8400x85 tensor crosses to the
    CPU and no numpy NMS runs per frame. The pose stage keeps easy_dwpose's numpy code.
    """

    def __init__(self, session_det: TrtSession, session_pose: TrtSession):
        self.session_det = session_det
        self.session_pose = session_pose
        dev = session_det.device
        grids, strides = [], []
        for stride in (8, 16, 32):
            h = w = 640 // stride
            yv, xv = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
            grids.append(torch.stack((xv, yv), 2).reshape(-1, 2).float())
            strides.append(torch.full((h * w, 1), float(stride), device=dev))
        self._grid = torch.cat(grids, 0)          # (8400, 2)
        self._stride = torch.cat(strides, 0)      # (8400, 1)

    def detect(self, oriImg: np.ndarray) -> np.ndarray:
        import cv2
        from torchvision.ops import nms
        from easy_dwpose.body_estimation.detector import preprocess

        img, ratio = preprocess(oriImg, (640, 640))
        s = self.session_det
        with torch.cuda.stream(s.stream):
            s.buffers["images"].copy_(torch.from_numpy(img[None]))
            if not s.context.execute_async_v3(s.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            out = s.buffers["output"][0]                                  # (8400, 85) on GPU
            xy = (out[:, :2] + self._grid) * self._stride
            wh = torch.exp(out[:, 2:4]) * self._stride
            score = out[:, 4] * out[:, 5]                                 # objectness x person
            boxes = torch.cat((xy - wh / 2, xy + wh / 2), 1) / ratio
            m = score > 0.1
            boxes, score = boxes[m], score[m]
            keep = nms(boxes, score, 0.45)
            boxes, score = boxes[keep], score[keep]
            keep = score > 0.3
            res = boxes[keep].cpu().numpy()                               # syncs the stream
        return res

    def __call__(self, oriImg: np.ndarray):
        from easy_dwpose.body_estimation.pose import inference_pose

        det_result = self.detect(oriImg)
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)
        keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        neck[:, 2:4] = np.logical_and(keypoints_info[:, 5, 2:4] > 0.3, keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info
        return keypoints_info[..., :2], keypoints_info[..., 2]


def make_trt_wholebody(model_det_path: str, model_pose_path: str, device, warn=None) -> "TrtWholebody":
    """Drop-in replacement for easy_dwpose.Wholebody backed by TensorRT (see TrtWholebody)."""
    sd, sp = make_trt_sessions(model_det_path, model_pose_path, device, warn=warn)
    return TrtWholebody(sd, sp)


def make_trt_sessions(model_det_path: str, model_pose_path: str, device, warn=None):
    """Build (once) and load the TensorRT engines for YOLOX-S and DW-LL; returns two TrtSession."""
    det_engine = str(ENGINE_DIR / "yolox_s_640.engine")
    pose_engine = str(ENGINE_DIR / "dw-ll_ucoco_384x288.engine")
    if not engine_ready(det_engine) or not engine_ready(pose_engine):
        if warn is not None:
            warn(True, "Building TensorRT engines for DWPose (YOLOX-S + DW-LL) - one-time, ~2 min")
        try:
            if not engine_ready(det_engine):
                _build_engine(model_det_path, det_engine)                                   # static 1x3x640x640
            if not engine_ready(pose_engine):
                _build_engine(model_pose_path, pose_engine, {"input": (1, 3, 384, 288)})    # dynamic batch -> 1
        finally:
            if warn is not None:
                warn(False)
    return TrtSession(det_engine, device), TrtSession(pose_engine, device)
