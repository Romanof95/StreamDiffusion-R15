#! fork: https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/utilities.py

#
# Copyright 2022 The HuggingFace Inc. team.
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import gc
from collections import OrderedDict
from typing import *

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import tensorrt as trt
import torch
from cuda import cudart
from PIL import Image
from polygraphy import cuda
from polygraphy.backend.common import bytes_from_path
from ..engine_cache import mark_engine
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_bytes,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.backend.trt import util as trt_util

from .models import CLIP, VAE, BaseModel, UNet, VAEEncoder


TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

torch_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1]
    return None


def onnx_input_names(onnx_path):
    """Names of the graph inputs of an ONNX file (weights not loaded); None if unreadable."""
    try:
        m = onnx.load(onnx_path, load_external_data=False)
        return {i.name for i in m.graph.input}
    except Exception:
        return None


class Engine:
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None
        self.graph = None
        # Captured CUDA graphs keyed by ``graph_key`` (None for the plain single graph).
        self.graphs = {}
        # Latched on CUDA Graph capture failure; permanently falls back to
        # execute_async_v3 for this engine. Replay failures are NOT caught
        # (they indicate corrupted graph state and must surface).
        self._cuda_graph_disabled = False
        # Reusable CUDA event (no timing) used to order downstream torch work
        # after a CUDA-Graph replay WITHOUT a host-blocking stream sync.
        # Lazily created on first replay.
        self._replay_event = None
        # Reusable event ordering the engine after the input copies (see infer()).
        self._input_event = None
        # Profiling: GPU time of the last replayed graph (events on the TRT stream), see graph_ms().
        self.profile_graph = False
        self._graph_events = None

    def release(self):
        """Free everything this engine owns, now: captured CUDA graphs (never freed by
        dropping the Python object), the cudart events, the I/O buffers, the execution
        context (its activation memory) and the deserialized engine. Idempotent; the
        object is unusable afterwards."""
        try:
            self._destroy_graphs()
        except Exception:
            pass
        for name in ("_input_event", "_replay_event"):
            ev = getattr(self, name, None)
            if ev is not None:
                try:
                    cudart.cudaEventDestroy(ev)
                except Exception:
                    pass
                setattr(self, name, None)
        try:
            [buf.free() for buf in getattr(self, "buffers", {}).values() if isinstance(buf, cuda.DeviceArray)]
        except Exception:
            pass
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.context = None   # context before engine
        self.engine = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def refit(self, onnx_path, onnx_refit_path):
        def convert_int64(arr):
            if len(arr.shape) == 0:
                return np.int32(arr)
            return arr

        def add_to_map(refit_dict, name, values):
            if name in refit_dict:
                assert refit_dict[name] is None
                if values.dtype == np.int64:
                    values = convert_int64(values)
                refit_dict[name] = values

        print(f"Refitting TensorRT engine with {onnx_refit_path} weights")
        refit_nodes = gs.import_onnx(onnx.load(onnx_refit_path)).toposort().nodes

        name_map = {}
        for n, node in enumerate(gs.import_onnx(onnx.load(onnx_path)).toposort().nodes):
            refit_node = refit_nodes[n]
            assert node.op == refit_node.op
            if node.op == "Constant":
                name_map[refit_node.outputs[0].name] = node.outputs[0].name
            elif node.op == "Conv":
                if node.inputs[1].__class__ == gs.Constant:
                    name_map[refit_node.name + "_TRTKERNEL"] = node.name + "_TRTKERNEL"
                if node.inputs[2].__class__ == gs.Constant:
                    name_map[refit_node.name + "_TRTBIAS"] = node.name + "_TRTBIAS"
            else:
                for i, inp in enumerate(node.inputs):
                    if inp.__class__ == gs.Constant:
                        name_map[refit_node.inputs[i].name] = inp.name

        def map_name(name):
            if name in name_map:
                return name_map[name]
            return name

        refit_dict = {}
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        all_weights = refitter.get_all()
        for layer_name, role in zip(all_weights[0], all_weights[1]):
            if role == trt.WeightsRole.KERNEL:
                name = layer_name + "_TRTKERNEL"
            elif role == trt.WeightsRole.BIAS:
                name = layer_name + "_TRTBIAS"
            else:
                name = layer_name

            assert name not in refit_dict, "Found duplicate layer: " + name
            refit_dict[name] = None

        for n in refit_nodes:
            if n.op == "Constant":
                name = map_name(n.outputs[0].name)
                print(f"Add Constant {name}\n")
                add_to_map(refit_dict, name, n.outputs[0].values)

            elif n.op == "Conv":
                if n.inputs[1].__class__ == gs.Constant:
                    name = map_name(n.name + "_TRTKERNEL")
                    add_to_map(refit_dict, name, n.inputs[1].values)

                if n.inputs[2].__class__ == gs.Constant:
                    name = map_name(n.name + "_TRTBIAS")
                    add_to_map(refit_dict, name, n.inputs[2].values)

            else:
                for inp in n.inputs:
                    name = map_name(inp.name)
                    if inp.__class__ == gs.Constant:
                        add_to_map(refit_dict, name, inp.values)

        for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
            if weights_role == trt.WeightsRole.KERNEL:
                custom_name = layer_name + "_TRTKERNEL"
            elif weights_role == trt.WeightsRole.BIAS:
                custom_name = layer_name + "_TRTBIAS"
            else:
                custom_name = layer_name

            # Skip Trilu (scalar int64 weights, value=1, CLIP).
            if layer_name.startswith("onnx::Trilu"):
                continue

            if refit_dict[custom_name] is not None:
                refitter.set_weights(layer_name, weights_role, refit_dict[custom_name])
            else:
                print(f"[W] No refit weights for layer: {layer_name}")

        if not refitter.refit_cuda_engine():
            print("Failed to refit!")
            exit(0)

    def build(
        self,
        onnx_path,
        fp16,
        input_profile=None,
        enable_refit=False,
        enable_all_tactics=False,
        timing_cache=None,
        workspace_size=0,
    ):
        import logging
        import os

        logging.info(f"[TensorRT Engine.build] Building engine from {os.path.basename(onnx_path)}")
        logging.info(f"[TensorRT Engine.build] Target: {os.path.basename(self.engine_path)}")

        p = Profile()
        if input_profile:
            present = onnx_input_names(onnx_path)
            for name, dims in input_profile.items():
                assert len(dims) == 3
                if present is not None and name not in present:
                    # Spec input the export dropped (e.g. an unread StreamV2V cache port).
                    continue
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])
                logging.info(f"[TensorRT Engine.build] Profile '{name}': min={dims[0]}, opt={dims[1]}, max={dims[2]}")

        config_kwargs = {}

        if workspace_size > 0:
            config_kwargs["memory_pool_limits"] = {trt.MemoryPoolType.WORKSPACE: workspace_size}
            logging.info(f"[TensorRT Engine.build] Workspace size: {workspace_size / (2**30):.2f} GB")
        if not enable_all_tactics:
            config_kwargs["tactic_sources"] = []

        network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])

        logging.info(f"[TensorRT Engine.build] Building engine (5-15 minutes, no further logs expected)...")
        try:
            engine = engine_from_network(
                network,
                config=CreateConfig(
                    fp16=fp16, refittable=enable_refit, profiles=[p], load_timing_cache=timing_cache, **config_kwargs
                ),
                save_timing_cache=timing_cache,
            )
        except Exception as build_err:
            # The default "Invalid Engine" error hides the real TRT reason.
            # Re-run with VERBOSE polygraphy logging to surface the actual
            # build failure (e.g. dynamic-axis profile conflict), then raise.
            import io
            import contextlib
            from polygraphy.logger import G_LOGGER
            logging.error(f"[TensorRT Engine.build] Build failed: {build_err}")
            logging.error("[TensorRT Engine.build] Re-running with VERBOSE logging...")
            buf = io.StringIO()
            try:
                with G_LOGGER.verbosity(G_LOGGER.VERBOSE), \
                        contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
                    network2 = network_from_onnx_path(
                        onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM]
                    )
                    engine_from_network(
                        network2,
                        config=CreateConfig(
                            fp16=fp16, refittable=enable_refit, profiles=[p],
                            load_timing_cache=timing_cache, **config_kwargs
                        ),
                        save_timing_cache=timing_cache,
                    )
            except Exception:
                pass
            verbose_out = buf.getvalue()
            tail = "\n".join(verbose_out.splitlines()[-60:])
            logging.error(f"[TensorRT Engine.build] VERBOSE TRT output (tail):\n{tail}")
            raise

        save_engine(engine, path=self.engine_path)
        mark_engine(self.engine_path)
        logging.info(f"[TensorRT Engine.build] Engine saved: {os.path.basename(self.engine_path)}")

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))

    def activate(self, reuse_device_memory=None):
        if reuse_device_memory:
            self.context = self.engine.create_execution_context_without_device_memory()
            self.context.device_memory = reuse_device_memory
        else:
            self.context = self.engine.create_execution_context()

    def _destroy_graph(self, key):
        entry = self.graphs.pop(key, None)
        if entry is None:
            return
        graph, instance = entry
        try:
            cudart.cudaGraphExecDestroy(instance)
        except Exception:
            pass
        try:
            cudart.cudaGraphDestroy(graph)
        except Exception:
            pass
        if self.cuda_graph_instance is instance:
            self.cuda_graph_instance = None
            self.graph = None

    def _destroy_graphs(self):
        for graph, instance in self.graphs.values():
            try:
                cudart.cudaGraphExecDestroy(instance)
            except Exception:
                pass
            try:
                cudart.cudaGraphDestroy(graph)
            except Exception:
                pass
        self.graphs = {}
        self.cuda_graph_instance = None
        self.graph = None

    def allocate_buffers(self, shape_dict=None, device="cuda", external=()):
        """Allocate engine I/O buffers. Names in ``external`` are owned by the caller and
        bound per run with ``bind_external`` (StreamV2V ring cache): only their input shape
        is set here."""
        any_reallocated = False
        external = set(external)
        for idx in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(idx)
            if shape_dict and tensor_name in shape_dict:
                shape = shape_dict[tensor_name]
            else:
                shape = self.engine.get_tensor_shape(tensor_name)
            is_input = self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT

            if tensor_name in external:
                if is_input:
                    self.context.set_input_shape(tensor_name, shape)
                if tensor_name in self.tensors:
                    del self.tensors[tensor_name]
                    any_reallocated = True
                continue

            existing = self.tensors.get(tensor_name)
            if existing is not None and tuple(existing.shape) == tuple(shape):
                if is_input:
                    self.context.set_input_shape(tensor_name, shape)
                continue

            if existing is not None:
                del self.tensors[tensor_name]
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            if is_input:
                self.context.set_input_shape(tensor_name, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[tensor_name] = tensor
            any_reallocated = True

        # Any fresh allocation invalidates the captured CUDA Graphs — replaying
        # one would touch deallocated memory. Drop them so the next infer()
        # recaptures against the new binding pointers.
        if any_reallocated:
            self._destroy_graphs()
            # Rebind engine I/O addresses ONCE, on (re)allocation only. The
            # entries in self.tensors are only ever copy_'d into (never
            # reassigned) by infer(), so data_ptr() is stable between
            # reallocations; the context is created once in activate() and
            # never recreated. Doing this here instead of every infer() saves
            # ~160 set_tensor_address FFI calls/frame on the SDXL v2v engine.
            for name, tensor in self.tensors.items():
                self.context.set_tensor_address(name, tensor.data_ptr())

    def bind_external(self, name, tensor):
        """Point an ``external`` I/O tensor at a caller-owned buffer (captured into the
        CUDA graph of the current ``graph_key`` on the next capture)."""
        self.context.set_tensor_address(name, tensor.data_ptr())

    def graph_ms(self):
        """GPU time (ms) of the last replayed CUDA graph, None when not profiled / not replayed.
        Pure kernel time on the engine stream: the difference with the [PERF] phase time is GPU
        idle (host launch latency, other processes)."""
        ev = self._graph_events
        if not self.profile_graph or ev is None:
            return None
        try:
            ev[1].synchronize()
            return float(ev[0].elapsed_time(ev[1]))
        except Exception:
            return None

    def infer(self, feed_dict, stream, use_cuda_graph=False, graph_key=None):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        # Order the engine after the input copies just issued on the torch stream. When that
        # stream is the legacy default stream the ordering is implicit, but the preprocessor
        # orchestrator runs its processors (Depth-Anything engine included) on non-blocking
        # side streams: without this event the engine could read a half-copied input, i.e. an
        # occasional control map mixing two frames.
        if self._input_event is None:
            self._input_event = CUASSERT(
                cudart.cudaEventCreateWithFlags(cudart.cudaEventDisableTiming)
            )
        CUASSERT(cudart.cudaEventRecord(self._input_event, torch.cuda.current_stream().cuda_stream))
        CUASSERT(cudart.cudaStreamWaitEvent(stream.ptr, self._input_event, 0))

        if use_cuda_graph and not self._cuda_graph_disabled:
            entry = self.graphs.get(graph_key)
            if entry is not None:
                # Replay path — uncaught: failure here is a hard bug.
                if self.profile_graph:
                    if self._graph_events is None:
                        self._graph_events = (torch.cuda.Event(enable_timing=True),
                                              torch.cuda.Event(enable_timing=True))
                    _trt_stream = torch.cuda.ExternalStream(int(stream.ptr))
                    self._graph_events[0].record(_trt_stream)
                CUASSERT(cudart.cudaGraphLaunch(entry[1], stream.ptr))
                if self.profile_graph:
                    self._graph_events[1].record(_trt_stream)
                # Order downstream torch work after the replay WITHOUT a
                # host-blocking sync: record an event on the TRT stream and
                # make the current torch stream wait on it. The host returns
                # immediately, so CPU prep of the next stage (scheduler math,
                # postprocess, IPC) overlaps GPU execution — a real win on
                # WDDM where the old cudaStreamSynchronize stalled per engine
                # per frame. GPU-side ordering (e.g. the VAE clamp that reads
                # TRT outputs) is preserved.
                if self._replay_event is None:
                    self._replay_event = CUASSERT(
                        cudart.cudaEventCreateWithFlags(cudart.cudaEventDisableTiming)
                    )
                CUASSERT(cudart.cudaEventRecord(self._replay_event, stream.ptr))
                CUASSERT(
                    cudart.cudaStreamWaitEvent(
                        torch.cuda.current_stream().cuda_stream, self._replay_event, 0
                    )
                )
            else:
                # Capture path — guarded: some engines do unsafe syncs that
                # block stream capture; we latch the disable flag and fall
                # back to non-graph execution on failure. One graph per
                # ``graph_key`` (the StreamV2V ring uses one key per phase,
                # each with its own baked cache addresses).
                try:
                    noerror = self.context.execute_async_v3(stream.ptr)
                    if not noerror:
                        raise ValueError("ERROR: inference failed.")
                    CUASSERT(
                        cudart.cudaStreamBeginCapture(stream.ptr, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
                    )
                    self.context.execute_async_v3(stream.ptr)
                    graph = CUASSERT(cudart.cudaStreamEndCapture(stream.ptr))
                    instance = CUASSERT(cudart.cudaGraphInstantiate(graph, 0))
                    self.graphs[graph_key] = (graph, instance)
                    self.graph, self.cuda_graph_instance = graph, instance
                    import logging
                    import os as _os
                    logging.info(
                        f"[TensorRT Engine] CUDA Graph captured for "
                        f"{_os.path.basename(self.engine_path)}"
                        + (f" (phase {graph_key})" if graph_key is not None else "")
                    )
                except Exception as graph_err:
                    import logging
                    import os as _os
                    logging.warning(
                        f"[TensorRT Engine] CUDA Graph capture failed for "
                        f"{_os.path.basename(self.engine_path)}: {graph_err!r} — "
                        f"falling back to non-graph execution."
                    )
                    self._destroy_graphs()
                    self._cuda_graph_disabled = True
                    # If capture started but didn't end, end it so the stream
                    # isn't left in capturing state.
                    try:
                        cudart.cudaStreamEndCapture(stream.ptr)
                    except Exception:
                        pass
                    noerror = self.context.execute_async_v3(stream.ptr)
                    if not noerror:
                        raise ValueError("ERROR: inference failed.")
        else:
            noerror = self.context.execute_async_v3(stream.ptr)
            if not noerror:
                raise ValueError("ERROR: inference failed.")

        return self.tensors

def decode_images(images: torch.Tensor):
    images = (
        ((images + 1) * 255 / 2).clamp(0, 255).detach().permute(0, 2, 3, 1).round().type(torch.uint8).cpu().numpy()
    )
    return [Image.fromarray(x) for x in images]


def preprocess_image(image: Image.Image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h))
    init_image = np.array(image).astype(np.float32) / 255.0
    init_image = init_image[None].transpose(0, 3, 1, 2)
    init_image = torch.from_numpy(init_image).contiguous()
    return 2.0 * init_image - 1.0


def prepare_mask_and_masked_image(image: Image.Image, mask: Image.Image):
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32).contiguous() / 127.5 - 1.0
    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask).to(dtype=torch.float32).contiguous()

    masked_image = image * (mask < 0.5)

    return mask, masked_image


def create_models(
    model_id: str,
    use_auth_token: Optional[str],
    device: Union[str, torch.device],
    max_batch_size: int,
    unet_in_channels: int = 4,
    embedding_dim: int = 768,
):
    models = {
        "clip": CLIP(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
        "unet": UNet(
            hf_token=use_auth_token,
            fp16=True,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
            unet_dim=unet_in_channels,
        ),
        "vae": VAE(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
        "vae_encoder": VAEEncoder(
            hf_token=use_auth_token,
            device=device,
            max_batch_size=max_batch_size,
            embedding_dim=embedding_dim,
        ),
    }
    return models


def build_engine(
    engine_path: str,
    onnx_opt_path: str,
    model_data: BaseModel,
    opt_image_height: int,
    opt_image_width: int,
    opt_batch_size: int,
    build_static_batch: bool = False,
    build_dynamic_shape: bool = False,
    build_all_tactics: bool = False,
    build_enable_refit: bool = False,
):
    import os
    import logging

    logging.info(f"[TensorRT Build] Starting: {os.path.basename(engine_path)} "
                 f"(model={model_data.name}, batch={opt_batch_size}, res={opt_image_height}x{opt_image_width})")

    _, free_mem, _ = cudart.cudaMemGetInfo()
    GiB = 2**30
    if free_mem > 6 * GiB:
        activation_carveout = 4 * GiB
        max_workspace_size = free_mem - activation_carveout
    else:
        max_workspace_size = 0

    logging.info(f"[TensorRT Build] GPU free memory: {free_mem / GiB:.2f} GB, workspace: {max_workspace_size / GiB:.2f} GB")

    engine = Engine(engine_path)

    input_profile = model_data.get_input_profile(
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        static_batch=build_static_batch,
        static_shape=not build_dynamic_shape,
    )

    engine.build(
        onnx_opt_path,
        fp16=True,
        input_profile=input_profile,
        enable_refit=build_enable_refit,
        enable_all_tactics=build_all_tactics,
        workspace_size=max_workspace_size,
    )
    logging.info(f"[TensorRT Build] Done: {os.path.basename(engine_path)}")

    return engine


def export_onnx(
    model,
    onnx_path: str,
    model_data: BaseModel,
    opt_image_height: int,
    opt_image_width: int,
    opt_batch_size: int,
    onnx_opset: int,
):
    import os
    import logging

    logging.info(f"[ONNX Export] Starting export to {os.path.basename(onnx_path)} "
                 f"(model={model_data.name}, batch={opt_batch_size}, res={opt_image_height}x{opt_image_width})")

    with torch.inference_mode(), torch.autocast("cuda"):
        inputs = model_data.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)

        # ``dynamo=False`` for PyTorch 2.10+ — dynamo export errors on the
        # dynamic_shapes signature used here.
        try:
            torch.onnx.export(
                model,
                inputs,
                onnx_path,
                export_params=True,
                opset_version=onnx_opset,
                do_constant_folding=True,
                input_names=model_data.get_input_names(),
                output_names=model_data.get_output_names(),
                dynamic_axes=model_data.get_dynamic_axes(),
                dynamo=False,
            )
        except Exception as e:
            logging.warning(f"[ONNX Export] Export with dynamic_axes failed: {e}; retrying without")
            torch.onnx.export(
                model,
                inputs,
                onnx_path,
                export_params=True,
                opset_version=onnx_opset,
                do_constant_folding=True,
                input_names=model_data.get_input_names(),
                output_names=model_data.get_output_names(),
                dynamo=False,
            )

    # Use the on-disk file size, NOT ``model_proto.ByteSize()``: the latter
    # crashes with "Failed to serialize proto" on >2 GB SDXL UNets — the
    # exact case we're detecting.
    onnx_size_on_disk_gb = os.path.getsize(onnx_path) / 1e9
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        onnx_size_on_disk_gb += os.path.getsize(data_path) / 1e9
    logging.info(f"[ONNX Export] Model size: {onnx_size_on_disk_gb:.2f} GB")

    needs_external_conversion = (
        onnx_size_on_disk_gb > 2.0 and not os.path.exists(data_path)
    )
    if needs_external_conversion:
        logging.info(f"[ONNX Export] Model > 2GB - converting to external data format...")
        # Stream from disk rather than holding the whole ~5 GB SDXL proto in memory.
        model_proto = onnx.load(onnx_path)
        onnx.save_model(
            model_proto,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(onnx_path) + ".data",
            size_threshold=1024,
            convert_attribute=False,
        )
        del model_proto
        logging.info(f"[ONNX Export] Saved with external data: {data_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    logging.info(f"[ONNX Export] Done: {os.path.basename(onnx_path)}")


def _cleanup_intermediate_onnx(onnx_path: str) -> None:
    """Delete intermediate .onnx + per-tensor external data files left by
    torch.onnx.export after ``optimize_onnx`` produced the .opt.onnx pair.

    PyTorch leaves up to ~5 GB and ~1500 files per SDXL build. Whitelist
    by extension (``.engine`` / ``.opt.onnx`` / ``.opt.onnx.data`` /
    ``.cache``) rather than blacklist by name pattern — the latter would
    miss exotic weight suffixes (BatchNorm running_mean, quant scales, etc.)
    """
    import os
    import logging
    if not onnx_path:
        return
    onnx_dir = os.path.dirname(onnx_path)
    if not onnx_dir or not os.path.isdir(onnx_dir):
        return

    def should_keep(filename: str) -> bool:
        if filename.endswith(".engine"):
            return True
        if filename.endswith(".opt.onnx"):
            return True
        if filename.endswith(".opt.onnx.data"):
            return True
        if filename.endswith(".cache"):
            return True
        if filename.endswith(".trtver"):
            return True
        return False

    deleted_count = 0
    deleted_bytes = 0
    for filename in os.listdir(onnx_dir):
        if should_keep(filename):
            continue
        path = os.path.join(onnx_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            deleted_bytes += os.path.getsize(path)
            os.remove(path)
            deleted_count += 1
        except OSError:
            pass

    if deleted_count > 0:
        logging.info(
            f"[ONNX Cleanup] Removed {deleted_count} intermediate files "
            f"({deleted_bytes / 1e9:.2f} GB) — kept .engine, .opt.onnx, .opt.onnx.data"
        )


def optimize_onnx(
    onnx_path: str,
    onnx_opt_path: str,
    model_data: BaseModel,
):
    import os
    import logging

    logging.info(f"[ONNX Optimize] Starting optimization of {os.path.basename(onnx_path)}")

    # Tiny .onnx (<100 MB) means weights are external — large model path.
    onnx_file_size_gb = os.path.getsize(onnx_path) / 1e9
    is_large_model = onnx_file_size_gb < 0.1

    if is_large_model:
        logging.info(f"[ONNX Optimize] Large model with external weights")
        onnx_graph = onnx.load(onnx_path, load_external_data=True)
        onnx_opt_graph = model_data.optimize(onnx_graph)
        onnx.save_model(
            onnx_opt_graph,
            onnx_opt_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(onnx_opt_path) + ".data",
            size_threshold=1024,
            convert_attribute=False,
        )
    else:
        onnx_graph = onnx.load(onnx_path)
        onnx_opt_graph = model_data.optimize(onnx_graph)
        onnx.save(onnx_opt_graph, onnx_opt_path)

    if os.path.exists(onnx_opt_path):
        _cleanup_intermediate_onnx(onnx_path)

    del onnx_opt_graph
    gc.collect()
    torch.cuda.empty_cache()
    logging.info(f"[ONNX Optimize] Done: {os.path.basename(onnx_opt_path)}")
