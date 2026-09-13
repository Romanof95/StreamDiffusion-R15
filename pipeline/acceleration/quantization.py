"""Block-scaled low-precision TensorRT engines (MXFP8 / NVFP4) via NVIDIA ModelOpt.

Why block-scaled formats: on RTX 50 (SM120) TensorRT only ships native FP8/FP4
GEMM kernels for block-scaled layouts (MXFP8 = E4M3 elements + E8M0 scale per 32,
NVFP4 = E2M1 elements + E4M3 scale per 16). Per-tensor FP8 falls back to Ada
(sm89) kernels and gains nothing. Measured on an RTX 5080, SDXL UNet 1024x1024,
batch 1, TensorRT 10.16: fp16 41.2 ms, MXFP8 36.0 ms (-12.6%), NVFP4 29.1 ms (-29%).

Pipeline: quantize the fused PyTorch module in place (Linear layers only, convs
and time/text embeddings stay fp16) -> static fp16-native ONNX export with the
TensorRT custom dynamic-quantize ops -> ModelOpt post-processing (packed weights
+ block scales) -> strongly-typed TensorRT build. The quantized PyTorch module is
slow in eager mode (simulated quantization) and is meant to be discarded once
the engine exists.

Windows notes: ModelOpt's CUDA/Triton extensions need an MSVC toolchain, so the
simulated-quantization ops are replaced by pure-PyTorch fallbacks. They only run
while tracing the ONNX graph (and during NVFP4 calibration, where quantizers are
in collect-only mode), never at inference.
"""
import gc
import logging
import os
import re
import shutil
import time
from contextlib import nullcontext
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch

from .engine_cache import mark_engine

PRECISIONS = ("fp16", "mxfp8", "nvfp4")

# Layers kept in high precision (same policy as NVIDIA's diffusers quantization example).
_HIGH_PRECISION_LAYERS = re.compile(
    r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|"
    r"pos_embed|time_text_embed|context_embedder|norm_out|x_embedder).*"
)


def normalize_precision(precision) -> str:
    """Map user input to one of PRECISIONS. 'fp8' means MXFP8 (the only FP8 with SM120 kernels)."""
    p = (str(precision) if precision is not None else "fp16").strip().lower()
    if p in ("", "fp16", "half", "none", "default"):
        return "fp16"
    if p in ("mxfp8", "fp8", "e4m3"):
        return "mxfp8"
    if p in ("nvfp4", "fp4", "e2m1"):
        return "nvfp4"
    raise ValueError(f"Unknown precision {precision!r}; expected one of {PRECISIONS}")


def precision_suffix(precision: str) -> str:
    """Cache-key suffix so fp16 and quantized engines coexist."""
    p = normalize_precision(precision)
    return "" if p == "fp16" else f"--prec-{p}"


def quantization_available() -> bool:
    try:
        import modelopt.torch.quantization  # noqa: F401
        from modelopt.onnx.export import MXFP8QuantExporter, NVFP4QuantExporter  # noqa: F401
        return True
    except Exception as e:
        logging.warning(f"[Quant] nvidia-modelopt (+ onnx exporter deps) not available: {e}")
        return False


# --------------------------------------------------------------------------- fallbacks
def _e2m1(y: torch.Tensor) -> torch.Tensor:
    a = y.abs()
    lv = torch.where(a < 0.25, 0.0, torch.where(a < 0.75, 0.5, torch.where(a < 1.25, 1.0, torch.where(
        a < 1.75, 1.5, torch.where(a < 2.5, 2.0, torch.where(a < 3.5, 3.0, torch.where(a < 5.0, 4.0, 6.0)))))))
    return torch.sign(y) * lv


def _py_dynamic_block_quantize(inputs, block_size, amax, num_bits, exponent_bits, scale_num_bits, scale_exponent_bits):
    """Simulated block quantization (MXFP8 or NVFP4) along the last dim. Tracing only."""
    x = inputs
    last = x.shape[-1]
    pad = (-last) % block_size
    xf = x.float()
    if pad:
        xf = torch.nn.functional.pad(xf, (0, pad))
    xb = xf.reshape(*xf.shape[:-1], -1, block_size)
    bamax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    if num_bits == 4 and exponent_bits == 2:
        g = (float(amax) / (6.0 * 448.0)) if amax is not None and float(amax) > 0 else 1.0
        bs = ((bamax / 6.0) / g).to(torch.float8_e4m3fn).float().clamp(min=2 ** -9) * g
        q = _e2m1(xb / bs) * bs
    else:
        scale = torch.exp2(torch.ceil(torch.log2(bamax / 448.0)))
        q = (xb / scale).to(torch.float8_e4m3fn).float() * scale
    out = q.reshape(*xf.shape)
    if pad:
        out = out[..., :last]
    return out.to(x.dtype)


def _install_python_fallbacks() -> None:
    import modelopt.torch.quantization.tensor_quant as tq

    if getattr(tq, "_smode_fallbacks_installed", False):
        return

    def _quantize_op(inputs, amax, num_bits=8, exponent_bits=0, unsigned=False, narrow_range=True):
        if num_bits == 8 and exponent_bits == 4:
            return tq.fp8_eager(inputs, amax)
        raise AttributeError("quantize_op disabled: use the legacy python path")  # caught by ModelOpt

    tq.quantize_op = _quantize_op
    tq.dynamic_block_quantize_op = _py_dynamic_block_quantize
    tq._smode_fallbacks_installed = True
    logging.info("[Quant] ModelOpt simulated-quant ops replaced by pure-PyTorch fallbacks (no MSVC needed)")


# --------------------------------------------------------------------------- quantize
def bake_lora(pipe) -> None:
    """Drop PEFT LoRA wrappers after fuse_lora (fused weights are kept)."""
    try:
        pipe.unload_lora_weights()
        logging.info("[Quant] LoRA baked: PEFT wrappers unloaded, fused weights kept")
    except Exception as e:
        logging.warning(f"[Quant] unload_lora_weights failed ({e}); continuing with PEFT wrappers")


def quantize_model(model: torch.nn.Module, precision: str,
                   forward_loop: Optional[Callable[[torch.nn.Module], None]] = None) -> int:
    """Quantize Linear layers of `model` in place. Returns the number of enabled quantizers."""
    import modelopt.torch.quantization as mtq

    p = normalize_precision(precision)
    if p == "fp16":
        return 0
    _install_python_fallbacks()
    if p == "mxfp8":
        elem = {"num_bits": (4, 3), "block_sizes": {-1: 32, "type": "dynamic", "scale_bits": (8, 0)}}
        algorithm = None  # dynamic scales: no calibration needed
    else:
        elem = {"num_bits": (2, 1), "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}}
        algorithm = "max"  # per-tensor global scale from a short calibration
        if forward_loop is None:
            raise ValueError("NVFP4 needs a calibration forward_loop")
    cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": elem},
            {"quantizer_name": "*input_quantizer", "cfg": elem},
            {"quantizer_name": "*output_quantizer", "enable": False},
            # No block-scaled conv kernels (and fp8 convs are slower): keep every conv in fp16,
            # matched by class (Union ControlNet convs are named *_blocks.N / cond_embedding.*).
            {"quantizer_name": "*conv*", "enable": False},
            {"quantizer_name": "*", "parent_class": "nn.Conv1d", "enable": False},
            {"quantizer_name": "*", "parent_class": "nn.Conv2d", "enable": False},
            {"quantizer_name": "*", "parent_class": "nn.Conv3d", "enable": False},
            {"quantizer_name": "*", "parent_class": "nn.ConvTranspose2d", "enable": False},
        ],
        "algorithm": algorithm,
    }
    t0 = time.time()
    if algorithm is None:
        mtq.quantize(model, cfg)
    else:
        mtq.quantize(model, cfg, forward_loop)
    mtq.disable_quantizer(model, lambda name: _HIGH_PRECISION_LAYERS.match(name) is not None)
    qs = [m for _, m in model.named_modules() if type(m).__name__ == "TensorQuantizer"]
    n_on = sum(1 for m in qs if m.is_enabled)
    logging.info(f"[Quant] {p}: {n_on}/{len(qs)} quantizers enabled in {time.time() - t0:.0f}s")
    if n_on == 0:
        raise RuntimeError("quantization produced no enabled quantizer")
    return n_on


# --------------------------------------------------------------------------- export
def export_quantized_onnx(module: torch.nn.Module, quantized_model: torch.nn.Module,
                          inputs: Sequence[torch.Tensor], input_names: List[str], output_names: List[str],
                          onnx_path: str, precision: str) -> None:
    """Static fp16-native ONNX export of a quantized module, post-processed for TensorRT."""
    import onnx
    import onnx_graphsurgeon as gs
    from modelopt.onnx.export import MXFP8QuantExporter, NVFP4QuantExporter
    from modelopt.torch.quantization.export_onnx import configure_linear_module_onnx_quantizers

    p = normalize_precision(precision)
    _install_python_fallbacks()
    module = module.to("cuda", torch.float16).eval()
    inputs = tuple(t.to("cuda").half() if t.is_floating_point() else t.to("cuda") for t in inputs)

    tmp_dir = onnx_path + ".export_tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)
    tmp = os.path.join(tmp_dir, "model.onnx")
    t0 = time.time()
    logging.info(f"[Quant] ONNX export ({p}, static, opset 20): {os.path.basename(onnx_path)}")
    with torch.inference_mode(), configure_linear_module_onnx_quantizers(quantized_model):
        torch.onnx.export(
            module, inputs, tmp, export_params=True, opset_version=20, do_constant_folding=False,
            input_names=list(input_names), output_names=list(output_names), dynamic_axes=None, dynamo=False,
        )
    logging.info(f"[Quant] torch.onnx.export done in {time.time() - t0:.0f}s")
    gc.collect()
    torch.cuda.empty_cache()

    m = onnx.load(tmp, load_external_data=True)
    g = gs.import_onnx(m)
    g.cleanup().toposort()
    m = gs.export_onnx(g)
    del g
    exporter = NVFP4QuantExporter if p == "nvfp4" else MXFP8QuantExporter
    m = exporter.process_model(m)  # packed low-precision weights + block-scale initializers
    ops = {}
    for n in m.graph.node:
        if "uant" in n.op_type or n.op_type.startswith("TRT_"):
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
    logging.info("[Quant] quantization ops: " + ", ".join(f"{k}={v}" for k, v in ops.items()))
    if not ops:
        raise RuntimeError("no quantization op in the exported graph")

    # Strongly-typed TensorRT: normalization scale/bias must have the input's dtype.
    inits = {t.name: t for t in m.graph.initializer}
    consts = {n.output[0]: n for n in m.graph.node if n.op_type == "Constant"}
    producer = {o: n for n in m.graph.node for o in n.output}
    n_norm = 0
    for n in m.graph.node:
        if n.op_type in ("LayerNormalization", "GroupNormalization", "InstanceNormalization"):
            src = producer.get(n.input[0])
            in_fp32 = src is not None and src.op_type == "Cast" and any(
                a.name == "to" and a.i == onnx.TensorProto.FLOAT for a in src.attribute)
            want = np.float32 if in_fp32 else np.float16
            want_dt = onnx.TensorProto.FLOAT if in_fp32 else onnx.TensorProto.FLOAT16
            for inp in n.input[1:]:
                t = inits.get(inp) or (consts[inp].attribute[0].t if inp in consts else None)
                if t is not None and t.data_type != want_dt and t.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16):
                    arr = onnx.numpy_helper.to_array(t).astype(want)
                    t.CopyFrom(onnx.numpy_helper.from_array(arr, t.name))
                    n_norm += 1
    if n_norm:
        logging.info(f"[Quant] normalization params retyped: {n_norm}")

    data_name = os.path.basename(onnx_path) + ".data"
    for stale in (onnx_path, onnx_path + ".data"):
        if os.path.exists(stale):
            os.remove(stale)  # onnx appends to an existing external-data file
    onnx.save_model(m, onnx_path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=data_name, size_threshold=1024)
    del m
    gc.collect()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logging.info(f"[Quant] ONNX saved: {onnx_path} ({os.path.getsize(onnx_path + '.data') / 2**30:.2f} GB)")


# --------------------------------------------------------------------------- build
def build_strongly_typed_engine(onnx_path: str, engine_path: str, input_profile: dict) -> None:
    """Strongly-typed TensorRT build (precision follows the ONNX graph). Marks the engine."""
    import tensorrt as trt
    from polygraphy.backend.trt import CreateConfig, Profile, engine_from_network, network_from_onnx_path, save_engine
    from .tensorrt.utilities import cudart

    from .tensorrt.utilities import onnx_input_names
    present = onnx_input_names(onnx_path)
    profile = Profile()
    for name, dims in input_profile.items():
        if present is not None and name not in present:
            continue  # spec input the export dropped (e.g. an unread StreamV2V cache port)
        profile.add(name, min=dims[0], opt=dims[1], max=dims[2])
    _, free_mem, _ = cudart.cudaMemGetInfo()
    # Keep ~1.5 GB for activations but never starve the builder: TensorRT needs a real
    # workspace (a 0-byte pool fails with "Compiling this graph requires workspace memory").
    workspace = max(free_mem - int(1.5 * 2**30), 1 * 2**30)
    t0 = time.time()
    logging.info(f"[Quant] TensorRT strongly-typed build: {os.path.basename(engine_path)} "
                 f"(free VRAM {free_mem / 2**30:.1f} GB, workspace {workspace / 2**30:.1f} GB, 3-6 min)")
    try:
        network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM], strongly_typed=True)
        engine = engine_from_network(
            network,
            config=CreateConfig(profiles=[profile], tactic_sources=[],
                                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: workspace}),
        )
    except Exception as e:
        # Strongly typed can reject mixed-dtype normalization params (seen with NVFP4 weight DQ);
        # the precision-flag build honours the Q/DQ ops in the graph just as well (measured).
        logging.warning(f"[Quant] strongly-typed build failed ({type(e).__name__}: {str(e)[:120]}); "
                        f"retrying with fp16/fp8 precision flags")
        network = network_from_onnx_path(onnx_path, flags=[trt.OnnxParserFlag.NATIVE_INSTANCENORM])
        engine = engine_from_network(
            network,
            config=CreateConfig(fp16=True, fp8=True, profiles=[profile], tactic_sources=[],
                                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: workspace}),
        )
    save_engine(engine, path=engine_path)
    mark_engine(engine_path)
    del engine
    gc.collect()
    logging.info(f"[Quant] Engine built in {time.time() - t0:.0f}s: {engine_path} "
                 f"({os.path.getsize(engine_path) / 2**30:.2f} GB)")


# --------------------------------------------------------------------------- calibration
_CALIB_PROMPTS = [
    "cinematic portrait of a woman with red hair, soft studio light, detailed skin",
    "a futuristic city at night, neon reflections on wet streets, ultra detailed",
    "misty mountain landscape at sunrise, pine forest, volumetric light",
    "macro photo of a dragonfly on a leaf, dew drops, bokeh",
    "oil painting of a stormy sea with a lighthouse, thick brush strokes",
    "a dancer mid-jump on a stage, dramatic spotlight, smoke",
    "abstract fluid art, iridescent colors, glossy, 3d render",
    "an old man reading in a library, warm lamp light, film grain",
    "a red sports car on a mountain road, motion blur",
    "underwater coral reef with tropical fish, sun rays",
    "cyberpunk street food market, crowd, steam, holograms",
    "snowy village at dusk, warm windows",
    "a robot playing violin in a concert hall, spotlight",
    "desert dunes with a lone camel, golden hour",
    "graffiti mural on a brick wall, vivid colors",
    "a cat wearing a spacesuit floating above earth",
]


def make_stream_calibration_loop(stream, n_images: int = 16, n_prompts: int = 2) -> Callable:
    """NVFP4 calibration on the production img2img path: frames generated by the pipeline
    itself (txt2img with varied prompts/seeds) are fed back through `stream(image)` at the
    stream's own timesteps. Quantizers are in collect-only mode here (normal fp16 speed)."""
    def forward_loop(_model):
        imgs = []
        # no_grad, NOT inference_mode: the pipeline lazily allocates state during these
        # passes (SimilarImageFilter.prev_tensor, latent buffers) that it later updates
        # in place at runtime; inference tensors would raise "Inplace update to inference
        # tensor outside InferenceMode" at the first real frame.
        with torch.no_grad():
            for i in range(n_images):
                stream.prepare(_CALIB_PROMPTS[i % len(_CALIB_PROMPTS)], "", num_inference_steps=50,
                               guidance_scale=1.0, delta=1.0, seed=1000 + i)
                img = stream.txt2img()
                img = img[0] if img.dim() == 4 else img
                # decode_image returns [-1, 1]; stream() expects the [0, 1] frame format.
                imgs.append((img.detach().to(torch.float16) / 2 + 0.5).clamp(0, 1))
            n = 0
            for pi in range(n_prompts):
                stream.prepare(_CALIB_PROMPTS[(pi * 5) % len(_CALIB_PROMPTS)], "", num_inference_steps=50,
                               guidance_scale=1.0, delta=1.0, seed=7 + pi)
                for img in imgs:
                    stream(img)
                    n += 1
        logging.info(f"[Quant] NVFP4 calibration: {n} img2img passes")
        # Do not carry calibration frames into the live stream (frame-skip filter state).
        sf = getattr(stream, "similar_filter", None)
        if sf is not None:
            sf.prev_tensor = None
            sf._prev_flat = None
            sf._pending_decision = None
            sf.skip_count = 0
    return forward_loop
