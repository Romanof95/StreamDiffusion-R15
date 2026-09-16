"""TensorRT engine cache validity.

TensorRT engines are only loadable by the exact TensorRT version that built them.
Every engine written by this package gets a side-car ``<engine>.trtver`` holding
that version; ``engine_ready`` treats a missing or mismatching marker as a cache
miss so a TensorRT upgrade triggers a clean rebuild instead of a deserialization
failure at load time.

No TensorRT import at module level: this module is imported by code paths that
must work when TensorRT is not installed (torch.compile / none acceleration).
"""
import logging
import os

MARKER_SUFFIX = ".trtver"


def _trt_version():
    try:
        import tensorrt as trt
        return str(trt.__version__)
    except Exception:
        return None


def discard_build_intermediates(engine_path: str) -> None:
    """Delete the ONNX export artifacts next to a freshly built engine: ``<engine>.onnx``,
    ``<engine>.opt.onnx``, their ``.data`` external-weight files, and the per-tensor weight
    shards torch.onnx.export spills into the directory for >2 GB models. They are only read
    while building; left behind they weigh more than the engines themselves (~60 GB seen).
    A later rebuild (TensorRT upgrade) simply re-exports."""
    d = os.path.dirname(engine_path) or "."
    base = os.path.basename(engine_path)
    freed = 0
    try:
        names = os.listdir(d)
    except OSError:
        return
    targets = []
    for n in names:
        p = os.path.join(d, n)
        is_export = n.startswith(base + ".onnx") or n.startswith(base + ".opt.onnx")
        is_shard = (n.startswith("onnx__") or n.endswith(".weight") or n.endswith(".bias")) and ".engine" not in n
        if (is_export or is_shard) and os.path.isfile(p):
            targets.append(p)
    for attempt in range(2):
        left = []
        for p in targets:
            try:
                size = os.path.getsize(p)
                os.remove(p)
                freed += size
            except OSError:
                left.append(p)  # typically still mapped by the ONNX parser right after a build
        targets = left
        if not targets:
            break
        import gc
        gc.collect()
    if targets:
        # Picked up at the next cache hit (engine_ready calls this again).
        logging.info(f"[Engine cache] {base}: {len(targets)} ONNX intermediate(s) still in use, deferred")
    tmp = engine_path + ".onnx.export_tmp"
    if os.path.isdir(tmp):
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    if freed:
        logging.info(f"[Engine cache] {base}: ONNX build intermediates removed ({freed / 2**30:.2f} GB)")


def mark_engine(engine_path: str) -> None:
    """Record the TensorRT version next to a freshly built engine and drop its ONNX
    intermediates (see ``discard_build_intermediates``)."""
    ver = _trt_version()
    if ver is not None:
        try:
            with open(engine_path + MARKER_SUFFIX, "w", encoding="utf-8") as f:
                f.write(ver)
        except OSError as e:
            logging.warning(f"[Engine cache] Could not write marker for {engine_path}: {e}")
    discard_build_intermediates(engine_path)


def engine_ready(engine_path) -> bool:
    """True when the engine file exists and was built by the running TensorRT."""
    if engine_path is None:
        return False
    engine_path = str(engine_path)
    if not os.path.exists(engine_path):
        return False
    ver = _trt_version()
    if ver is None:
        return True
    marker = engine_path + MARKER_SUFFIX
    try:
        with open(marker, "r", encoding="utf-8") as f:
            built_with = f.read().strip()
    except OSError:
        logging.info(
            f"[Engine cache] {os.path.basename(engine_path)}: no TensorRT version marker "
            f"(built before the marker existed) - rebuilding with TensorRT {ver}"
        )
        return False
    if built_with != ver:
        logging.info(
            f"[Engine cache] {os.path.basename(engine_path)}: built with TensorRT {built_with}, "
            f"running {ver} - rebuilding"
        )
        return False
    discard_build_intermediates(engine_path)
    return True
