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


def mark_engine(engine_path: str) -> None:
    """Record the TensorRT version next to a freshly built engine."""
    ver = _trt_version()
    if ver is None:
        return
    try:
        with open(engine_path + MARKER_SUFFIX, "w", encoding="utf-8") as f:
            f.write(ver)
    except OSError as e:
        logging.warning(f"[Engine cache] Could not write marker for {engine_path}: {e}")


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
    return True
