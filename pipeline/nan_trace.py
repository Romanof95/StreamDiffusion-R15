"""Per-stage NaN/Inf tracer for the generation pipeline. Diagnostic, off by default.

STREAMDIFFUSION_NAN_TRACE=1 turns it on. Each stage of a frame (input, depth prediction,
control map, encoded latent, ControlNet residuals / UNet output / x0 of every denoising
step, StreamV2V ring, decoded output) records one scalar: max |x|. NaN and Inf propagate
through that max, so the scalar tells both whether the stage is clean and how close its
values run to the fp16 ceiling (65504). The scalars are read back once per frame.

When a frame carries a non-finite value, one log line names the FIRST stage where it
appeared, followed by every stage's magnitude in order. The input frame and control map
of the first few such frames are saved under nan_dumps/ (exact tensors in a .pt, plus
PNGs to look at) so the frame can be replayed offline. Every 10 s a summary line gives
the peak magnitude of each stage over the window.
"""
import logging
import math
import os
import time
from pathlib import Path

import torch

ENABLED = os.environ.get("STREAMDIFFUSION_NAN_TRACE", "0") == "1"
MAX_DUMPS = int(os.environ.get("STREAMDIFFUSION_NAN_TRACE_DUMPS", "8"))
SUMMARY_EVERY_S = 10.0
DUMP_DIR = Path(__file__).resolve().parent.parent / "nan_dumps"

_names: list = []
_vals: list = []
_keep: dict = {}
_peaks: dict = {}
_frames = 0
_bad_frames = 0
_window_frames = 0
_window_bad = 0
_dumps = 0
_last_summary = time.time()

_announced = False
# Preview modes run the preprocessors without a generation, so no frame ever ends: cap what
# accumulates instead of growing forever.
_MAX_PENDING = 512


def _max_abs(parts: list) -> torch.Tensor:
    """max |x| over tensors, NaN/Inf propagating. One grouped kernel launch for many
    tensors (the StreamV2V ring alone is ~200 buffers): ~5x cheaper than abs().amax() each,
    which also copies every buffer."""
    try:
        norms = torch._foreach_norm(parts, float("inf"))
    except Exception:  # private API: fall back if it ever goes away
        norms = [torch.linalg.vector_norm(p, float("inf")) for p in parts]
    if len(norms) == 1:
        return norms[0].float()
    return torch.stack([n.float() for n in norms]).amax()


def _flatten(t) -> list:
    if isinstance(t, (list, tuple)):
        return [x for item in t for x in _flatten(item)]
    return [] if t is None else [t]


def record(name: str, t) -> None:
    """Queue max |t| for this frame (nested lists of tensors count as one stage)."""
    if not ENABLED or t is None:
        return
    parts = _flatten(t)
    if not parts:
        return
    if len(_vals) >= _MAX_PENDING:
        _names.clear()
        _vals.clear()
    _names.append(name)
    _vals.append(_max_abs([p.detach() for p in parts]))


def record_v2v_rings(name: str, unet) -> None:
    """Max |x| over every StreamV2V cache buffer of a TensorRT UNet (or pair of engines)."""
    if not ENABLED or unet is None:
        return
    engines = [unet] if "_rings" in vars(unet) else [
        e for e in vars(unet).values() if "_rings" in getattr(e, "__dict__", {})]
    bufs = []
    for e in engines:
        for ring in e._rings.values():
            for slot in ring:
                bufs.extend(slot)
        for cache in e._kvo_caches.values():
            bufs.extend(cache)
    if bufs:
        record(name, bufs)


def keep(name: str, t) -> None:
    """Hold a copy of a tensor to dump if this frame turns out non-finite."""
    if ENABLED and t is not None:
        _keep[name] = t.detach().clone()


def _to_png(t: torch.Tensor, path: Path, lo: float, hi: float) -> None:
    from PIL import Image
    x = t.detach().float()
    while x.dim() > 3:
        x = x[0]
    if x.dim() == 2:
        x = x.unsqueeze(0)
    x = torch.nan_to_num(x, nan=0.0, posinf=hi, neginf=lo)
    x = ((x - lo) / (hi - lo)).clamp(0, 1)
    if x.shape[0] == 1:
        x = x.expand(3, -1, -1)
    arr = (x[:3] * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)


def _dump(tag: str, vals: list) -> None:
    global _dumps
    DUMP_DIR.mkdir(exist_ok=True)
    base = DUMP_DIR / tag
    torch.save({"stages": list(zip(_names, vals)),
                "tensors": {k: v.cpu() for k, v in _keep.items()}}, f"{base}.pt")
    if "input" in _keep:          # the input texture is in [0, 1]
        _to_png(_keep["input"], Path(f"{base}_input.png"), 0.0, 1.0)
    if "control_map" in _keep:    # control maps are in [0, 1]
        _to_png(_keep["control_map"], Path(f"{base}_control.png"), 0.0, 1.0)
    _dumps += 1


def end_frame() -> None:
    """Read this frame's stages back (one sync), log a non-finite frame, clear."""
    global _frames, _bad_frames, _window_frames, _window_bad, _last_summary, _announced
    if not ENABLED:
        return
    if not _announced:  # logging is not configured yet when this module is imported
        _announced = True
        logging.info(f"[NaN trace] on: first non-finite stage per frame is logged, "
                     f"up to {MAX_DUMPS} frames dumped to {DUMP_DIR}")
    try:
        if not _vals:
            return
        vals = torch.stack(_vals).tolist()
        _frames += 1
        _window_frames += 1
        for n, v in zip(_names, vals):
            if math.isfinite(v):
                _peaks[n] = max(_peaks.get(n, 0.0), v)
        first = next((i for i, v in enumerate(vals) if not math.isfinite(v)), None)
        if first is not None:
            _bad_frames += 1
            _window_bad += 1
            detail = " | ".join(f"{n}={v:.4g}" for n, v in zip(_names, vals))
            logging.warning(f"[NaN trace] frame {_frames}: first non-finite stage = "
                            f"{_names[first]} ({vals[first]}) :: {detail}")
            if _dumps < MAX_DUMPS:
                tag = time.strftime("%Y%m%d_%H%M%S") + f"_f{_frames}"
                try:
                    _dump(tag, vals)
                    logging.warning(f"[NaN trace] dumped {DUMP_DIR / tag}.pt")
                except Exception as e:  # a failed dump must never stop the stream
                    logging.warning(f"[NaN trace] dump failed: {e}")
        now = time.time()
        if now - _last_summary >= SUMMARY_EVERY_S:
            peaks = " | ".join(f"{n}={v:.4g}" for n, v in _peaks.items())
            logging.info(f"[NaN trace] last {now - _last_summary:.0f}s: {_window_bad}/"
                         f"{_window_frames} non-finite frames (total {_bad_frames}/{_frames}); "
                         f"peak |x| per stage (fp16 max 65504): {peaks}")
            _peaks.clear()
            _window_frames = _window_bad = 0
            _last_summary = now
    finally:
        _names.clear()
        _vals.clear()
        _keep.clear()
