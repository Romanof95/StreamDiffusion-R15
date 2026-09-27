"""Static-region stabilizer: keeps the previous output where the input image did not move.

The output is an exponential average whose per-pixel weight comes from a motion mask of the
input: where the input is still, the output moves toward the new frame by ``weight`` only
(0.2 = 80 % of the previous output kept), so the regenerated background stops flickering;
where the input moves, the new frame replaces the output at once. The mask compares the
input with the input average integrated by the same weights, so slow motion accumulates and
eventually updates the output instead of leaving a trail.

STREAMDIFFUSION_STATIC_FREEZE = weight in still regions (0 or unset = off, e.g. 0.2),
STREAMDIFFUSION_STATIC_FREEZE_THRESH = "low,high" input luma difference (0-1 image scale)
between fully still and fully moving (default "0.02,0.06").
"""
import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def static_freeze_config() -> Tuple[float, float, float]:
    """(weight, low, high); weight 0 = off."""
    try:
        weight = float(os.environ.get("STREAMDIFFUSION_STATIC_FREEZE", "0") or 0)
    except ValueError:
        weight = 0.0
    if weight <= 0.0 or weight >= 1.0:
        return 0.0, 0.0, 0.0
    low, high = 0.02, 0.06
    raw = os.environ.get("STREAMDIFFUSION_STATIC_FREEZE_THRESH", "")
    if raw:
        try:
            low, high = (float(v) for v in raw.split(","))
        except ValueError:
            logging.warning(f"[StaticFreeze] Bad STREAMDIFFUSION_STATIC_FREEZE_THRESH '{raw}', using 0.02,0.06")
    if high <= low:
        high = low + 0.01
    return weight, low, high


class StaticRegionStabilizer:
    # Mean motion above which the frame is treated as a cut and the average restarts.
    CUT_FRACTION = 0.35

    def __init__(self, weight: float, low: float, high: float) -> None:
        self.weight = weight
        # Thresholds are given on the 0-1 image scale, the tensors are in [-1, 1].
        self.low = 2.0 * low
        self.high = 2.0 * high
        self._in_avg: Optional[torch.Tensor] = None
        self._out_avg: Optional[torch.Tensor] = None
        self.last_mask: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._in_avg = None
        self._out_avg = None
        self.last_mask = None

    @staticmethod
    def _luma(x: torch.Tensor) -> torch.Tensor:
        return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).float()

    def __call__(self, inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """``inp``: the input frame this output was generated from, ``out``: the decoded
        output, both (B, 3, H, W) in [-1, 1]. Returns the stabilized output."""
        if inp.shape[-2:] != out.shape[-2:]:
            inp = F.interpolate(inp.float(), size=out.shape[-2:], mode="bilinear", align_corners=False)
        if (self._out_avg is None or self._out_avg.shape != out.shape
                or self._in_avg.shape[-2:] != inp.shape[-2:]):
            self._out_avg = out.float().clone()
            self._in_avg = inp.float().clone()
            return out
        h, w = out.shape[-2:]
        # The mask is computed at ~256 px (the averaging removes sensor noise), then upscaled.
        s = max(1, min(h, w) // 256)
        k = 5
        diff = (self._luma(inp) - self._luma(self._in_avg)).abs()
        if s > 1:
            diff = F.avg_pool2d(diff, s)
        diff = F.avg_pool2d(diff, k, stride=1, padding=k // 2)                 # sensor noise
        motion = ((diff - self.low) / (self.high - self.low)).clamp_(0.0, 1.0)
        if float(motion.mean()) > self.CUT_FRACTION:
            # Scene cut / camera move: regions whose luma happens to match would keep the
            # old picture (ghosts), start again from the new frame.
            self._out_avg.copy_(out.float())
            self._in_avg.copy_(inp.float())
            self.last_mask = torch.ones_like(motion)
            return out
        motion = F.max_pool2d(motion, 2 * k + 1, stride=1, padding=k)         # grow around motion
        motion = F.avg_pool2d(motion, 2 * k + 1, stride=1, padding=k)         # feathered edge
        if s > 1:
            motion = F.interpolate(motion, size=(h, w), mode="bilinear", align_corners=False)
        wgt = self.weight + (1.0 - self.weight) * motion
        self._out_avg.add_(wgt * (out.float() - self._out_avg))
        self._in_avg.add_(wgt * (inp.float() - self._in_avg))
        self.last_mask = motion
        return self._out_avg.to(out.dtype)
