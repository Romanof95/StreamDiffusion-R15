"""Low-latency mode controller for the realtime inference loop.

Two levers (both safe, both toggleable at runtime):
  1. **Controlled GC**: ``gc.disable()`` + periodic manual ``gc.collect()`` every
     N frames. Avoids random 20-50 ms GC pauses during generation. Cleanup
     still happens (every N frames) so there is no memory leak on long runs.
  2. **HIGH process priority**: bumps the Python process to Windows
     HIGH_PRIORITY_CLASS so the OS scheduler does not preempt it for
     low-priority apps (browser, Discord, etc.).

Both are idempotent: calling ``LowLatencyController.apply(enabled=True)``
twice is a no-op; toggling to ``False`` restores the previous state.
"""
from __future__ import annotations

import gc
import logging
import platform
from typing import Optional

try:
    import psutil  # installed as a dep via insightface / other tooling
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False


class LowLatencyController:
    """Applies / reverts low-latency OS and runtime tweaks safely."""

    # Force a manual gc.collect() roughly every 500 frames (~10 s at 50 FPS).
    # Keeps memory bounded without ever running GC at a random moment.
    DEFAULT_GC_INTERVAL_FRAMES = 500

    def __init__(self) -> None:
        self._applied: bool = False
        self._gc_was_enabled: bool = True
        self._prev_priority: Optional[int] = None
        self._frame_counter: int = 0
        self._gc_interval: int = self.DEFAULT_GC_INTERVAL_FRAMES

    # ------------------------------------------------------------------ apply
    def apply(self, enabled: bool) -> None:
        """Enable or disable low-latency mode. Idempotent and safe to toggle."""
        if enabled and not self._applied:
            self._enable()
        elif not enabled and self._applied:
            self._disable()

    def _enable(self) -> None:
        # Remember the previous state so we can restore it on disable.
        self._gc_was_enabled = gc.isenabled()
        gc.collect()  # clean slate before taking control
        gc.disable()

        if _PSUTIL_OK:
            try:
                p = psutil.Process()
                self._prev_priority = p.nice()
                if platform.system() == "Windows":
                    p.nice(psutil.HIGH_PRIORITY_CLASS)
                else:
                    # Unix: lower value = higher priority; -5 is a mild bump
                    p.nice(-5)
                logging.info(
                    "[low-latency] Process priority: %s -> %s",
                    self._prev_priority, p.nice()
                )
            except Exception as e:
                logging.warning("[low-latency] Could not raise process priority: %s", e)
                self._prev_priority = None

        self._applied = True
        self._frame_counter = 0
        logging.info(
            "[low-latency] Enabled (gc disabled, manual collect every %d frames)",
            self._gc_interval,
        )

    def _disable(self) -> None:
        if self._gc_was_enabled:
            gc.enable()
        gc.collect()  # one big collect to clean up the accumulated garbage

        if _PSUTIL_OK and self._prev_priority is not None:
            try:
                psutil.Process().nice(self._prev_priority)
            except Exception as e:
                logging.warning("[low-latency] Could not restore priority: %s", e)

        self._applied = False
        logging.info("[low-latency] Disabled (gc re-enabled, priority restored)")

    # --------------------------------------------------------------- per-tick
    def tick(self) -> None:
        """Call once per processed frame.

        When low-latency is on, triggers a manual ``gc.collect()`` every
        ``_gc_interval`` frames. When off, does nothing (Python's GC runs on
        its own schedule).
        """
        if not self._applied:
            return
        self._frame_counter += 1
        if self._frame_counter >= self._gc_interval:
            gc.collect()
            self._frame_counter = 0

    # ------------------------------------------------------------------ misc
    @property
    def is_applied(self) -> bool:
        return self._applied

    @property
    def gc_interval(self) -> int:
        return self._gc_interval

    @gc_interval.setter
    def gc_interval(self, value: int) -> None:
        self._gc_interval = max(1, int(value))
