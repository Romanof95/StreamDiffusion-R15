"""Configuration manager: typed config with thread-safe hot-reload."""
import json
import logging
import threading
import time
import torch
from pathlib import Path
from typing import Optional, Tuple

from .schema import StreamDiffusionConfig


class ConfigManager:
    """Thread-safe configuration manager with hot-reload support."""

    def __init__(self, config_path: str, device: torch.device, dtype: torch.dtype):
        self._path = Path(config_path)
        self._device = device
        self._dtype = dtype
        self._lock = threading.Lock()
        self._config: Optional[StreamDiffusionConfig] = None
        self._raw_dict: Optional[dict] = None
        self._last_check_time: float = 0.0
        self._last_modified: float = 0.0
        self._gaussian_kernel_cache: Optional[torch.Tensor] = None
        self._gaussian_kernel_size: int = 0
        self.check_interval: float = 2.0

        self._config = self._load()
        self._update_gaussian_kernel()

    def _load(self) -> StreamDiffusionConfig:
        """Load config from JSON file. Returns defaults if file is missing or invalid."""
        try:
            if self._path.exists():
                with open(self._path, "r", encoding="utf-8") as f:
                    self._raw_dict = json.load(f)
                return StreamDiffusionConfig.from_dict(self._raw_dict)
            else:
                logging.warning(f"Config file not found: {self._path}, using defaults")
                self._raw_dict = {}
                return StreamDiffusionConfig()
        except Exception as e:
            logging.warning(f"Failed to load config: {e}")
            if self._config is not None:
                return self._config
            self._raw_dict = {}
            return StreamDiffusionConfig()

    def get(self) -> StreamDiffusionConfig:
        """Get current typed config (thread-safe, no I/O)."""
        return self._config

    def get_raw_dict(self) -> Optional[dict]:
        """Get the raw dictionary from the last JSON load."""
        return self._raw_dict

    def reload(self) -> Tuple[bool, StreamDiffusionConfig]:
        """Force reload config from file. Returns (changed, config)."""
        with self._lock:
            old_config = self._config
            self._config = self._load()
            self._last_check_time = time.time()

            changed = self._has_changed(old_config, self._config)
            if changed:
                self._update_gaussian_kernel()
                logging.info("Config reloaded (changes detected)")

            return changed, self._config

    def reload_if_due(self) -> Tuple[bool, StreamDiffusionConfig]:
        """Reload config if check_interval has elapsed since last check."""
        now = time.time()
        if now - self._last_check_time < self.check_interval:
            return False, self._config

        return self.reload()

    def _has_changed(self, old: StreamDiffusionConfig, new: StreamDiffusionConfig) -> bool:
        if old is None:
            return True
        try:
            old_dict = old.to_dict()
            new_dict = new.to_dict()
            return old_dict != new_dict
        except Exception:
            return True

    def _update_gaussian_kernel(self):
        """Pre-compute Gaussian blur kernel for Depth processing."""
        config = self._config
        blur_kernel = config.depth.blur_kernel

        if config.depth.enabled and blur_kernel > 1:
            if blur_kernel != self._gaussian_kernel_size:
                self._gaussian_kernel_cache = self._compute_gaussian_kernel(blur_kernel)
                self._gaussian_kernel_size = blur_kernel
        else:
            self._gaussian_kernel_cache = None
            self._gaussian_kernel_size = 0

    def _compute_gaussian_kernel(self, blur_kernel: int) -> torch.Tensor:
        """Compute 2D Gaussian blur kernel on GPU."""
        kernel_size = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

        kernel_1d = torch.exp(
            -torch.arange(
                -(kernel_size // 2),
                kernel_size // 2 + 1,
                dtype=torch.float32,
                device=self._device,
            )
            ** 2
            / (2 * sigma**2)
        )
        kernel_1d = kernel_1d / kernel_1d.sum()

        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)

        return kernel_2d

    @property
    def gaussian_kernel(self) -> Optional[torch.Tensor]:
        """Pre-computed Gaussian kernel for Depth processing, or None if not needed."""
        return self._gaussian_kernel_cache
