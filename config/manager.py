"""Configuration manager: typed config with thread-safe hot-reload."""
import json
import logging
import threading
import time
import torch
from pathlib import Path
from typing import Optional, Tuple

from .schema import StreamDiffusionConfig


import threading
from copy import deepcopy
from typing import Optional

import torch

from .schema import ControlNetConfig

class ConfigManager:
    """Thread-safe ControlNet configuration manager."""

    def __init__(self, config: ControlNetConfig, device: torch.device, dtype: torch.dtype,
    ):
        self._config = config
        self._device = device
        self._dtype = dtype
        self._lock = threading.Lock()
        self._gaussian_kernel_cache: Optional[torch.Tensor] = None
        self._gaussian_kernel_size: int = 0

        self._update_gaussian_kernel()

    def get(self) -> ControlNetConfig:
        """Get the current configuration."""
        with self._lock:
            return self._config

    def update(self, config: ControlNetConfig) -> bool:
        """Update the configuration True if the configuration changed."""
        with self._lock:
            if config == self._config:
                return False

            self._config = config
            self._update_gaussian_kernel()
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
