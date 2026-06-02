"""Base class for preprocessors (Canny, Depth, OpenPose, FaceID)."""
import logging
from abc import ABC, abstractmethod
from typing import Optional

import torch


class BasePreprocessor(ABC):
    """Abstract base class for ControlNet / IP-Adapter preprocessors.

    Subclasses implement ``load_model``, ``unload_model``, ``process``, and
    the ``name`` property. The base class tracks loaded state, caches the
    last result for frame-skip reuse, and handles cleanup.
    """

    def __init__(self, device: torch.device, torch_dtype: torch.dtype, max_buffer_size: int = 1024):
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_buffer_size = max_buffer_size
        self._loaded = False
        self._cached_result: Optional[torch.Tensor] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name (e.g. 'canny', 'depth')."""
        ...

    @abstractmethod
    def load_model(self, config) -> None:
        """Load any ML models required by this processor."""
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Unload ML models and free GPU memory."""
        ...

    @abstractmethod
    def process(self, image_tensor: torch.Tensor, config) -> Optional[torch.Tensor]:
        """Run preprocessing. Input: CHW [0,1] on GPU. Output: CHW [0,1] on GPU, or None."""
        ...

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_cached_result(self) -> Optional[torch.Tensor]:
        return self._cached_result

    def set_cached_result(self, result: torch.Tensor) -> None:
        self._cached_result = result

    def cleanup(self) -> None:
        if self._loaded:
            try:
                self.unload_model()
            except Exception as e:
                logging.warning(f"[{self.name}] Error during cleanup: {e}")
        self._cached_result = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
