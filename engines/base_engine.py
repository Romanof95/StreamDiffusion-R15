"""Abstract base class for generation engines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseEngine(ABC):
    """Contract every generation backend must honor.

    Lifecycle: __init__ -> load(config) -> run(image, prompt) -> cleanup().
    update_prompt() / update_params() hot-swap without full reload.
    """

    def __init__(self) -> None:
        self.wrapper: Any = None
        self._loaded: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def load(self, config: Dict[str, Any], **runtime: Any) -> None:
        ...

    @abstractmethod
    def run(
        self,
        image: Optional[torch.Tensor] = None,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def update_prompt(self, prompt: str) -> None:
        ...

    @abstractmethod
    def update_params(self, config: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def cleanup(self) -> None:
        ...

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        return f"<{type(self).__name__} loaded={self._loaded}>"
