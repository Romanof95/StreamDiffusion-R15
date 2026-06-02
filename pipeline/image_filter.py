from typing import Optional
import random

import torch


class SimilarImageFilter:
    """Cosine-similarity frame skipper used by SSF to reduce GPU activations."""

    def __init__(self, threshold: float = 0.98, max_skip_frame: float = 10) -> None:
        self.threshold = threshold
        self.prev_tensor = None
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        self.max_skip_frame = max_skip_frame
        self.skip_count = 0
        self._prev_flat = None  # zero-copy view of prev_tensor
        # Probe-and-cache: decide_skip() commits state + stashes the decision so
        # a subsequent __call__ on the same frame doesn't re-roll the RNG.
        self._pending_decision = None  # None | 'skip' | 'process'

    def decide_skip(self, x: torch.Tensor) -> bool:
        """Probe + commit the skip decision for the current frame."""
        if self.prev_tensor is None:
            self.prev_tensor = x.detach().clone()
            self._prev_flat = self.prev_tensor.view(-1)
            self._pending_decision = 'process'
            return False

        x_flat = x.view(-1)
        cos_sim = self.cos(self._prev_flat, x_flat).item()
        sample = random.uniform(0, 1)
        if self.threshold >= 1:
            skip_prob = 0
        else:
            skip_prob = max(0, 1 - (1 - cos_sim) / (1 - self.threshold))

        if skip_prob < sample:
            old_tensor = self.prev_tensor
            self.prev_tensor = x.detach().clone()
            self._prev_flat = self.prev_tensor.view(-1)
            del old_tensor
            self._pending_decision = 'process'
            return False

        if self.skip_count > self.max_skip_frame:
            self.skip_count = 0
            old_tensor = self.prev_tensor
            self.prev_tensor = x.detach().clone()
            self._prev_flat = self.prev_tensor.view(-1)
            del old_tensor
            self._pending_decision = 'process'
            return False

        self.skip_count += 1
        self._pending_decision = 'skip'
        return True

    def __call__(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        # Consume a pending decision stashed by decide_skip() to keep the two
        # probes aligned.
        if self._pending_decision is not None:
            decision = self._pending_decision
            self._pending_decision = None
            return None if decision == 'skip' else x

        if self.prev_tensor is None:
            self.prev_tensor = x.detach().clone()
            self._prev_flat = self.prev_tensor.view(-1)
            return x

        x_flat = x.view(-1)
        cos_sim = self.cos(self._prev_flat, x_flat).item()
        sample = random.uniform(0, 1)
        if self.threshold >= 1:
            skip_prob = 0
        else:
            skip_prob = max(0, 1 - (1 - cos_sim) / (1 - self.threshold))

        if skip_prob < sample:
            old_tensor = self.prev_tensor
            self.prev_tensor = x.detach().clone()
            self._prev_flat = self.prev_tensor.view(-1)
            del old_tensor
            return x

        if self.skip_count > self.max_skip_frame:
            self.skip_count = 0
            old_tensor = self.prev_tensor
            self.prev_tensor = x.detach().clone()
            self._prev_flat = self.prev_tensor.view(-1)
            del old_tensor
            return x

        self.skip_count += 1
        return None

    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold

    def set_max_skip_frame(self, max_skip_frame: float) -> None:
        self.max_skip_frame = max_skip_frame
