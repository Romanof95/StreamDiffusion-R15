"""Shared CUDA texture between Smode (OpenGL) and Python (PyTorch)."""
from __future__ import annotations

import torch
from torch.multiprocessing import reductions


class StreamDiffusionSmodeTexture:
    """Pair of GPU tensors for zero-copy frame exchange with Smode.

    ``smode_tensor`` is float32 HWC and exposed via CUDA IPC.
    ``stream_diffusion_tensor`` is engine-dtype HWC, internal working copy.
    """

    def __init__(
        self, device: int, width: int, height: int, channels: int, dtype: torch.dtype
    ):
        self.device = device
        self.width = width
        self.height = height
        self.channels = channels
        self.dtype = dtype

        self.stream_diffusion_tensor = torch.empty(
            (self.height, self.width, channels), dtype=self.dtype, device=self.device
        )
        self.smode_tensor = torch.empty(
            (self.height, self.width, channels), dtype=torch.float32, device=self.device
        )
        self.smode_tensor_ipc_info = reductions.reduce_tensor(
            self.smode_tensor
        )[1]

        # Pre-allocated CHW buffer for the fused permute+vflip.
        # Smode delivers top-left origin (OpenGL); engines expect bottom-left.
        self._chw_buffer = torch.empty(
            (channels, self.height, self.width), dtype=self.dtype, device=self.device
        )
        self._vflip_row_idx = torch.arange(
            self.height - 1, -1, -1, device=self.device, dtype=torch.long
        )

    def copy_smode_to_stream_diffusion(self):
        """Copy smode_tensor into stream_diffusion_tensor with implicit dtype cast."""
        self.stream_diffusion_tensor.copy_(self.smode_tensor)

    def get_permuted_input_tensor(self) -> torch.Tensor:
        """Return CHW + vertically flipped input in a single GPU copy."""
        permuted = self.stream_diffusion_tensor.permute(2, 0, 1)
        torch.index_select(permuted, 1, self._vflip_row_idx, out=self._chw_buffer)
        return self._chw_buffer

    def write_chw_to_smode(self, chw_tensor: torch.Tensor) -> None:
        """Write a CHW engine-output tensor directly into the Smode IPC buffer."""
        self.smode_tensor.copy_(chw_tensor.permute(1, 2, 0))
