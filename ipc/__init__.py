"""IPC layer between Smode (host) and the StreamDiffusion Python process.

Three communication channels:
- Win32 named events (``events``) - per-frame ready signalling
- TCP socket (``socket_helpers``) - control messages
- CUDA IPC (``texture``) - zero-copy GPU tensor sharing
"""

from .protocol import (
    MAGIC_NUMBER,
    ENDIAN_FORMAT,
    UINT32,
    UINT64,
    FLOAT32,
    CommandType,
    Mode,
    Acceleration,
    ConfigType,
    config_type_to_str,
    Args,
)
from .events import InterProcessEvent
from .packets import (
    Packet,
    FrameDataPacket,
    ConfigPacket,
    UuidPacket,
    StreamCreationPacket,
    WarningPacket,
    _parse_config_with_cache,
)
from .texture import StreamDiffusionSmodeTexture
from .socket_helpers import (
    recv_all,
    recv_message,
    send_message,
    read_string,
    is_socket_connected,
)

__all__ = [
    "MAGIC_NUMBER", "ENDIAN_FORMAT", "UINT32", "UINT64", "FLOAT32",
    "CommandType", "Mode", "Acceleration", "ConfigType",
    "config_type_to_str", "Args",
    "InterProcessEvent",
    "Packet", "FrameDataPacket", "ConfigPacket",
    "UuidPacket", "StreamCreationPacket", "WarningPacket",
    "_parse_config_with_cache",
    "StreamDiffusionSmodeTexture",
    "recv_all", "recv_message", "send_message",
    "read_string", "is_socket_connected",
]
