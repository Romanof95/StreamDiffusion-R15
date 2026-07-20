"""Wire protocol constants, enums, and CLI args for Smode <-> Python IPC."""
from __future__ import annotations

import enum
from typing import NamedTuple


MAGIC_NUMBER = 0xE280A0
ENDIAN_FORMAT: str = "<"

UINT32 = "I"
INT32 = "i"
UINT64 = "Q"
FLOAT32 = "f"


class CommandType(enum.Enum):
    OUTPUT = 1
    STOP = 2
    CONFIG = 3
    UUID = 5
    INPUT = 6
    STREAM_CREATION = 7
    WARNING = 8


class Mode(enum.IntEnum):
    IMAGE_TO_IMAGE = 1
    TEXT_TO_IMAGE = 2


class Acceleration(enum.IntEnum):
    NONE = 0
    TORCHCOMPILE = 1
    TENSORRT = 2
    


class ConfigType(enum.IntEnum):
    NONE = 1
    FULL = 2
    SELF = 3
    INITIALIZE = 4


def config_type_to_str(config_type: ConfigType) -> str:
    if config_type == ConfigType.NONE:
        return "none"
    elif config_type == ConfigType.FULL:
        return "full"
    elif config_type == ConfigType.SELF:
        return "self"
    elif config_type == ConfigType.INITIALIZE:
        return "initialize"
    else:
        raise ValueError(f"Unknown config type: {config_type}")


class Args(NamedTuple):
    port: int
    uuid: str
    width: int
    height: int
    device: int
    model: str
