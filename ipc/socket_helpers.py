"""Socket helpers for the Smode <-> Python wire protocol."""
from __future__ import annotations

import logging
import socket
import struct

from .protocol import (
    ENDIAN_FORMAT,
    MAGIC_NUMBER,
    CommandType,
)


def recv_all(sock: socket.socket, n: int) -> bytes:
    """Receive exactly n bytes from the socket."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        data += chunk
    return data


def recv_message(sock: socket.socket):
    """Receive one packet. Returns (CommandType, payload_bytes)."""
    header = recv_all(sock, 8)
    magic, size = struct.unpack(ENDIAN_FORMAT + "II", header)
    if magic != MAGIC_NUMBER:
        logging.error(
            f"Invalid magic number received: {hex(magic)} (expected {hex(MAGIC_NUMBER)})"
        )
        return None, None
    payload = recv_all(sock, size)
    if len(payload) < 4:
        logging.error("Payload too short to contain command code")
        return None, None
    cmd_int, = struct.unpack(ENDIAN_FORMAT + "I", payload[:4])
    try:
        cmd = CommandType(cmd_int)
    except ValueError:
        logging.error(f"Unknown command code received: {cmd_int}")
        return None, None
    return cmd, payload[4:]


def send_message(sock: socket.socket, packet):
    """Serialise and send a Packet over the socket."""
    sock.sendall(packet.to_bytes())


def read_string(data: bytes, offset: int):
    """Read a length-prefixed string. Returns (string, new_offset)."""
    if offset + 4 > len(data):
        raise ValueError("Insufficient data for string length")
    str_len, = struct.unpack_from(ENDIAN_FORMAT + "I", data, offset)
    offset += 4
    if offset + str_len > len(data):
        raise ValueError("Insufficient data for string content")
    s = data[offset: offset + str_len].decode("utf-8")
    offset += str_len
    return s, offset


def is_socket_connected(sock):
    """Non-blocking peek to check if the socket is still connected."""
    try:
        data = sock.recv(1, socket.MSG_PEEK)
        return len(data) > 0
    except BlockingIOError:
        return True
    except socket.error:
        return False
