"""Win32 named event wrapper for Smode <-> Python per-frame signalling."""
from __future__ import annotations

import win32event
import win32api


class InterProcessEvent:
    def __init__(self):
        self.event = None
        self.signal_awakes_all_clients = True

    def __del__(self):
        if self.event:
            self.close()

    def create(self, name, signal_awakes_all_clients=True, initial_signaled_state=False) -> bool:
        if self.event is not None:
            raise RuntimeError("Event already assigned")

        CREATE_EVENT_MANUAL_RESET = 0x00000001
        CREATE_EVENT_INITIAL_SET = 0x00000002
        self.signal_awakes_all_clients = signal_awakes_all_clients

        if signal_awakes_all_clients:
            flags |= CREATE_EVENT_MANUAL_RESET

        if initial_signaled_state:
            flags |= CREATE_EVENT_INITIAL_SET

        self.event = win32event.CreateEvent(None, flags, win32event.EVENT_ALL_ACCESS, name)
        if win32api.GetLastError() not in (0, 183):
            err = win32api.GetLastError()
            raise RuntimeError(f"Failed to create event {name} code: {win32api.GetLastError()}")
        return True

    def open(self, name) -> bool:
        if self.event is not None:
            raise RuntimeError("Event already assigned")
        self.event = win32event.OpenEvent(win32event.SYNCHRONIZE, False, name)
        result = win32event.WaitForSingleObject(self.event, 10000)
        if result == win32event.WAIT_TIMEOUT:
            raise RuntimeError(f"Failed to open event {name} code: {win32api.GetLastError()}")
        if result == win32event.WAIT_FAILED:
            raise RuntimeError(f"Failed to open event {name} code: {win32api.GetLastError()}")
        if win32api.GetLastError() not in (0, 183): # array already exist signaling that the event is already open
            raise RuntimeError(f"Failed to open event {name} code: {win32api.GetLastError()}")
        return True

    def close(self) -> bool:
        if self.event is None:
            raise RuntimeError("Event not assigned")
        if win32api.CloseHandle(self.event) == 0:
            raise RuntimeError(f"Failed to close event {self.event} code: {win32api.GetLastError()}")
        self.event = None
        return True

    def wait(self, timeout=win32event.INFINITE) -> int:
        if self.event is None:
            raise RuntimeError("Event not assigned")
        return win32event.WaitForSingleObject(self.event, timeout)

    def signal(self) -> bool:
        if self.event is None:
            raise RuntimeError("Event not assigned")
        if win32event.SetEvent(self.event) == 0:
            raise RuntimeError(f"Failed to signal event {self.event} code: {win32api.GetLastError()}")
        if self.signal_awakes_all_clients:
            win32event.ResetEvent(self.event)
            return win32api.GetLastError() == 0
        return True
