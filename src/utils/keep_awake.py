"""
src.utils.keep_awake
======================
Keeps a long run going when nobody is at the keyboard (added 2026-09-30).
A full scan with the LLM reviewer takes ~40 min, but on battery this
laptop's power plan sleeps after 3 idle minutes, so a run only advanced
while someone used the machine. While a run lasts:

  - the PC is kept from sleeping (SetThreadExecutionState - the screen may
    still turn off and lock; normal sleep resumes when the run ends), and
  - Windows' background "efficiency" throttling is switched off for this
    process and for Ollama's processes, which serve the LLM reviewer
    (SetProcessInformation / ProcessPowerThrottling), so a run in a window
    that isn't in front doesn't crawl.

Windows only; elsewhere these are no-ops. Nothing here changes results.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from contextlib import contextmanager
from typing import Iterator

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
_PROCESS_POWER_THROTTLING = 4           # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
_THROTTLE_EXECUTION_SPEED = 0x1         # PROCESS_POWER_THROTTLING_EXECUTION_SPEED
_PROCESS_SET_INFORMATION = 0x0200


class _PowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]


def _kernel32():
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.OpenProcess.restype = ctypes.c_void_p
    k.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    k.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    k.SetThreadExecutionState.restype = ctypes.c_uint
    k.SetThreadExecutionState.argtypes = [ctypes.c_uint]
    return k


def disable_power_throttling(pid: int | None = None) -> bool:
    """Opts a process (default: this one) out of background execution-speed
    throttling. True if Windows accepted it."""
    if sys.platform != "win32":
        return False
    k = _kernel32()
    handle = k.GetCurrentProcess() if pid is None else k.OpenProcess(_PROCESS_SET_INFORMATION, 0, pid)
    if not handle:
        return False
    state = _PowerThrottlingState(1, _THROTTLE_EXECUTION_SPEED, 0)  # control the speed flag, state = off
    try:
        return bool(k.SetProcessInformation(handle, _PROCESS_POWER_THROTTLING, ctypes.byref(state), ctypes.sizeof(state)))
    finally:
        if pid is not None:
            k.CloseHandle(handle)


def _ollama_pids() -> list[int]:
    try:
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH", "/FI", "IMAGENAME eq ollama*"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    pids = []
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) > 1 and parts[1].isdigit():
            pids.append(int(parts[1]))
    return pids


@contextmanager
def keep_awake() -> Iterator[None]:
    """For the duration of a run - call it on the thread that does the work
    (Windows ties the no-sleep request to the calling thread)."""
    if sys.platform != "win32":
        yield
        return
    k = _kernel32()
    k.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    disable_power_throttling()
    for pid in _ollama_pids():
        disable_power_throttling(pid)
    try:
        yield
    finally:
        k.SetThreadExecutionState(ES_CONTINUOUS)
