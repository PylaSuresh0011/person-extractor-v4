"""Tests for src.utils.keep_awake (no-sleep and background-throttling opt-out)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.utils import keep_awake as ka


@pytest.mark.skipif(sys.platform != "win32", reason="Windows power APIs")
def test_no_sleep_request_is_held_during_a_run_and_released_after():
    k = ka._kernel32()
    with ka.keep_awake():
        held = k.SetThreadExecutionState(ka.ES_CONTINUOUS | ka.ES_SYSTEM_REQUIRED)
    released = k.SetThreadExecutionState(ka.ES_CONTINUOUS)
    assert held & ka.ES_SYSTEM_REQUIRED
    assert not released & ka.ES_SYSTEM_REQUIRED


@pytest.mark.skipif(sys.platform != "win32", reason="Windows power APIs")
def test_this_process_opts_out_of_background_throttling():
    assert ka.disable_power_throttling() is True


def test_elsewhere_it_is_a_no_op(monkeypatch):
    monkeypatch.setattr(ka.sys, "platform", "linux")
    assert ka.disable_power_throttling() is False
    with ka.keep_awake():
        pass
