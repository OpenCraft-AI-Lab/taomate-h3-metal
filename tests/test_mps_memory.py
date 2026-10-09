# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Allocator telemetry mocks run without GPU; the tiny device case is explicit."""

from types import SimpleNamespace

import pytest
import torch

from taomate_h3.runtime.mps_memory import mps_memory_receipt, reset_mps_peak_memory


def test_allocator_mapping_and_explicit_device(monkeypatch):
    devices = []

    def stats(device):
        devices.append(device)
        return {
            "allocated_bytes.all.current": 11,
            "reserved_bytes.all.current": 37,
            "allocated_bytes.all.peak": 23,
            "reserved_bytes.all.peak": 41,
        }

    monkeypatch.setattr(
        torch,
        "accelerator",
        SimpleNamespace(
            memory_stats=stats, reset_peak_memory_stats=lambda device: devices.append(device)
        ),
    )
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 53)
    monkeypatch.setattr(
        torch.mps, "current_allocated_memory", lambda: pytest.fail("unneeded fallback")
    )
    assert reset_mps_peak_memory()
    result = mps_memory_receipt()
    assert [
        result[k]
        for k in (
            "current_allocated_bytes",
            "current_reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "driver_allocated_bytes",
        )
    ] == [11, 37, 23, 41, 53]
    assert devices == [torch.device("mps"), torch.device("mps")]
    assert "not process physical footprint" in result["note"]


@pytest.mark.parametrize("mode", ["missing", "raises", "empty", "invalid"])
def test_old_api_fallback_never_invents_zero(monkeypatch, mode):
    def unavailable(*args):
        raise RuntimeError("not supported")

    result = (
        {}
        if mode == "empty"
        else {
            "allocated_bytes.all.current": -1,
            "reserved_bytes.all.current": float("nan"),
            "allocated_bytes.all.peak": True,
        }
    )
    accelerator = (
        SimpleNamespace()
        if mode == "missing"
        else SimpleNamespace(
            memory_stats=unavailable if mode == "raises" else lambda device: result,
            reset_peak_memory_stats=unavailable,
        )
    )
    monkeypatch.setattr(torch, "accelerator", accelerator)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 17)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 43)
    sample = mps_memory_receipt()
    assert sample["current_allocated_bytes"] == 17
    assert sample["driver_allocated_bytes"] == 43
    assert sample["current_reserved_bytes"] is None
    assert sample["peak_allocated_bytes"] is None
    assert sample["peak_reserved_bytes"] is None
    assert not reset_mps_peak_memory()
    monkeypatch.setattr(torch.mps, "current_allocated_memory", unavailable)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", unavailable)
    sample = mps_memory_receipt()
    assert sample["current_allocated_bytes"] is None
    assert sample["driver_allocated_bytes"] is None


def test_runtime_reset_and_scope(monkeypatch):
    from taomate_h3.runtime import mps

    runtime = object.__new__(mps.MPSRuntime)
    calls = []
    runtime._synchronize = lambda: calls.append("sync")
    monkeypatch.setattr(mps, "mps_memory_receipt", lambda: {"peak_allocated_bytes": 12})
    for succeeded in (True, False):
        monkeypatch.setattr(mps, "reset_mps_peak_memory", lambda: succeeded)
        runtime.reset_dit_peak_memory()
        receipt = runtime.dit_peak_memory_receipt()
        assert receipt["peak_reset_succeeded"] is succeeded
        assert (receipt["peak_scope"] == "since_dit_reset") is succeeded
    assert calls == ["sync"] * 4


def test_mps_tiny_allocator_counters():
    if not torch.backends.mps.is_available():
        pytest.skip("Requires explicit native MPS test window")
    warm = torch.zeros(1, device="mps")
    torch.mps.synchronize()
    if not reset_mps_peak_memory():
        pytest.skip("Torch allocator peak API unavailable")
    before = mps_memory_receipt()
    value = torch.ones(1024 * 1024, device="mps", dtype=torch.float32)
    torch.mps.synchronize()
    after = mps_memory_receipt()
    assert after["current_allocated_bytes"] >= before["current_allocated_bytes"] + value.nbytes
    assert after["peak_allocated_bytes"] >= after["current_allocated_bytes"]
    assert after["peak_reserved_bytes"] >= after["current_reserved_bytes"]
    assert after["driver_allocated_bytes"] >= after["current_allocated_bytes"]
    assert value[0].item() == 1 and warm[0].item() == 0


def test_real_zero_is_preserved(monkeypatch):
    monkeypatch.setattr(
        torch,
        "accelerator",
        SimpleNamespace(
            memory_stats=lambda device: {
                "allocated_bytes.all.current": 0,
                "reserved_bytes.all.current": 0,
                "allocated_bytes.all.peak": 0,
                "reserved_bytes.all.peak": 0,
            }
        ),
    )
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 0)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: pytest.fail("fallback"))
    assert all(value == 0 for key, value in mps_memory_receipt().items() if key != "note")
