"""Watchdog decision boundaries and a real bounded CPU allocation termination."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_guarded.py"
spec = importlib.util.spec_from_file_location("taomate_test_memory_guard", SCRIPT)
guard = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = guard
spec.loader.exec_module(guard)


def test_memory_guard_limits_and_failed_telemetry():
    sample = dict(sample_ok=True, kernel_peak_footprint_bytes=guard.GIB,
                  available_estimate_bytes=30 * guard.GIB, system_swap_bytes=5 * guard.GIB)
    limits = guard.Limits()
    assert guard.violation(sample, 5 * guard.GIB, limits) is None
    for changes, expected in (
        ({"sample_ok": False}, "telemetry unavailable"),
        ({"kernel_peak_footprint_bytes": 106 * guard.GIB}, "footprint"),
        ({"available_estimate_bytes": 5 * guard.GIB}, "reserve"),
        ({"system_swap_bytes": 8 * guard.GIB}, "swap growth"),
    ):
        assert expected in guard.violation({**sample, **changes}, 5 * guard.GIB, limits)


@pytest.mark.skipif(sys.platform != "darwin", reason="native macOS watcher")
def test_watchdog_stops_bounded_cpu_allocation(tmp_path):
    watcher = Path("/private/tmp/taomate-memory-sample")
    if not watcher.exists():
        pytest.skip("compile scripts/memory_sample.c before native watchdog test")
    receipt = tmp_path / "receipt.json"
    code = ("import time; data = bytearray(96 * 1024**2); "
            "print('allocated', flush=True); time.sleep(20)")
    result = guard.run(
        [sys.executable, "-u", "-c", code], repository=tmp_path, watcher=watcher,
        lock_path=tmp_path / "lock", log_path=tmp_path / "log",
        memory_path=tmp_path / "memory.jsonl", receipt_path=receipt,
        limits=guard.Limits(footprint_bytes=64 * 1024**2),
    )
    recorded = json.loads(receipt.read_text())
    assert result == 125
    assert recorded["returncode"] < 0
    assert recorded["guard_reason"] == "process footprint exceeded budget"
    assert recorded["wall_seconds"] < 8
    assert recorded["memory_samples"] > 0
    assert 64 * 1024**2 < recorded["kernel_peak_footprint_bytes"] < guard.GIB


@pytest.mark.skipif(sys.platform != "darwin", reason="native macOS watcher")
def test_monitor_start_failure_cleans_up_child(monkeypatch, tmp_path):
    watcher = Path("/private/tmp/taomate-memory-sample")
    if not watcher.exists():
        pytest.skip("compile native memory watcher")
    original = guard.subprocess.Popen
    calls, children = [], []

    def inject(*args, **kwargs):
        calls.append(args)
        if len(calls) == 3:
            raise OSError("injected monitor start failure")
        result = original(*args, **kwargs)
        children.append(result)
        return result

    monkeypatch.setattr(guard.subprocess, "Popen", inject)
    with pytest.raises(OSError, match="injected"):
        guard.run(
            [sys.executable, "-c", "import time; time.sleep(20)"],
            repository=tmp_path, watcher=watcher,
            lock_path=tmp_path / "lock", log_path=tmp_path / "log",
            memory_path=tmp_path / "memory.jsonl", receipt_path=tmp_path / "receipt.json",
        )
    assert all(child.poll() is not None for child in children)
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["returncode"] < 0 and receipt["guard_reason"] == "launcher interrupted"


@pytest.mark.skipif(sys.platform != "darwin", reason="native macOS watchdog")
def test_launcher_sigterm_cleans_up_session(tmp_path):
    import os
    import subprocess

    watcher = Path("/private/tmp/taomate-memory-sample")
    if not watcher.exists():
        pytest.skip("compile native memory watcher")
    receipt = tmp_path / "receipt.json"
    command = [sys.executable, str(SCRIPT), "--watcher", str(watcher),
               "--lock", str(tmp_path / "lock"), "--log", str(tmp_path / "log"),
               "--memory", str(tmp_path / "memory.jsonl"), "--receipt", str(receipt),
               "--", sys.executable, "-c", "import time; time.sleep(20)"]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True) as launcher:
        first = launcher.stdout.readline()
        pid = int(first.split("child pid=")[1])
        # Ensure handler installation precedes the signal, without relying on a
        # fixed sleep to decide whether the experiment is running.
        import time
        deadline = time.monotonic() + 5
        while not (tmp_path / "memory.jsonl").stat().st_size:
            if time.monotonic() > deadline:
                pytest.fail("watcher did not start")
            time.sleep(0.01)
        launcher.terminate()
        launcher.communicate(timeout=8)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    recorded = json.loads(receipt.read_text())
    assert recorded["returncode"] < 0
    assert recorded["guard_reason"] == "launcher interrupted"
