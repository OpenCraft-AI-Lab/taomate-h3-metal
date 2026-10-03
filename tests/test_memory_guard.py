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
    assert recorded["postflight"]["sample_ok"] is True
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


@pytest.mark.skipif(sys.platform != "darwin", reason="native macOS watcher")
def test_closed_stdout_after_child_spawn_cleans_up(monkeypatch, tmp_path):
    watcher = Path("/private/tmp/taomate-memory-sample")
    if not watcher.exists():
        pytest.skip("compile native memory watcher")
    original = guard.subprocess.Popen
    children = []

    def collect(*args, **kwargs):
        result = original(*args, **kwargs)
        children.append(result)
        return result

    calls = []

    def closed_stdout(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise BrokenPipeError("injected closed stdout")

    monkeypatch.setattr(guard.subprocess, "Popen", collect)
    monkeypatch.setattr(guard, "print", closed_stdout, raising=False)
    with pytest.raises(BrokenPipeError, match="injected"):
        guard.run(
            [sys.executable, "-c", "import time; time.sleep(20)"],
            repository=tmp_path, watcher=watcher,
            lock_path=tmp_path / "lock", log_path=tmp_path / "log",
            memory_path=tmp_path / "memory.jsonl", receipt_path=tmp_path / "receipt.json",
        )
    assert all(child.poll() is not None for child in children)
    recorded = json.loads((tmp_path / "receipt.json").read_text())
    assert recorded["returncode"] < 0


def _fake_watcher(tmp_path, body):
    path = tmp_path / "watcher"
    path.write_text("#!" + sys.executable + "\n" + body)
    path.chmod(0o755)
    return path


def test_first_sample_reaps_watcher_inheriting_ignored_sigterm(tmp_path):
    import os
    import signal
    import time

    pid_path = tmp_path / "pid"
    watcher = _fake_watcher(
        tmp_path,
        "import os,signal,time,json\n"
        f'open({str(pid_path)!r},"w").write(str(os.getpid()))\n'
        'print(json.dumps({"sample_ok":True,"term_ignored":signal.getsignal(signal.SIGTERM)==signal.SIG_IGN}),flush=True)\n'
        "time.sleep(30)\n",
    )
    previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    started = time.monotonic()
    try:
        sample = guard.first_sample(watcher, timeout=1)
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert sample["term_ignored"] is True
    assert time.monotonic() - started < 2
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), 0)


@pytest.mark.parametrize("output", ["", '{"sample_ok":true}', "x" * 70000 + "\n"])
def test_first_sample_bounds_incomplete_or_oversized_output_and_reaps(tmp_path, output):
    import os

    pid_path = tmp_path / "pid"
    watcher = _fake_watcher(
        tmp_path,
        "import os,time,sys\n"
        f'open({str(pid_path)!r},"w").write(str(os.getpid()))\n'
        f"sys.stdout.write({output!r});sys.stdout.flush()\n"
        "time.sleep(30)\n",
    )
    with pytest.raises((TimeoutError, RuntimeError)):
        guard.first_sample(watcher, timeout=0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), 0)


def test_successful_run_records_valid_postflight_under_cleanup_signal_mask(tmp_path):
    watcher = _fake_watcher(
        tmp_path,
        "import json,time\n"
        'sample={"sample_ok":True,"kernel_peak_footprint_bytes":1,"available_estimate_bytes":100000000000,"system_swap_bytes":0}\n'
        "while True:\n print(json.dumps(sample),flush=True);time.sleep(.05)\n",
    )
    receipt = tmp_path / "receipt.json"
    assert (
        guard.run(
            [sys.executable, "-c", 'print("done")'],
            repository=tmp_path,
            watcher=watcher,
            lock_path=tmp_path / "lock",
            log_path=tmp_path / "log",
            memory_path=tmp_path / "memory",
            receipt_path=receipt,
        )
        == 0
    )
    result = json.loads(receipt.read_text())
    assert result["preflight"]["sample_ok"] and result["postflight"]["sample_ok"]
    assert result["postflight"]["system_swap_bytes"] == 0
