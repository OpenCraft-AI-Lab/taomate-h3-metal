# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Exclusive local experiment runner with an active macOS memory watchdog."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import selectors
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3


@dataclass(frozen=True)
class Limits:
    footprint_bytes: int = 105 * GIB
    min_available_bytes: int = 6 * GIB
    max_swap_growth_bytes: int = 2 * GIB
    max_wall_seconds: float = 7200


def violation(sample: dict, initial_swap: int, limits: Limits) -> str | None:
    if not sample.get("sample_ok"):
        return "memory telemetry unavailable"
    if sample["kernel_peak_footprint_bytes"] > limits.footprint_bytes:
        return "process footprint exceeded budget"
    if sample["available_estimate_bytes"] < limits.min_available_bytes:
        return "system reclaimable memory fell below reserve"
    if sample["system_swap_bytes"] - initial_swap > limits.max_swap_growth_bytes:
        return "system swap growth exceeded budget"
    return None


def stop_group(child: subprocess.Popen) -> None:
    # The child starts a fresh session. Never signal any pre-existing process.
    for sig, wait_seconds in ((signal.SIGTERM, 3), (signal.SIGKILL, 1)):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            continue
        # Remove any descendants that outlived the direct child.
        if sig == signal.SIGTERM:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return


def first_sample(watcher: Path, *, timeout: float = 3.0) -> dict:
    """Read one bounded JSON line and unconditionally reap this auxiliary process.

    Cleanup may deliberately ignore SIGTERM in the launcher. That disposition is
    inherited across exec, so terminate() cannot be relied on for a new watcher.
    SIGKILL is appropriate after its one required sample and cannot be ignored.
    Nonblocking reads also bound missing-newline and silent-watcher failures.
    """
    if timeout <= 0:
        raise ValueError("sampling timeout must be positive")
    with subprocess.Popen([str(watcher), str(os.getpid())], stdout=subprocess.PIPE) as process:
        try:
            assert process.stdout is not None
            descriptor = process.stdout.fileno()
            os.set_blocking(descriptor, False)
            deadline = time.monotonic() + timeout
            data = bytearray()
            with selectors.DefaultSelector() as selector:
                selector.register(descriptor, selectors.EVENT_READ)
                while b"\n" not in data:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError("memory telemetry sample timed out")
                    chunk = os.read(descriptor, 4096)
                    if not chunk:
                        raise RuntimeError("memory telemetry ended before a complete sample")
                    data.extend(chunk)
                    if len(data) > 64 * 1024:
                        raise RuntimeError("memory telemetry sample exceeds 64 KiB")
            sample = json.loads(data.split(b"\n", 1)[0])
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
    if not isinstance(sample, dict) or not sample.get("sample_ok"):
        raise RuntimeError("cannot start without working memory telemetry")
    return sample


def _validate_mps_low_watermark(value: float) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 0.8:
        raise ValueError("MPS low watermark must be finite and in (0, 0.8]")
    return float(value)


def run(command, *, repository, watcher, lock_path, log_path, memory_path, receipt_path,
        limits=Limits(), preflight_available_bytes=20 * GIB, mps_low_watermark=0.8):
    mps_low_watermark = _validate_mps_low_watermark(mps_low_watermark)
    for path in (log_path, memory_path, receipt_path):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    source = {
        str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((repository / "src").rglob("*.py"))
    }
    environment = dict(os.environ)
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    environment["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.9"
    environment["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = str(mps_low_watermark)
    env_receipt = {k: v for k, v in environment.items()
                   if k.startswith(("TAOMATE_MPS_", "PYTORCH_MPS_"))
                   or k == "PYTORCH_ENABLE_MPS_FALLBACK"}
    receipt = {
        "command": list(command), "cwd": str(repository), "source_sha256": source,
        "environment": env_receipt, "limits": vars(limits),
        "watcher_sha256": hashlib.sha256(watcher.read_bytes()).hexdigest(),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = first_sample(watcher)
        receipt["preflight"] = initial
        if initial["available_estimate_bytes"] < preflight_available_bytes:
            receipt.update(returncode=125, guard_reason="insufficient preflight memory")
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            return 125
        started = time.monotonic()
        state = {"sample": None, "last_sample_at": started, "reason": None, "samples": 0}
        peak_footprint = 0
        with log_path.open("x") as logfile, memory_path.open("x") as memoryfile:
            child = monitor = None
            output_thread = monitor_thread = None
            previous_handlers = {}

            def copy_output():
                for line in child.stdout:
                    logfile.write(line)
                    logfile.flush()
                    print(line, end="", flush=True)

            def sample_memory():
                nonlocal peak_footprint
                try:
                    for line in monitor.stdout:
                        sample = json.loads(line)
                        memoryfile.write(line)
                        memoryfile.flush()
                        state["sample"] = sample
                        state["last_sample_at"] = time.monotonic()
                        state["samples"] += 1
                        peak_footprint = max(peak_footprint,
                                             sample["kernel_peak_footprint_bytes"])
                        reason = violation(sample, initial["system_swap_bytes"], limits)
                        if reason and state["reason"] is None:
                            state["reason"] = reason
                except Exception as error:
                    state["reason"] = f"memory telemetry error: {type(error).__name__}"

            def interrupted(signum, frame):
                raise InterruptedError(f"launcher received signal {signum}")

            try:
                for signum in (signal.SIGTERM, signal.SIGHUP):
                    previous_handlers[signum] = signal.signal(signum, interrupted)
                child = subprocess.Popen(
                    command, cwd=repository, env=environment, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
                )
                receipt["pid"] = child.pid
                print(f"GPU lock and memory guard active; child pid={child.pid}", flush=True)
                monitor = subprocess.Popen(
                    [str(watcher), str(child.pid)], stdout=subprocess.PIPE, text=True, bufsize=1,
                )
                output_thread = threading.Thread(target=copy_output, daemon=True)
                monitor_thread = threading.Thread(target=sample_memory, daemon=True)
                output_thread.start()
                monitor_thread.start()
                while child.poll() is None:
                    now = time.monotonic()
                    if now - started > limits.max_wall_seconds:
                        state["reason"] = "experiment wall time exceeded budget"
                    if now - state["last_sample_at"] > 5:
                        state["reason"] = "memory telemetry stopped for five seconds"
                    if state["reason"]:
                        print("MEMORY GUARD: " + state["reason"], flush=True)
                        stop_group(child)
                        break
                    time.sleep(0.1)
            except BaseException:
                state["reason"] = "launcher interrupted"
                if child is not None:
                    stop_group(child)
                raise
            finally:
                # Cleanup covers failures starting the monitor or either thread,
                # not just errors after the polling loop was entered.
                try:
                    for signum in previous_handlers:
                        signal.signal(signum, signal.SIG_IGN)
                    if child is not None:
                        stop_group(child)
                        child.wait()
                    if monitor is not None:
                        if monitor.poll() is None:
                            monitor.terminate()
                        try:
                            monitor.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            monitor.kill()
                            monitor.wait()
                    for thread in (monitor_thread, output_thread):
                        if thread is not None and thread.ident is not None:
                            thread.join(timeout=3)
                    try:
                        postflight = first_sample(watcher)
                    except Exception as error:
                        postflight = {"sample_ok": False, "error": type(error).__name__}
                    receipt.update(
                        returncode=child.returncode if child is not None else 125,
                        guard_reason=state["reason"],
                        wall_seconds=time.monotonic() - started, memory_samples=state["samples"],
                        kernel_peak_footprint_bytes=peak_footprint,
                        last_sample=state["sample"], postflight=postflight,
                    )
                    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
                    print(json.dumps({k: receipt[k] for k in (
                        "returncode", "guard_reason", "wall_seconds", "kernel_peak_footprint_bytes"
                    )}), flush=True)
                finally:
                    for signum, handler in previous_handlers.items():
                        signal.signal(signum, handler)
        return 125 if state["reason"] else child.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("watcher", "lock", "log", "memory", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--footprint-gib", type=float, default=105)
    parser.add_argument("--reserve-gib", type=float, default=6)
    parser.add_argument("--swap-growth-gib", type=float, default=2)
    parser.add_argument("--max-wall-seconds", type=float, default=7200)
    parser.add_argument("--mps-low-watermark", type=float, default=0.8,
                        help="MPS adaptive allocation threshold in (0, 0.8]; high cap stays 0.9")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        _validate_mps_low_watermark(args.mps_low_watermark)
    except ValueError as error:
        parser.error(str(error))
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    if any(value <= 0 for value in (args.footprint_gib, args.reserve_gib,
                                   args.swap_growth_gib, args.max_wall_seconds)):
        parser.error("all safety limits must be positive")
    limits = Limits(int(args.footprint_gib * GIB), int(args.reserve_gib * GIB),
                    int(args.swap_growth_gib * GIB), args.max_wall_seconds)
    return run(
        command, repository=Path(__file__).resolve().parents[1],
        watcher=args.watcher.resolve(strict=True), lock_path=args.lock,
        log_path=args.log, memory_path=args.memory, receipt_path=args.receipt, limits=limits,
        mps_low_watermark=args.mps_low_watermark,
    )


if __name__ == "__main__":
    raise SystemExit(main())
