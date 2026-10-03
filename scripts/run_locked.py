"""Run an opt-in GPU verification command exclusively and keep its provenance."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.log.exists() or args.receipt.exists():
        parser.error("log and receipt must be new paths")
    repository = Path(__file__).resolve().parents[1]
    source = {
        str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((repository / "src").rglob("*.py"))
    }
    environment_keys = (
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_FAST_MATH",
        "PYTORCH_MPS_PREFER_METAL",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO",
        "MINIMAX_H3_TORCH_SDPA_BACKEND",
    )

    def swap():
        return subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True
        ).strip()

    receipt = {
        "command": command,
        "cwd": str(repository),
        "platform": platform.platform(),
        "python": sys.version,
        "source_sha256": source,
        "environment": {key: os.environ.get(key) for key in environment_keys},
        "swap_before": swap(),
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("a+") as lock, args.log.open("x") as logfile:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.perf_counter()
        with subprocess.Popen(
            command,
            cwd=repository,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as child:
            receipt["pid"] = child.pid
            print(f"GPU lock acquired; child pid={child.pid}", flush=True)
            for line in child.stdout:
                logfile.write(line)
                logfile.flush()
                print(line, end="", flush=True)
            returncode = child.wait()
        receipt.update(
            returncode=returncode,
            wall_seconds=time.perf_counter() - started,
            max_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            swap_after=swap(),
        )
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            json.dumps(
                {key: receipt[key] for key in ("returncode", "wall_seconds", "max_child_rss_bytes")}
            ),
            flush=True,
        )
        return returncode


if __name__ == "__main__":
    raise SystemExit(main())
