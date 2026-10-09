# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
# Modified for TaoMate-H3 Metal inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Device operations; MPS timings are synchronized wall time, never CUDA events."""

from __future__ import annotations

import time
from typing import Any

import torch


def require_mps() -> None:
    if not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; run on an Apple Silicon macOS host")


def synchronize(device: str | torch.device) -> None:
    device = torch.device(device)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def empty_cache(device: str | torch.device) -> None:
    if torch.device(device).type == "mps":
        torch.mps.empty_cache()
    elif torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


class WallEvent:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.timestamp = None

    def record(self) -> None:
        synchronize(self.device)
        self.timestamp = time.perf_counter()

    def elapsed_time(self, other: WallEvent) -> float:
        if self.timestamp is None or other.timestamp is None:
            raise RuntimeError("timing events must be recorded before reading elapsed time")
        return (other.timestamp - self.timestamp) * 1000


def timing_event(device: torch.device) -> Any:
    return torch.cuda.Event(enable_timing=True) if device.type == "cuda" else WallEvent(device)


def progress(event: str, **fields: Any) -> None:
    """Small host-side stage messages; no device sync or tensor copies."""
    import json
    import sys

    print(json.dumps({"event": event, **fields}), file=sys.stderr, flush=True)
