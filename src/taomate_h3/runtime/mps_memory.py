# Modified for TaoMate-H3 Metal inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Best-effort allocator telemetry; none of these counters is physical footprint."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import torch


def _bytes(value: object) -> int | None:
    return (
        int(value)
        if isinstance(value, Integral) and not isinstance(value, bool) and value >= 0
        else None
    )


def _direct(name: str) -> int | None:
    try:
        return _bytes(getattr(torch.mps, name)())
    except (AttributeError, RuntimeError, NotImplementedError, TypeError, ValueError):
        return None


def mps_memory_receipt() -> dict[str, int | str | None]:
    """Read A/R allocator counters and D driver memory without synchronizing.

    Explicitly target MPS so another available accelerator is never misreported.
    Missing APIs/keys remain None; direct MPS APIs retain older Torch support.
    """
    stats = {}
    try:
        result = torch.accelerator.memory_stats(torch.device("mps"))
        if isinstance(result, Mapping):
            stats = result
    except (AttributeError, RuntimeError, NotImplementedError, TypeError, ValueError):
        pass
    allocated = _bytes(stats.get("allocated_bytes.all.current"))
    if allocated is None:
        allocated = _direct("current_allocated_memory")
    return {
        "current_allocated_bytes": allocated,
        "current_reserved_bytes": _bytes(stats.get("reserved_bytes.all.current")),
        "driver_allocated_bytes": _direct("driver_allocated_memory"),
        "peak_allocated_bytes": _bytes(stats.get("allocated_bytes.all.peak")),
        "peak_reserved_bytes": _bytes(stats.get("reserved_bytes.all.peak")),
        "note": "Allocator peaks since last successful reset; not process physical footprint. "
        "Unavailable counters are null. Driver memory is a current sample, not a peak.",
    }


def reset_mps_peak_memory() -> bool:
    """Return whether the optional generic allocator reset call succeeded."""
    try:
        torch.accelerator.reset_peak_memory_stats(torch.device("mps"))
    except (AttributeError, RuntimeError, NotImplementedError, TypeError, ValueError):
        return False
    return True
