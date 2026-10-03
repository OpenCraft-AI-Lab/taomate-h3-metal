# Modified for TaoMate-H3 Metal inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Best-effort allocator telemetry; none of these counters is physical footprint."""

from __future__ import annotations

import os
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


def memory_trace_enabled(device_type: str) -> bool:
    """Strict opt-in diagnostic, never active on CPU/CUDA."""
    return device_type == "mps" and os.environ.get("TAOMATE_MPS_MEMORY_TRACE") == "1"


def trace_mps_memory(point: str, **fields: int | str | bool | None) -> None:
    """Emit only scalar metadata, after synchronization, without tensor/frame retention.

    Callers gate by the actual tensor device. This diagnostic intentionally adds
    synchronization and must not be described as an uninstrumented timing run.
    Neither GC nor empty_cache is invoked, so live storage ownership is unchanged.
    """
    if not memory_trace_enabled("mps"):
        return
    if any(type(value) not in (int, str, bool, type(None)) for value in fields.values()):
        raise TypeError("memory trace accepts scalar metadata only")
    from .backend import progress

    torch.mps.synchronize()
    progress(
        "mps_allocation_boundary", point=point, synchronized=True, **mps_memory_receipt(), **fields
    )
