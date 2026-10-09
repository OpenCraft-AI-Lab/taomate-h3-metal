# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
# Modified for TaoMate-H3 Metal inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Bounded CPU frame postprocessing/publication, preserving original arithmetic."""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable, Iterator, Sequence

import torch


def _check_chunk_size(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("frame_chunk_size must be a positive integer")


@torch.no_grad()
def revert_cpu_frames_inplace_(
    frames: torch.Tensor, revert_tensor: Callable, *, frame_chunk_size: int
) -> torch.Tensor:
    """Use the bundled processor unchanged on small temporal slices.

    Owns/mutates decoder output. Each original frame is processed exactly once;
    no normalization approximations, dtype conversions or full contiguous copies.
    """
    _check_chunk_size(frame_chunk_size)
    if frames.device.type != "cpu" or frames.dtype != torch.float32:
        raise ValueError("stream publication requires CPU FP32 decoded frames")
    if frames.ndim != 5 or frames.shape[:2] != (1, 3) or frames.shape[2] < 1:
        raise ValueError("expected nonempty [1,3,T,H,W] decoded frames")
    # Expanded/overlapping views are unsafe to mutate. Regular strided views and
    # transposes are supported; reject ambiguous overlap conservatively.
    extent = 1
    for stride, size in sorted(zip(frames.stride(), frames.shape)):
        if size == 1:
            continue
        if stride < extent:
            raise ValueError("decoded frame storage must not overlap")
        extent = stride * size
    for start in range(0, frames.shape[2], frame_chunk_size):
        destination = frames[:, :, start : start + frame_chunk_size]
        restored = revert_tensor(destination)
        if (
            restored.shape != destination.shape
            or restored.dtype != torch.float32
            or restored.device.type != "cpu"
        ):
            raise ValueError("bundled processor changed frame shape/dtype/device")
        destination.copy_(restored)
        del restored, destination
    return frames


def rgb24_frame_chunks(frames: torch.Tensor, *, frame_chunk_size: int) -> Iterator[bytes]:
    """Original NumPy operation order and rounding, one frame block at a time."""
    _check_chunk_size(frame_chunk_size)
    if frames.ndim != 5 or frames.shape[:2] != (1, 3) or frames.device.type != "cpu":
        raise ValueError("bounded publication expects CPU [1,3,T,H,W]")
    for start in range(0, frames.shape[2], frame_chunk_size):
        video = frames[0, :, start : start + frame_chunk_size].permute(1, 2, 3, 0).detach().numpy()
        if frames.dtype != torch.uint8:
            video = (video.clip(0, 1) * 255.0).round().astype("uint8")
        yield video.tobytes()


def encode_frame_chunks(
    command: Sequence[str], frames: torch.Tensor, *, frame_chunk_size: int
) -> subprocess.CompletedProcess:
    """Stream raw RGB with bounded buffers and nonblocking stderr consumption."""
    _check_chunk_size(frame_chunk_size)
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
        )
        try:
            assert process.stdin is not None
            try:
                for chunk in rgb24_frame_chunks(frames, frame_chunk_size=frame_chunk_size):
                    process.stdin.write(chunk)
            except BrokenPipeError:
                pass
            finally:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            returncode = process.wait()
            errors.seek(0)
            # Error reporting is bounded too, even if an encoder floods stderr.
            detail = errors.read(1024 * 1024)
            return subprocess.CompletedProcess(command, returncode, stderr=detail)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
