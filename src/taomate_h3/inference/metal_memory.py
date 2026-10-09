# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
# Modified for TaoMate-H3 Metal inference, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Opt-in, exact fixed-schedule AdaLN precomputation for the Metal backend."""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import nn

from ..runtime.backend import empty_cache, progress, synchronize


class ReleasedAdaLNLinear(nn.Module):
    def forward(self, value):
        raise RuntimeError(
            "AdaLN weights were released after fixed-schedule precomputation; "
            "an unprepared timestep cannot be evaluated"
        )


@torch.no_grad()
def precompute_release_adaln_(model, branch, *, sigmas_video, sigmas_audio):
    """Warm the original identity-keyed caches, then free one projection at a time.

    Timestep rows come from the real denoise branch, including its FP32 unique
    ordering and any reference-audio clock. Neither AdaLN math nor its BF16
    boundaries change. Input tensors stay strongly owned by the original model
    caches, so their identity keys cannot be recycled after releasing weights.
    A schedule miss fails explicitly rather than computing with absent weights.
    """
    if getattr(model, "_metal_adaln_released", False):
        return model._metal_adaln_release_receipt
    if model.training or not model._inference_static_timestep_cache_enabled:
        raise RuntimeError("AdaLN release requires an eval model with static caching enabled")
    context = model.parallel_context
    if context.tp_world_size != 1 or context.ulysses_world_size != 1:
        raise ValueError("Metal AdaLN release requires a single-device model")
    if len(sigmas_video) != len(sigmas_audio) or len(sigmas_video) < 2:
        raise ValueError("video/audio sigma schedules must match")
    device = next(model.parameters()).device
    start = time.perf_counter()
    allocation_before = torch.mps.current_allocated_memory() if device.type == "mps" else None
    plans = branch.prepare_timestep_plan(
        video_timesteps=[1.0 - sigma for sigma in sigmas_video[:-1]] + [1.0],
        audio_timesteps=[1.0 - sigma for sigma in sigmas_audio[:-1]] + [1.0],
    )
    inputs = []
    timestep_rows = []
    for unique_timesteps, _, _ in plans:
        unique_timesteps = unique_timesteps.view(-1).to(device)
        timestep_rows.append(unique_timesteps.float().cpu().tolist())
        embedding = model.time_embedder(unique_timesteps)
        key = id(embedding)
        value = model._inference_static_adaln_input_cache.get(key)
        if value is None:
            value = F.silu(embedding).to(torch.bfloat16)
            model._inference_static_adaln_input_cache[key] = value
            model._inference_static_adaln_input_misses += 1
        inputs.append(value)
    modules = [*(block.adaln_proj for block in model.blocks), model.final_layer.adaln_proj]
    released_bytes = cached_bytes = 0
    for projection in modules:
        # CPU staging avoids allocating every large AdaLN matrix on Metal at
        # once. Each projection still executes the original BF16 MPS math.
        if next(projection.linear.parameters()).device != device:
            projection.linear.to(device)
        if not projection._inference_static_cache_enabled:
            raise RuntimeError("every AdaLN projection must have its original cache enabled")
        # Warm every live timestep before releasing this projection's parameters.
        for value in inputs:
            projection(value)
        cached_bytes += sum(
            tensor.numel() * tensor.element_size()
            for output in projection._inference_static_cache.values() for tensor in output
        )
        released_bytes += sum(p.numel() * p.element_size() for p in projection.linear.parameters())
        synchronize(device)
        projection.linear = ReleasedAdaLNLinear()
        empty_cache(device)
    model._metal_adaln_released = True
    receipt = {
        "mode": "exact_original_bf16_projection_cache_then_release",
        "projection_count": len(modules), "timestep_rows": timestep_rows,
        "released_parameter_bytes": released_bytes, "cached_output_bytes": cached_bytes,
        "precompute_seconds": time.perf_counter() - start,
        "unknown_timestep": "fail_closed",
        "mps_allocated_before": allocation_before,
        "mps_allocated_after": (
            torch.mps.current_allocated_memory() if device.type == "mps" else None
        ),
        "mps_driver_after": torch.mps.driver_allocated_memory() if device.type == "mps" else None,
    }
    model._metal_adaln_release_receipt = receipt
    progress("adaln_precomputed_and_released", **receipt)
    return receipt
