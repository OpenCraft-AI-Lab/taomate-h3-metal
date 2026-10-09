# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Golden data exported from the unmodified upstream, not from this backend."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pytest
import torch

from taomate_h3.config import DIRECT_POLICY
from taomate_h3.denoise_schedule import DISTILLED_STATE_INDICES, select_time_shift_sigmas
from taomate_h3.model.geometry import direct_5s_geometry
from taomate_h3.model.pipeline import MiniMaxH3NativePipeline
from taomate_h3.streaming.cache import MAIN_LAYER_NAMES, CleanAVKVCache, KVContract
from taomate_h3.streaming.geometry import canonical_continuation_plan, direct_5s_plan
from taomate_h3.teacher import _official_audio_noise

FIXTURE = json.loads((Path(__file__).parent / "fixtures/upstream_contracts.json").read_text())


def digest(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("golden", FIXTURE["noise"])
def test_official_cpu_noise_golden(golden):
    # Exact CPU RNG bytes are a fixed-environment oracle; do not impose this
    # Torch/architecture-specific hash on the upstream CUDA installation.
    environment = {
        "torch": torch.__version__,
        "system": platform.system(),
        "machine": platform.machine(),
    }
    if environment != FIXTURE["noise_fixture_environment"]:
        pytest.skip("regenerate upstream noise fixtures for this Torch/CPU environment")
    geometry = direct_5s_geometry(width=480, height=864)
    generator = torch.Generator().manual_seed(golden["video_seed"])
    video = MiniMaxH3NativePipeline._initial_video_noise(geometry, generator=generator)
    audio = _official_audio_noise(torch, geometry=geometry, seed=golden["audio_seed"])
    assert digest(video) == golden["video_sha256"]
    assert digest(audio) == golden["audio_sha256"]


def test_schedules_and_thirteen_request_geometries_match_upstream():
    assert DIRECT_POLICY.audio_kv_reset_window_requests == FIXTURE["audio_reset_window_requests"]
    for shift, golden in FIXTURE["stage3_sigmas"].items():
        assert (
            select_time_shift_sigmas(
                shift_scale=float(shift), state_indices=DISTILLED_STATE_INDICES
            )
            == golden
        )
    base = direct_5s_plan()
    for index, golden in FIXTURE["plans"].items():
        plan = (
            base if int(index) == 0 else canonical_continuation_plan(base, request_index=int(index))
        )
        assert plan.to_dict() == golden


@pytest.mark.parametrize("device", ["cpu", "mps"])
@torch.inference_mode()
def test_52_clean_commits_retention_and_audio_reset_match_upstream(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal device unavailable")
    cache = CleanAVKVCache(KVContract(local_heads=1, head_dim=2, device_type=device))
    tags = torch.tensor([1, 0, 2, 0, 2, 1], device=device)
    mask = (tags == 0) | (tags == 2)
    for golden in FIXTURE["cache_trace_13_requests"]:
        block = golden["block"]
        removed = cache.drop_audio_history() if block == 48 else 0
        assert removed == golden["audio_removed_before_commit"]
        cache.begin_clean_commit(block)
        for layer, name in enumerate(MAIN_LAYER_NAMES):
            # Host construction keeps the fixture identical across backends.
            key = (
                (torch.arange(12).reshape(6, 1, 2).float() / 16 + block + layer / 8)
                .bfloat16()
                .to(device)
            )
            cache.stage(name, key, -key, tags, mask)
        cache.commit()
        cache.retain_sink_and_recent_commits()
        assert cache.history_tokens == golden["tokens"]
        assert cache.history_video_tokens == golden["video"]
        assert cache.history_audio_tokens == golden["audio"]
        combined = torch.cat([cache.history(name).key.flatten() for name in MAIN_LAYER_NAMES])
        assert digest(combined) == golden["all_layer_keys_sha256"]
    assert cache.committed_blocks == 52
