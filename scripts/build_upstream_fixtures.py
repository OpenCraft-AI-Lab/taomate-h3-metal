# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""Export small contracts from an unmodified, pinned TaoMate-H3 source checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

PIN = "6b2f9984acf681e2f387188f5d1f3b26e504315b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, required=True, help="directory containing taomate_h3/"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = args.source_root.resolve()
    repo = Path(__file__).resolve().parents[1]
    files = sorted((reference / "taomate_h3").rglob("*.py"))
    sources = {}
    for path in files:
        relative = "src/" + str(path.relative_to(reference))
        expected = subprocess.check_output(["git", "-C", str(repo), "show", f"{PIN}:{relative}"])
        if path.read_bytes() != expected:
            raise RuntimeError("reference is not pinned upstream: " + relative)
        sources[relative] = hashlib.sha256(expected).hexdigest()
    sys.path.insert(0, str(reference))
    import torch

    import taomate_h3

    assert Path(taomate_h3.__file__).resolve().is_relative_to(reference)
    from taomate_h3.config import DIRECT_POLICY
    from taomate_h3.denoise_schedule import DISTILLED_STATE_INDICES, select_time_shift_sigmas
    from taomate_h3.model.geometry import direct_5s_geometry
    from taomate_h3.model.pipeline import MiniMaxH3NativePipeline
    from taomate_h3.streaming.cache import MAIN_LAYER_NAMES, CleanAVKVCache, KVContract
    from taomate_h3.streaming.geometry import canonical_continuation_plan, direct_5s_plan
    from taomate_h3.teacher import _official_audio_noise

    def digest(tensor):
        return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    geometry = direct_5s_geometry(width=480, height=864)
    noise = []
    for request_index in (0, 1, 12):
        video_seed = (8301 + request_index * 1_000_003) % (2**63)
        generator = torch.Generator().manual_seed(video_seed)
        video = MiniMaxH3NativePipeline._initial_video_noise(geometry, generator=generator)
        audio = _official_audio_noise(torch, geometry=geometry, seed=8301 + request_index)
        noise.append(
            {
                "request": request_index,
                "video_seed": video_seed,
                "audio_seed": 8301 + request_index,
                "video_sha256": digest(video),
                "audio_sha256": digest(audio),
            }
        )
    base = direct_5s_plan()
    plans = {
        str(i): (base if i == 0 else canonical_continuation_plan(base, request_index=i)).to_dict()
        for i in range(13)
    }
    cache = CleanAVKVCache(KVContract(local_heads=1, head_dim=2, device_type="cpu"))
    tags = torch.tensor([1, 0, 2, 0, 2, 1])
    mask = (tags == 0) | (tags == 2)
    trace = []
    for block in range(52):
        removed = cache.drop_audio_history() if block == 48 else 0
        cache.begin_clean_commit(block)
        for layer, name in enumerate(MAIN_LAYER_NAMES):
            key = (torch.arange(12).reshape(6, 1, 2).float() / 16 + block + layer / 8).bfloat16()
            cache.stage(name, key, -key, tags, mask)
        cache.commit()
        cache.retain_sink_and_recent_commits()
        combined = torch.cat([cache.history(name).key.flatten() for name in MAIN_LAYER_NAMES])
        trace.append(
            {
                "block": block,
                "tokens": cache.history_tokens,
                "video": cache.history_video_tokens,
                "audio": cache.history_audio_tokens,
                "audio_removed_before_commit": removed,
                "all_layer_keys_sha256": digest(combined),
            }
        )
    fixture = {
        "upstream_revision": PIN,
        "source_sha256": sources,
        "torch": torch.__version__,
        "noise_fixture_environment": {
            "torch": torch.__version__,
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "noise": noise,
        "plans": plans,
        "cache_trace_13_requests": trace,
        "audio_reset_window_requests": DIRECT_POLICY.audio_kv_reset_window_requests,
        "stage3_sigmas": {
            str(shift): select_time_shift_sigmas(
                shift_scale=shift, state_indices=DISTILLED_STATE_INDICES
            )
            for shift in (3.0, 12.0)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    print(json.dumps({"reference_files_verified": len(sources), "output": str(args.output)}))


if __name__ == "__main__":
    main()
