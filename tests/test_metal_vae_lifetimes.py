# Modified by OpenCraft AI Lab for the Metal/MPS port, 2026.
# Licensed under the MiniMax H3 Community License Agreement; see LICENSE.
"""CPU storage ownership and ordering checks; no model weights/GPU required."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from taomate_h3.model.output import H3LatentDecoder


@pytest.mark.parametrize("overlap", [False, True])
def test_temporal_window_storage_released_before_next_decode(overlap):
    def run(release):
        storage_refs = []
        live_at_next = []
        vae = SimpleNamespace(
            tokens_chunk_size=2,
            vae_ratio_t=2,
            token_drop=int(overlap),
            token_overlap=0,
            frame_pre_padding=0,
            frame_overlap=2 if overlap else 0,
        )
        output_frames = 16 if overlap else 12
        vae._decode_temporal_output_frame_plan = lambda *args: (output_frames, 0, output_frames)

        def adaptive(_):
            if storage_refs:
                live_at_next.append(storage_refs[-1]() is not None)
            # Own storage directly, so a slice's _base pins this object.
            value = torch.empty(1, 3, 8 if overlap else 4, 2, 2)
            value.fill_(len(storage_refs) + 1)
            storage_refs.append(weakref.ref(value))
            return value

        vae._adaptive_decode = adaptive

        def blend(previous, current, count, dim):
            assert dim == -3 and count == 2
            return torch.cat(
                ((previous[:, :, -count:] + current[:, :, :count]) / 2, current[:, :, count:]),
                dim=2,
            )

        vae.blend = blend
        decoder = H3LatentDecoder.__new__(H3LatentDecoder)
        decoder.video_parallel_rank = 0
        decoder.video_vae = SimpleNamespace(model=vae)
        decoder.enable_video_vae_cpu_temporal_output(release_window_aliases=release)
        output = vae._decode_temporal_streaming(torch.zeros(1, 24, 6, 1, 1), None, None, 3, 0, None)
        assert all(ref() is None for ref in storage_refs)
        return output, live_at_next

    baseline, old_live = run(False)
    optimized, new_live = run(True)
    assert old_live == [True, True]  # test exposes the real old alias lifetime
    assert new_live == [False, False]
    assert torch.equal(optimized, baseline)
    if overlap:
        assert (
            optimized[0, 0, :, 0, 0].tolist() == [1] * 4 + [1.5] * 2 + [2] * 2 + [2.5] * 2 + [3] * 6
        )
    else:
        assert optimized[0, 0, :, 0, 0].tolist() == [1] * 4 + [2] * 4 + [3] * 4


@pytest.mark.parametrize("encode_failure", [False, True])
def test_mps_vae_disposed_before_encoding_without_cpu_weight_copy(
    monkeypatch, tmp_path, encode_failure
):
    import taomate_h3.model.output as output
    import taomate_h3.runtime.mps as mps

    events = []

    class Weights:
        def cpu(self):
            raise AssertionError("must not copy disposable VAE weights back to CPU")

    class Decoder:
        def __init__(self):
            self.video_vae = Weights()
            self.last_video_decode_receipt = {}

        def decode_video(self, **kw):
            assert kw["offload_after_decode"] is False
            events.append("decode")
            return torch.zeros(1, 3, 2, 8, 8)

    decoder = Decoder()
    reference = weakref.ref(decoder.video_vae)

    def save(path, **kw):
        assert reference() is None and decoder.video_vae is None
        assert events == ["decode", "empty_cache"]
        events.append("encode")
        if encode_failure:
            raise RuntimeError("injected encoder failure")
        return path

    monkeypatch.setattr(output, "save_h3_video_mp4", save)
    monkeypatch.setattr(mps, "empty_cache", lambda _: events.append("empty_cache"))
    runtime = mps.MPSRuntime.__new__(mps.MPSRuntime)
    runtime.device = torch.device("mps")
    runtime.pipeline = SimpleNamespace(_load_output_decoder=lambda: decoder)
    runtime._timings = {}
    if encode_failure:
        with pytest.raises(RuntimeError, match="injected"):
            runtime.decode_video_latents(torch.empty(0), output_path=tmp_path / "out.mp4")
    else:
        runtime.decode_video_latents(torch.empty(0), output_path=tmp_path / "out.mp4")
    assert reference() is None and decoder.video_vae is None


def test_failed_mps_decode_also_discards_vae(monkeypatch, tmp_path):
    import taomate_h3.runtime.mps as mps

    class Weights:
        pass

    def fail(**kwargs):
        raise RuntimeError("decode failure")

    decoder = SimpleNamespace(video_vae=Weights(), decode_video=fail)
    reference = weakref.ref(decoder.video_vae)
    monkeypatch.setattr(mps, "empty_cache", lambda _: None)
    runtime = mps.MPSRuntime.__new__(mps.MPSRuntime)
    runtime.device = torch.device("mps")
    runtime.pipeline = SimpleNamespace(_load_output_decoder=lambda: decoder)
    with pytest.raises(RuntimeError, match="decode failure"):
        runtime.decode_video_latents(torch.empty(0), output_path=tmp_path / "out.mp4")
    assert reference() is None and decoder.video_vae is None
