"""CPU pixel/FFmpeg equivalence and bounded publication ownership tests."""

import importlib
import os
import shutil
import sys
import types
from pathlib import Path

import pytest
import torch

from taomate_h3.inference.metal_publication import (
    encode_frame_chunks,
    revert_cpu_frames_inplace_,
    rgb24_frame_chunks,
)
from taomate_h3.model.output import save_h3_video_mp4


@pytest.fixture
def bundled_processor():
    root = (
        Path(
            os.environ.get(
                "TAOMATE_TEST_MODEL_ROOT", "/Users/supreme-paca/Projects/h3-metal/MiniMax-H3"
            )
        )
        / "FL2VA/video_vae"
    )
    if not (root / "vae_processor.py").is_file():
        pytest.skip("optional fixed upstream processor source unavailable")
    package = types.ModuleType("_tm_test_bundled_processor")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    cls = importlib.import_module(package.__name__ + ".vae_processor").VAEProcessor
    return cls(
        vae_ratio=16,
        vae_ratio_t=4,
        clip_length=17,
        frame_overlap=1,
        token_overlap=1,
        tokens_chunk_size=5,
        isolated_last_frame=False,
        latent_patch_size=2,
        crop_mode="center",
        pixel_norm_type="imagenet",
        use_3d_conv=True,
    )


@pytest.mark.parametrize("noncontiguous", [False, True])
def test_original_bundled_revert_exact_inplace(bundled_processor, noncontiguous):
    g = torch.Generator().manual_seed(93)
    frames = torch.randn(1, 3, 11, 8, 12, generator=g) * 4
    if noncontiguous:
        frames = frames.transpose(-1, -2)
        assert not frames.is_contiguous()
    expected = bundled_processor.revert_tensor(frames)
    storage = frames.untyped_storage().data_ptr()
    calls = []

    def revert(chunk):
        calls.append(chunk.shape[2])
        return bundled_processor.revert_tensor(chunk)

    actual = revert_cpu_frames_inplace_(frames, revert, frame_chunk_size=3)
    assert actual is frames and actual.untyped_storage().data_ptr() == storage
    assert torch.equal(actual, expected)
    assert calls == [3, 3, 3, 2]


@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
def test_rgb_bytes_rounding_layout_and_block_bound(dtype):
    # Halfway RGB values exercise original NumPy ties-to-even after F32 multiply.
    values = torch.tensor([-1, 0, 0.5 / 255, 1.5 / 255, 2.5 / 255, 127.5 / 255, 254.5 / 255, 1, 2])
    frames = values.repeat(3 * 13 * 8 * 10 // len(values) + 1)[: 3 * 13 * 8 * 10].reshape(
        1, 3, 13, 8, 10
    )
    if dtype == torch.uint8:
        frames = (frames.clamp(0, 1) * 255).round().byte()
    frames = frames.transpose(-1, -2)
    video = frames[0].permute(1, 2, 3, 0).numpy()
    if dtype != torch.uint8:
        video = (video.clip(0, 1) * 255.0).round().astype("uint8")
    chunks = list(rgb24_frame_chunks(frames, frame_chunk_size=4))
    assert b"".join(chunks) == video.tobytes()
    assert max(map(len, chunks)) <= 4 * 8 * 10 * 3
    assert len(chunks) == 4


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="requires ffmpeg")
@pytest.mark.parametrize("filtered", [False, True])
def test_actual_ffmpeg_mp4_bitwise_equivalence(tmp_path, filtered):
    g = torch.Generator().manual_seed(51)
    frames = torch.rand(1, 3, 17, 16, 24, generator=g).transpose(-1, -2)
    kwargs = {"video_filter": "fps=24:round=near", "published_frames": 17} if filtered else {}
    baseline = save_h3_video_mp4(tmp_path / "original.mp4", frames=frames, **kwargs)
    bounded = save_h3_video_mp4(
        tmp_path / "bounded.mp4", frames=frames, frame_chunk_size=4, **kwargs
    )
    assert baseline.read_bytes() == bounded.read_bytes()


def test_invalid_or_overlapping_storage_fails_before_mutation():
    frames = torch.ones(1, 3, 2, 4, 4)
    with pytest.raises(ValueError):
        revert_cpu_frames_inplace_(frames, lambda x: x, frame_chunk_size=0)
    overlapping = torch.as_strided(frames, (1, 3, 2, 4, 4), (0, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="overlap"):
        revert_cpu_frames_inplace_(overlapping, lambda x: x * 0, frame_chunk_size=1)
    assert frames.eq(1).all()


def test_encoder_failure_does_not_publish(tmp_path, monkeypatch):
    script = tmp_path / "fakeffmpeg"
    script.write_text('#!/bin/sh\nprintf "failure" >&2\nexit 3\n')
    script.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda _: str(script))
    target = tmp_path / "existing.mp4"
    target.write_bytes(b"previous good output")
    with pytest.raises(RuntimeError, match="failure"):
        save_h3_video_mp4(target, frames=torch.zeros(1, 3, 20, 16, 16), frame_chunk_size=2)
    assert target.read_bytes() == b"previous good output"


def test_large_stderr_does_not_deadlock(tmp_path):
    script = tmp_path / "noisy.py"
    script.write_text('import sys\nsys.stderr.write("x"*2000000)\nsys.exit(7)\n')
    result = encode_frame_chunks(
        [sys.executable, str(script)], torch.zeros(1, 3, 3, 8, 8), frame_chunk_size=1
    )
    assert result.returncode == 7 and len(result.stderr) == 1024**2


@pytest.mark.parametrize("enabled", [False, True])
def test_mps_runtime_explicit_switch_and_receipt(tmp_path, monkeypatch, enabled):
    import taomate_h3.model.output as output
    import taomate_h3.runtime.mps as mps

    frames = torch.zeros(1, 3, 2, 8, 8)
    captured = {}

    class Decoder:
        last_video_decode_receipt = {}
        video_vae = object()

        def decode_video(self, **kwargs):
            captured["decode"] = kwargs
            return frames

    decoder = Decoder()

    def save(path, **kwargs):
        captured["save"] = kwargs
        return path

    monkeypatch.setattr(output, "save_h3_video_mp4", save)
    monkeypatch.setattr(mps, "empty_cache", lambda _: None)
    monkeypatch.setenv("TAOMATE_MPS_STREAM_PUBLICATION", "1" if enabled else "0")
    runtime = mps.MPSRuntime.__new__(mps.MPSRuntime)
    runtime.device = torch.device("mps")
    runtime.pipeline = types.SimpleNamespace(_load_output_decoder=lambda: decoder)
    runtime._timings = {}
    runtime.decode_video_latents(torch.empty(0), output_path=tmp_path / "out.mp4")
    expected = 4 if enabled else None
    assert captured["decode"]["stream_publication_frame_chunk_size"] == expected
    assert captured["save"]["frame_chunk_size"] == expected
    assert runtime._video_vae_receipt["stream_publication"] is enabled
    assert runtime._video_vae_receipt["publication_frame_chunk_size"] == expected


def test_chunked_postprocess_rejects_non_mps_decoder_before_loading():
    from taomate_h3.model.output import H3LatentDecoder

    decoder = H3LatentDecoder.__new__(H3LatentDecoder)
    with pytest.raises(ValueError, match="MPS-only"):
        decoder.decode_video(
            video_latents=torch.empty(0),
            device=torch.device("cpu"),
            stream_publication_frame_chunk_size=4,
        )
