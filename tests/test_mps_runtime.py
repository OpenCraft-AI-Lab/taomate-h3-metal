"""No model downloads: official state/visibility contracts on single-device torch."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from taomate_h3.config import DirectRunConfig, build_run_config
from taomate_h3.distributed import ParallelContext, heads_to_sequence, tp_all_gather
from taomate_h3.distributed.collectives import (
    tp_all_reduce,
    tp_all_reduce_inplace_,
    ulysses_all_gather_rows,
)
from taomate_h3.streaming.attention_hook import H3StreamingAttentionHook, HookMode
from taomate_h3.streaming.cache import AVKV, CleanAVKVCache, KVContract


def test_mps_config_roundtrip(tmp_path):
    c = build_run_config(
        model_root=tmp_path,
        adapter=tmp_path,
        output_dir=tmp_path / "out",
        prompt="ocean",
        prompt_json=None,
        duration_seconds=10,
        resolution="864x480",
        seed=1,
        gpu_count=8,
        devices=None,
        backend="mps",
    )
    assert (c.gpu_count, c.tensor_parallel_size, c.ulysses_degree, c.devices) == (1, 1, 1, ("mps",))
    assert DirectRunConfig.from_dict(c.to_dict()) == c
    assert c.request_seeds == (1, 1)


def test_single_process_does_not_call_distributed(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("collective called")

    for name in ("all_reduce", "all_gather", "all_to_all_single", "is_initialized"):
        monkeypatch.setattr(torch.distributed, name, forbidden)
    context = ParallelContext(1, 0, 1, 0, None, None)
    x = torch.arange(24).reshape(3, 2, 4)
    assert tp_all_reduce(x, context=context) is x
    assert tp_all_reduce_inplace_(x, context=context) is x
    assert tp_all_gather(x, context=context) is x
    assert ulysses_all_gather_rows(x, context=context) is x
    assert heads_to_sequence(x, context=context) is x
    with pytest.raises(ValueError):
        ParallelContext(2, 0, 1, 0, None, None)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("rows,history,condition", [(5, 3, 2), (8, 4, 3)])
def test_rectangular_visibility_against_fp64(device, rows, history, condition):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS requires native host access")
    generator = torch.Generator().manual_seed(rows)
    q, k, v = [torch.randn(rows, 2, 8, generator=generator) for _ in range(3)]
    hk, hv = [torch.randn(history, 2, 8, generator=generator) for _ in range(2)]
    dtype = torch.bfloat16
    q, k, v, hk, hv = [x.to(dtype).to(device) for x in (q, k, v, hk, hv)]
    cache = CleanAVKVCache(KVContract(local_heads=2, head_dim=8, dtype=dtype, device_type=device))
    cache._history["blocks.0.attn"] = AVKV(hk, hv)
    hook = H3StreamingAttentionHook(cache, backend="torch_sdpa")
    hook.activate(HookMode.NOISY)
    tags = torch.tensor([1] * condition + [2] * (rows - condition), device=device)
    mask = tags != 1
    result = hook(
        attention=SimpleNamespace(softmax_scale=8**-0.5),
        layer_name="blocks.0.attn",
        query=q,
        key=k,
        value=v,
        token_tags=tags,
        commit_mask=mask,
        cu_seqlens_host=(0, rows),
    )
    q, k, v, hk, hv = [x.cpu().double() for x in (q, k, v, hk, hv)]

    def oracle(qq, kk, vv):
        logits = torch.einsum("qhd,khd->hqk", qq, kk) * 8**-0.5
        return torch.einsum("hqk,khd->qhd", logits.softmax(-1), vv)

    ref = torch.cat(
        (
            oracle(q[:condition], k[:condition], v[:condition]),
            oracle(
                q[condition:],
                torch.cat((k[:condition], hk, k[condition:])),
                torch.cat((v[:condition], hv, v[condition:])),
            ),
        )
    )
    torch.testing.assert_close(result.cpu().float(), ref.float(), atol=0.02, rtol=0.02)
    assert cache._history["blocks.0.attn"].key is not None
    assert hook.kernel_calls == 2


def test_text_released_before_dit_load(monkeypatch):
    from taomate_h3.model.pipeline import MiniMaxH3NativePipeline
    from taomate_h3.runtime import backend

    class Text:
        pass

    pipe = object.__new__(MiniMaxH3NativePipeline)
    pipe.device = torch.device("mps")
    pipe.text_encoder = Text()
    ref = weakref.ref(pipe.text_encoder)
    seen = []
    pipe._load_transformer = lambda: seen.append(ref() is None)
    monkeypatch.setattr(backend, "empty_cache", lambda device: None)
    pipe._offload_text()
    assert seen == [True]


def test_cli_mps_does_not_spawn_torchrun(monkeypatch, tmp_path):
    from taomate_h3 import cli, runner

    seen = []
    monkeypatch.setattr(runner, "run_direct", lambda c, **kw: seen.append(kw))
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: pytest.fail("spawned subprocess"))
    cli.launch(SimpleNamespace(backend="mps"), tmp_path / "config", tmp_path / "guidance")
    assert seen == [{"audio_guidance_dir": tmp_path / "guidance"}]


def test_cpu_rng_contract_unchanged():
    from taomate_h3.model.geometry import direct_5s_geometry
    from taomate_h3.model.pipeline import MiniMaxH3NativePipeline

    geometry = direct_5s_geometry(width=864, height=480)
    # Small synthetic geometry keeps the exact draw ordering under test inexpensive.
    geometry = SimpleNamespace(
        video_latent_t=2, video_latent_h=2, video_latent_w=2, audio_latent_t=3
    )
    generator = torch.Generator(device="cpu").manual_seed(17)
    video = MiniMaxH3NativePipeline._initial_video_noise(geometry, generator=generator)
    audio = MiniMaxH3NativePipeline._initial_audio_noise(geometry, generator=generator)
    assert video.device.type == "cpu" and audio.device.type == "cpu"
    generator2 = torch.Generator(device="cpu").manual_seed(17)
    torch.randn(1, 24, 2, 2, 2, generator=generator2, dtype=torch.float32)
    expected = torch.randn(2 * 3, 32, generator=generator2, dtype=torch.float32)
    assert torch.equal(audio, expected)


@pytest.mark.parametrize("audio_only", [False, True])
def test_small_official_dit_forward_on_mps(monkeypatch, audio_only):
    """Exercise original refiner, timestep, core, RoPE and F32 heads together."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    import taomate_h3.model.dit as dit_module
    from taomate_h3.model.architecture import MiniMaxH3Architecture
    from taomate_h3.model.denoise import MiniMaxH3DenoiseBranch
    from taomate_h3.model.packed_sequence import (
        minimax_h3_audio_only_packed_sequence,
        minimax_h3_packed_sequence,
    )

    arch = MiniMaxH3Architecture(
        num_layers=2,
        token_refiner_num_layers=2,
        hidden_size=128,
        num_attention_heads=1,
        ffn_hidden_size=128,
        text_dim=16,
        timestep_input_dim=16,
        time_embed_hidden_size=128,
        time_embed_dim=64,
        adaln_out_features=18 * 128,
        final_adaln_out_features=2 * 128,
    )
    monkeypatch.setattr(dit_module, "MiniMaxH3Architecture", lambda: arch)
    context = ParallelContext(1, 0, 1, 0, None, None)
    model = dit_module.MiniMaxH3DiT(parallel_context=context).eval()
    with torch.no_grad():
        torch.manual_seed(17)
        for p in model.parameters():
            p.copy_(torch.randn_like(p.float()).mul(0.02).to(p.dtype))
        model.rope.inv_freq.copy_(torch.linspace(0.01, 1, 16))
    model = model.to("mps")
    factory = minimax_h3_audio_only_packed_sequence if audio_only else minimax_h3_packed_sequence
    kwargs = {} if audio_only else {"latent_t": 2}
    packed = factory(text_len=40, audio_t=3, latent_h=2, latent_w=2, **kwargs)
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.randn(40, 16).bfloat16(),
        token_tags=packed["token_tags"],
        device=torch.device("mps"),
        parallel_context=context,
    )
    with torch.no_grad():
        branch.static_kwargs["prompt_embeds"] = model.refine_prompt_embeds(
            branch.static_kwargs["prompt_embeds"],
            branch.refiner_cu_seqlens,
            device=torch.device("mps"),
        )
        branch.static_kwargs["rope_cache"] = model.build_rope_cache(
            branch.img_position_ids, device=torch.device("mps")
        )
        timestep = branch.prepare_timestep_plan(video_timesteps=[0.3], audio_timesteps=[0.7])[0]
        outputs = model(
            **branch.forward_kwargs(
                video_rows=torch.randn(0 if audio_only else 2, 96, device="mps"),
                audio_rows=torch.randn(6, 32, device="mps"),
                step_timesteps=timestep,
            )
        )
    assert all(x.dtype == torch.float32 and torch.isfinite(x).all() for x in outputs)
    assert outputs[0].shape == (0 if audio_only else 2, 96)
    assert outputs[1].shape == (6, 32)


@pytest.mark.parametrize("fail_at", [None, 0, 7, 49])
def test_mps_retention_releases_old_layers_and_invalidates_failure(monkeypatch, fail_at):
    import taomate_h3.streaming.cache as cache_module

    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    cache = CleanAVKVCache(KVContract(local_heads=1, head_dim=2, device_type="mps"))
    cache._commit_token_counts = [2] * 4
    cache._commit_token_tags = [(2, 0)] * 4
    old = []
    for name in cache_module.MAIN_LAYER_NAMES:
        key = torch.arange(16, dtype=torch.float32).reshape(8, 1, 2).bfloat16().to("mps")
        cache._history[name] = AVKV(key, key.clone())
        old.append(weakref.ref(key))
    del key
    validate = cache_module._validate_av_pair
    seen = []

    def check(pair, contract, *, layer_name):
        index = len(seen)
        if index:
            assert old[index - 1]() is None
        seen.append(layer_name)
        if index == fail_at:
            raise RuntimeError("injected retention allocation failure")
        validate(pair, contract, layer_name=layer_name)

    monkeypatch.setattr(cache_module, "_validate_av_pair", check)
    if fail_at is None:
        cache.retain_sink_and_recent_commits()
        assert len(seen) == 50 and cache.history_tokens == 5
        expected = torch.tensor([2, 8, 10, 12, 14], dtype=torch.bfloat16)
        for pair in cache._history.values():
            assert torch.equal(pair.key.cpu()[:, 0, 0], expected)
    else:
        with pytest.raises(RuntimeError, match="injected"):
            cache.retain_sink_and_recent_commits()
        assert not cache._history
        with pytest.raises(RuntimeError, match="invalidated"):
            cache.history("blocks.0.attn")
        with pytest.raises(RuntimeError, match="invalidated"):
            cache.begin_clean_commit(0)


def test_failed_mps_runtime_cannot_resume():
    from taomate_h3.streaming.runtime import H3StreamingRuntime

    runtime = object.__new__(H3StreamingRuntime)
    runtime._failed = True
    with pytest.raises(RuntimeError, match="invalidated"):
        runtime.run()
