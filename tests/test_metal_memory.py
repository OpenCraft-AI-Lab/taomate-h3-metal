"""Exact AdaLN release equivalence, ownership and rejection of schedule misses."""

import gc
import weakref

import pytest
import torch

from taomate_h3.denoise_schedule import DISTILLED_STATE_INDICES, select_time_shift_sigmas
from taomate_h3.distributed import ParallelContext
from taomate_h3.inference.metal_memory import precompute_release_adaln_
from taomate_h3.model.architecture import MiniMaxH3Architecture
from taomate_h3.model.denoise import MiniMaxH3DenoiseBranch
from taomate_h3.model.packed_sequence import minimax_h3_packed_sequence


@pytest.mark.parametrize("device,cpu_staged", (("cpu", False), ("mps", False), ("mps", True)))
@torch.no_grad()
def test_original_dit_output_and_schedule_cache_survive_release(monkeypatch, device, cpu_staged):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")
    import taomate_h3.model.dit as dit_module

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
    torch.manual_seed(31)
    for p in model.parameters():
        p.copy_(torch.randn_like(p.float()).mul(0.02).to(p.dtype))
    model.rope.inv_freq.copy_(torch.linspace(0.01, 1, 16))
    model = model.to(device)
    model.enable_inference_static_timestep_cache()
    packed = minimax_h3_packed_sequence(text_len=40, latent_t=2, latent_h=2, latent_w=2, audio_t=3)
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.randn(40, 16).bfloat16(),
        token_tags=packed["token_tags"],
        device=torch.device(device),
        parallel_context=context,
    )
    branch.static_kwargs["prompt_embeds"] = model.refine_prompt_embeds(
        branch.static_kwargs["prompt_embeds"],
        branch.refiner_cu_seqlens,
        device=torch.device(device),
    )
    branch.static_kwargs["rope_cache"] = model.build_rope_cache(
        branch.img_position_ids,
        device=torch.device(device),
    )
    sigmas_video, sigmas_audio = [
        select_time_shift_sigmas(
            num_steps=50, shift_scale=shift, state_indices=DISTILLED_STATE_INDICES
        )
        for shift in (12, 3)
    ]
    plans = branch.prepare_timestep_plan(
        video_timesteps=[1 - s for s in sigmas_video[:-1]] + [1.0],
        audio_timesteps=[1 - s for s in sigmas_audio[:-1]] + [1.0],
    )
    video = torch.randn(2, 96, device=device)
    audio = torch.randn(6, 32, device=device)
    expected = [
        tuple(
            t.cpu().clone()
            for t in model(
                **branch.forward_kwargs(video_rows=video, audio_rows=audio, step_timesteps=plan)
            )
        )
        for plan in plans
    ]
    # Clear caches so the optimized path must actually recompute all projections.
    model.enable_inference_static_timestep_cache()
    projections = [*(b.adaln_proj for b in model.blocks), model.final_layer.adaln_proj]
    transfer_observations = []
    handles = []
    if cpu_staged:
        for projection in projections:
            projection.linear.to("cpu")
        assert all(p.linear.weight.device.type == "cpu" for p in projections)

        def assert_single_resident_projection(module, _inputs):
            resident = [
                p.linear
                for p in projections
                if any(parameter.device.type == "mps" for parameter in p.linear.parameters())
            ]
            assert resident == [module]
            transfer_observations.append(id(module))

        handles = [
            p.linear.register_forward_pre_hook(assert_single_resident_projection)
            for p in projections
        ]
    refs = [weakref.ref(p.linear.weight) for p in projections]
    released = sum(
        p.numel() * p.element_size() for module in projections for p in module.linear.parameters()
    )
    receipt = precompute_release_adaln_(
        model, branch, sigmas_video=sigmas_video, sigmas_audio=sigmas_audio
    )
    for handle in handles:
        handle.remove()
    if cpu_staged:
        assert len(set(transfer_observations)) == len(projections)
        assert len(transfer_observations) == len(projections) * len(plans)
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert receipt["released_parameter_bytes"] == released
    assert receipt["projection_count"] == 3
    assert len(receipt["timestep_rows"]) == 4
    for plan, reference in zip(plans, expected):
        actual = model(
            **branch.forward_kwargs(video_rows=video, audio_rows=audio, step_timesteps=plan)
        )
        for output, previous in zip(actual, reference):
            torch.testing.assert_close(output.cpu(), previous, rtol=0, atol=0)
    assert (
        precompute_release_adaln_(
            model, branch, sigmas_video=sigmas_video, sigmas_audio=sigmas_audio
        )
        is receipt
    )
    unknown = branch.prepare_timestep_plan(video_timesteps=[0.123], audio_timesteps=[0.456])[0]
    with pytest.raises(RuntimeError, match="unprepared timestep"):
        model(**branch.forward_kwargs(video_rows=video, audio_rows=audio, step_timesteps=unknown))
    # Re-enabling caching after weights were released must also fail explicitly.
    model.enable_inference_static_timestep_cache()
    with pytest.raises(RuntimeError, match="unprepared timestep"):
        model(**branch.forward_kwargs(video_rows=video, audio_rows=audio, step_timesteps=plans[0]))
