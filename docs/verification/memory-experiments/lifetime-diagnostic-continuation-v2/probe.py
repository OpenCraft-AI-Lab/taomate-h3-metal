#!/usr/bin/env python3
"""Prepared only: real-shape synthetic single-layer lifetime A/B, never a model run."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import weakref


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path)
    p.add_argument('--expected-ref')
    p.add_argument('--plan', action='store_true')
    p.add_argument('--case', choices=['first', 'continuation'], default='first')
    return p


def main():
    a = parser().parse_args()
    media = 24610 if a.case == 'first' else 20514
    text = 20
    used = media + text
    rows = ((used + 63) // 64) * 64
    plan = dict(case=a.case, media_rows=media, text_rows=text, padded_rows=rows,
                hidden=5376, heads=56, head_dim=128, ffn=14336, rank=128,
                query_chunk=512, kv_tile=1024, allocation='packed8',
                staged_layers=1, committed_layers=0,
                packed_stage_backing_bytes=2*8*media*56*128+2*50*media*56*4,
                guard_physical_gib=10,
                limitation='Single actual layer stages its full K/V but does not commit 50 layers; '
                           'fixed baseline-then-candidate order, synthetic weights/hidden, not model parity.')
    if a.plan:
        print(json.dumps(plan, indent=2)); return
    if not a.source_manifest or not a.expected_ref:
        raise SystemExit('Pinned ref and complete 53-file manifest required')
    actual_ref = subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()
    if actual_ref != a.expected_ref:
        raise SystemExit('Source ref mismatch')
    manifest = json.loads(a.source_manifest.read_text())
    if not isinstance(manifest, dict) or len(manifest) != 53:
        raise SystemExit('Manifest must map exactly 53 src relative paths to SHA256')
    actual_paths = {str(p.relative_to(a.source)) for p in (a.source/'src').rglob('*.py')}
    if actual_paths != set(manifest):
        raise SystemExit('Source file inventory mismatch')
    for name, digest in manifest.items():
        if hashlib.sha256((a.source/name).read_bytes()).hexdigest() != digest:
            raise SystemExit('Source hash mismatch: '+name)
    if os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO') != '0.9':
        raise SystemExit('Must run through guarded launcher with HIGH=.9')
    sys.path.insert(0, str(a.source/'src'))
    import torch
    from types import SimpleNamespace
    from taomate_h3.distributed import ParallelContext
    from taomate_h3.model import layers
    from taomate_h3.inference import lora_checkpoint as lora, metal_activations as act
    from taomate_h3.streaming import attention_hook as hooks
    from taomate_h3.streaming.cache import KVContract
    from taomate_h3.streaming.metal_kv import QuantizedCleanAVKVCache
    from taomate_h3.runtime.mps_memory import mps_memory_receipt
    torch.set_grad_enabled(False)
    os.environ['TAOMATE_MPS_INPLACE_ACTIVATIONS']='1'
    if not torch.backends.mps.is_available(): raise SystemExit('MPS unavailable')
    def emit(event, **kw):
        torch.mps.synchronize()
        print(json.dumps(dict(event=event, **kw, memory=mps_memory_receipt())),flush=True)
    def digest(t):
        # Bounded copies; hash every element in logical contiguous order.
        h=hashlib.sha256()
        for start in range(0,len(t),128):
            h.update(t[start:start+128].detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    g=torch.Generator().manual_seed(9183)
    def fill(t):
        for start in range(0,len(t),32):
            part=torch.randn(t[start:start+32].shape,generator=g,dtype=torch.float32)*.01
            t[start:start+32].copy_(part.to(dtype=t.dtype,device=t.device))
    arch=SimpleNamespace(hidden_size=5376,ffn_hidden_size=14336,num_attention_heads=56,
                         attention_head_dim=128,qk_norm_eps=1e-6)
    context=ParallelContext(1,0,1,0,None,None)
    with torch.device('meta'):
        model=torch.nn.Module()
        model.attn=layers.MiniMaxH3Attention(arch,context=context)
        model.mlp=layers.MiniMaxH3MLP(arch,context=context)
    model.to_empty(device='mps').eval()
    for p in model.parameters(): fill(p)
    targets=('attn.qkv_proj','attn.out_proj','mlp.fc1','mlp.fc2')
    state={}
    for name in targets:
        module=model.get_submodule(name)
        for leaf,shape in [('lora_a',(128,module.input_size)),('lora_b',(module.output_size,128))]:
            state[name+'.'+leaf]=torch.randn(shape,generator=g)*.01
    old_targets,old_load=lora.canonical_h3_lora_targets,lora.load_h3_lora_checkpoint
    try:
        lora.canonical_h3_lora_targets=lambda:targets
        lora.load_h3_lora_checkpoint=lambda _:lora.LoadedH3LoRA(128,128,state,8)
        lora.apply_h3_lora_checkpoint(model,'synthetic')
    finally:
        lora.canonical_h3_lora_targets, lora.load_h3_lora_checkpoint=old_targets,old_load
    lora.materialize_h3_lora_bf16_buffers_(model)
    del state
    backing=torch.empty((rows,5376*2),device='mps',dtype=torch.bfloat16)
    fill(backing)
    x=backing[:,::2]  # intentionally noncontiguous, unchanged in both variants
    input_hash=digest(x)
    positions=torch.arange(rows,device='mps')
    cos_sin=torch.empty((rows,128),device='mps',dtype=torch.bfloat16);fill(cos_sin)
    tags=torch.full((rows,),-1,device='mps',dtype=torch.long)
    tags[:text]=1; tags[text:text+2]=2; tags[text+2:used]=0
    mask=(tags==0)|(tags==2)
    model.attn.layer_name='blocks.0'
    original_release,original_cleanup=act.can_release_projection,hooks._release_clean_temporaries_enabled
    results={}
    print(json.dumps(dict(event='plan',source_ref=actual_ref,**plan)),flush=True)
    for variant in ('baseline','candidate'):
        gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
        owners=[]
        def release(projected,inputs,*consumers):
            owners.append(weakref.ref(projected))
            decision=original_release(projected,inputs,*consumers)
            return variant=='candidate' and decision
        act.can_release_projection=release
        hooks._release_clean_temporaries_enabled=(original_cleanup if variant=='candidate' else lambda *args:False)
        cache=QuantizedCleanAVKVCache(KVContract(local_heads=56,device_type='mps'),
                                     allocation='packed8',tile_tokens=1024)
        cache.max_joined_bytes=4*1024**3
        hook=hooks.H3StreamingAttentionHook(cache,backend='torch_sdpa',query_chunk_size=512)
        cache.begin_clean_commit(0);hook.activate(hooks.HookMode.CLEAN_COMMIT)
        temporary_refs=[]
        original_attention=hook._attention
        def traced_attention(query,parts,**kwargs):
            out=original_attention(query,parts,**kwargs)
            temporary_refs.append(weakref.ref(out))
            return out
        hook._attention=traced_attention
        original_history=cache.attention_with_history
        def traced_history(name,query,condition,current,**kwargs):
            temporary_refs.extend(weakref.ref(t) for t in
                (query,condition.key,condition.value,current.key,current.value))
            out=original_history(name,query,condition,current,**kwargs)
            temporary_refs.append(weakref.ref(out))
            return out
        cache.attention_with_history=traced_history
        original_stage=cache.stage
        def stage(*args,**kwargs):
            emit('kv_stage_entry',variant=variant,projection_owner_alive=[r() is not None for r in owners],
                 hook_temporary_alive=[r() is not None for r in temporary_refs])
            if variant=='candidate' and any(r() is not None for r in temporary_refs):
                raise RuntimeError('Hook temporaries survived to stage')
            return original_stage(*args,**kwargs)
        cache.stage=stage
        handles=[model.attn.out_proj.register_forward_pre_hook(lambda *_:emit('out_proj_entry',variant=variant,
                    projection_owner_alive=[r() is not None for r in owners])),
                 model.mlp.fc2.register_forward_pre_hook(lambda *_:emit('fc2_entry',variant=variant,
                    projection_owner_alive=[r() is not None for r in owners]))]
        layers.set_native_stream_attention_hook(hook)
        try:
            with layers.native_stream_attention_metadata(tags,mask):
                y=model.attn(x,rope_cache=(cos_sin,positions),cu_seqlens_host=(0,used,rows),
                             max_seqlen=used,ulysses_active=True)
            record={'attention_sha256':digest(y)};del y
            pair=cache._staged['blocks.0.attn']
            record['kv']={name:digest(getattr(pair,name)) for name in ('key','value','key_scale','value_scale')}
            record['cache_receipt']=cache.receipt()
            del pair
            cache.rollback();layers.set_native_stream_attention_hook(None)
            gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
            y=model.mlp(x);record['mlp_sha256']=digest(y);del y
            if digest(x)!=input_hash: raise RuntimeError('Input changed')
            results[variant]=record
            emit('variant_complete',variant=variant,**record)
        finally:
            for h in handles:h.remove()
            layers.set_native_stream_attention_hook(None)
            hook.deactivate();cache.clear()
            del cache,hook,original_stage,original_history,original_attention,handles
            gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
    act.can_release_projection=original_release
    hooks._release_clean_temporaries_enabled=original_cleanup
    if results['baseline']!=results['candidate']:raise RuntimeError('Variant digest/receipt mismatch')
    emit('comparison_passed',comparison='SHA256 of every BF16 output byte and every INT8 KV/F32 scale byte')

if __name__=='__main__':main()
