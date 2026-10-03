"""Real BF16 base/LoRA weights, fixed synthetic BF16 inputs, bounded MPS W8A16 screening."""
from pathlib import Path
import argparse, gc, hashlib, json, os, subprocess, sys, time

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--expected-ref',required=True);p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rows',type=int,default=257);a=p.parse_args()
assert 1<=a.rows<=1024
assert subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()==a.expected_ref
manifest=json.loads(a.source_manifest.read_text());assert len(manifest)==53
assert {str(f.relative_to(a.source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.source/'src').rglob('*.py')}==manifest
assert os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']=='0.9' and os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0'
sys.path.insert(0,str(a.source/'src'))
import torch
from torch.nn import functional as F
from safetensors import safe_open
from taomate_h3.inference.metal_quantization import MetalW8A16Linear
from taomate_h3.runtime.mps_memory import mps_memory_receipt
base=Path('/Users/supreme-paca/Projects/h3-metal/MiniMax-H3/FL2VA/transformer');adapter=Path('/Users/supreme-paca/Models/TaoMate-H3/adapter_model.safetensors')
index=json.loads((base/'model.safetensors.index.json').read_text())['weight_map']
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'plan.json').write_text(json.dumps(dict(source_ref=a.expected_ref,rows=a.rows,targets='blocks2..46 attn.out_proj and mlp.fc2',inputs='fixed CPU torch.Generator; Gaussian and BF16 SiLU-gated synthetic activations',reference='original BF16 base F.linear; trained BF16 LoRA A/B rank128 alpha128 scale1; original mul/add boundaries',candidate='W8A16 per-row INT8 + F32 scales + bounded64MiB BF16 dequantized weight tiles',base_revision='42ed227ee7df40d41602854ae760620d6eb651fe',adapter_revision='7d7a51f3e63972138882ec2f0e4d1d47728110cf',scope='Candidate screening only: real weights with synthetic activations, not actual generation hidden states, CUDA parity or perceptual impact.'),indent=2))

def digest(t):
 return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()

def metrics(ref,out):
 r=ref.detach().float().cpu().double();v=out.detach().float().cpu().double();assert torch.isfinite(r).all() and torch.isfinite(v).all();d=v-r
 norm=torch.linalg.vector_norm(r);vn=torch.linalg.vector_norm(v)
 return dict(relative_l2=(torch.linalg.vector_norm(d)/norm.clamp_min(1e-30)).item(),cosine=((r.flatten()@v.flatten())/(norm*vn).clamp_min(1e-30)).item(),reference_rms=r.square().mean().sqrt().item(),error_rms=d.square().mean().sqrt().item(),max_absolute_error=d.abs().max().item(),reference_sha256=digest(ref),candidate_sha256=digest(out),finite=True)

results=[];start=time.monotonic()
with torch.no_grad(), safe_open(str(adapter),framework='pt',device='cpu') as bank, (a.output/'layers.jsonl').open('w') as log:
 for group,width in [('attn.out_proj',7168),('mlp.fc2',14336)]:
  for block in range(2,47):
   name=f'blocks.{block}.{group}';key=name+'.weight'
   with safe_open(str(base/index[key]),framework='pt',device='cpu') as shard:
    source=shard.get_tensor(key);assert source.dtype==torch.bfloat16 and tuple(source.shape)==(5376,width)
    weight_sha=digest(source)
    quant=MetalW8A16Linear.from_weight(source,device='mps',temporary_bytes=64*1024**2,load_rows=128)
    weight=source.to('mps');del source
   la=bank.get_tensor(name+'.lora_a').bfloat16().to('mps');lb=bank.get_tensor(name+'.lora_b').bfloat16().to('mps');assert tuple(la.shape)==(128,width) and tuple(lb.shape)==(5376,128)
   record=dict(layer=name,shape=list(weight.shape),source_weight_sha256=weight_sha,source_bf16_bytes=weight.nbytes,quantized_persistent_bytes=quant.receipt()['persistent_bytes'],saved_bytes=weight.nbytes-quant.receipt()['persistent_bytes'],cases={})
   for ci,case in enumerate(['gaussian','bf16_swiglu']):
    seed=9173+block*17+ci
    gen=torch.Generator(device='cpu').manual_seed(seed)
    x=torch.randn(a.rows,width,generator=gen).bfloat16()
    if case=='bf16_swiglu':x=F.silu(x)*torch.randn(a.rows,width,generator=gen).bfloat16()
    x=x.to('mps');x_sha=digest(x)
    ref=F.linear(x,weight);out=quant(x);torch.mps.synchronize()
    detail=dict(seed=seed,input_sha256=x_sha,base=metrics(ref,out))
    update=F.linear(F.linear(x,la),lb)
    ref_av=ref+update*1.0;out_av=out+update*1.0
    detail['with_trained_lora']=metrics(ref_av,out_av);assert digest(x)==x_sha
    record['cases'][case]=detail
    del x,ref,out,update,ref_av,out_av
   record['memory']=mps_memory_receipt();results.append(record);log.write(json.dumps(record)+'\n');log.flush()
   print(json.dumps(dict(event='weight_screened',layer=name,saved_bytes=record['saved_bytes'],combined_rel_l2={c:d['with_trained_lora']['relative_l2'] for c,d in record['cases'].items()})),flush=True)
   del quant,weight,la,lb;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
summary=dict(elapsed_seconds=time.monotonic()-start,layer_count=len(results),groups={})
for group in ['attn.out_proj','mlp.fc2']:
 rs=[r for r in results if r['layer'].endswith(group)];errs=[r['cases'][c]['with_trained_lora']['relative_l2'] for r in rs for c in r['cases']]
 summary['groups'][group]=dict(layer_count=len(rs),saved_bytes=sum(r['saved_bytes'] for r in rs),min_relative_l2=min(errs),mean_relative_l2=sum(errs)/len(errs),max_relative_l2=max(errs),worst_layers=sorted([dict(layer=r['layer'],max_relative_l2=max(d['with_trained_lora']['relative_l2'] for d in r['cases'].values())) for r in rs],key=lambda v:v['max_relative_l2'],reverse=True)[:8])
summary['combined'] = dict(layer_count=90,saved_bytes=sum(r['saved_bytes'] for r in results),joint_model_error=None,note='Joint generation error cannot be inferred by adding isolated layer errors; requires full generation comparison.')
(a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(dict(event='screen_complete',**summary)),flush=True)
