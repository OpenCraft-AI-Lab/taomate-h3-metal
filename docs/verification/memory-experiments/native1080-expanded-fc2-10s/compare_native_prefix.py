"""CPU-only lossy first-request latent comparison, never MP4 time-prefix equality."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from safetensors import safe_open
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
b=a.repo/'outputs/memory-experiments/native1080-packed8-5s';c=a.candidate
br=json.loads((b/'process-receipt.json').read_text());cr=json.loads((c/'process-receipt.json').read_text());pre=json.loads((c/'preflight.json').read_text())
assert cr['returncode']==0 and cr['guard_reason'] is None
actual={str(f.relative_to(a.repo)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.repo/'src').rglob('*.py')};assert actual==cr['source_sha256']==pre['source_sha256'] and len(actual)==53
bc=json.loads((b/'media/run_config.json').read_text());cc=json.loads((c/'media/run_config.json').read_text())
for name in ['seed','width','height','model_root','adapter']:assert bc[name]==cc[name]
assert bc['prompts'][0]==cc['prompts'][0] and bc['request_seeds'][0]==cc['request_seeds'][0]
assert json.loads((b/'media/teacher-reuse.json').read_text())['guidance_sha256']==json.loads((c/'media/teacher-reuse.json').read_text())['guidance_sha256']
result=dict(note='Lossy FC2 candidate versus original90 INT8 weights; same first prompt/seed/original teacher. Publication prefix only; whole-sequence VAE and retiming prevent MP4 prefix equality assumptions.',source_files_verified=53,source_changes=[k for k in actual if br['source_sha256'].get(k)!=actual[k]],latents={})
with safe_open(str(b/'media/publication-latents.safetensors'),framework='numpy') as bf,safe_open(str(c/'media/publication-latents.safetensors'),framework='numpy') as cf:
 for name in bf.keys():
  x=bf.get_tensor(name);y=cf.get_slice(name);shape=y.get_shape();assert len(shape)==x.ndim
  axis=2;assert all(shape[i]==x.shape[i] for i in range(x.ndim) if i!=axis)
  sl=[slice(None)]*x.ndim;sl[axis]=slice(0,x.shape[axis]);y=y[tuple(sl)];assert y.shape==x.shape and np.isfinite(x).all() and np.isfinite(y).all()
  # Bound float64 scratch to one temporal slice for native1080.
  err2=normx=normy=dot=absolute=0.;maxerr=0.;exact=True
  for i in range(x.shape[axis]):
   xx=np.take(x,i,axis=axis).astype(np.float64);yy=np.take(y,i,axis=axis).astype(np.float64);dd=yy-xx
   err2+=float(np.sum(dd*dd));normx+=float(np.sum(xx*xx));normy+=float(np.sum(yy*yy));dot+=float(np.sum(xx*yy));absolute+=float(np.sum(np.abs(dd)));maxerr=max(maxerr,float(np.max(np.abs(dd))));exact=exact and np.array_equal(xx,yy)
  result['latents'][name]=dict(prefix_shape=list(x.shape),full_candidate_shape=shape,exact=exact,relative_l2=(err2/normx)**.5,cosine=dot/(normx*normy)**.5,rms_error=(err2/x.size)**.5,reference_rms=(normx/x.size)**.5,candidate_rms=(normy/x.size)**.5,mean_absolute_error=absolute/x.size,max_absolute_error=maxerr)
  del x,y
assert result['latents']['audio']['exact'],'Frozen original teacher audio prefix must remain exact'
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
