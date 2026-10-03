"""Re-publish already saved original decoder inputs; no teacher or DiT generation."""
import argparse, hashlib, json, time
from pathlib import Path
from types import SimpleNamespace
import torch
from safetensors.torch import load_file
from taomate_h3.config import DirectRunConfig
from taomate_h3.model.pipeline import MiniMaxH3NativePipeline
from taomate_h3.model.components import partition_root
from taomate_h3.runtime.mps import MPSRuntime
from taomate_h3.runner import _publish

p=argparse.ArgumentParser(); p.add_argument('--baseline',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
baseline=json.loads((args.baseline/'result.json').read_text())
latent_path=args.baseline/'publication-latents.safetensors'
tensors=load_file(str(latent_path),device='cpu')
args.output.mkdir(parents=True,exist_ok=False)
raw=baseline['config']; raw['output_dir']=str(args.output.resolve()); config=DirectRunConfig.from_dict(raw)
pipeline=MiniMaxH3NativePipeline.__new__(MiniMaxH3NativePipeline)
pipeline.partition_root=partition_root(config.model_root); pipeline.device=torch.device('mps'); pipeline.rank=0; pipeline.world_size=1; pipeline.output_decoder=None
runtime=MPSRuntime.__new__(MPSRuntime)
runtime.pipeline=pipeline; runtime.device=pipeline.device; runtime.rank=0; runtime.world_size=1; runtime._timings={}
executions=[SimpleNamespace(**e) for e in baseline['streaming']['executions']]
video=[]; audio=[]; vo=ao=0
for execution in executions:
 n=int(execution.published_video_latents); m=int(execution.published_audio_latents_per_channel)
 vp=int(execution.video_transport_prefix_latents); ap=int(execution.audio_transport_prefix_latents_per_channel)
 video.append(tensors['video'][:,:,vo-vp:vo+n].contiguous())
 audio.append(tensors['audio'][:,:,ao-ap:ao+m].contiguous())
 vo+=n; ao+=m
assert vo==tensors['video'].shape[2] and ao==tensors['audio'].shape[2]
start=time.perf_counter()
publication=_publish(runtime=runtime,config=config,streaming=SimpleNamespace(executions=executions),video_segments=video,audio_segments=audio)
result={'role':'fixed_latent_decode_only_no_teacher_no_dit','latent_source':str(latent_path.resolve()),'latent_sha256':hashlib.sha256(latent_path.read_bytes()).hexdigest(),'publication':publication,'timings_seconds':runtime._timings,'wall_seconds':time.perf_counter()-start}
(args.output/'decode-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
