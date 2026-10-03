"""CPU-only exact-input generation comparison; numeric distances are not quality scores."""
import argparse, hashlib, importlib.util, json, math, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from safetensors.numpy import load_file
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--policy',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('comparison',a.source/'scripts/compare_experiments.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
roots=[a.baseline,a.candidate];configs=[json.loads((r/'run_config.json').read_text()) for r in roots]
assert {k:v for k,v in configs[0].items() if k!='output_dir'}=={k:v for k,v in configs[1].items() if k!='output_dir'}
reuse=[json.loads((r/'teacher-reuse.json').read_text()) for r in roots];assert reuse[0]['guidance_sha256']==reuse[1]['guidance_sha256']
receipts=[json.loads((r.parent/'process-receipt.json').read_text()) for r in roots]
for receipt in receipts:assert receipt['returncode']==0 and receipt['guard_reason'] is None
pre=json.loads((a.candidate.parent/'preflight.json').read_text());assert pre['source_sha256']==receipts[1]['source_sha256']
assert pre['source_sha256']=={str(f.relative_to(a.source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.source/'src').rglob('*.py')}
model=json.loads((a.candidate/'result.json').read_text())['model'];load=model['metal_weight_loading'];allocation=load['metal_allocation'];assert allocation['policy']==a.policy
assert load['tensor_count']==535 and model['adapter']['loaded_tensor_count']==416
assert load['metal_streamed_w8a16_linear_count']==len(allocation['quantized_linear_names'])
video=[r/'video.mp4' for r in roots];probes=[c.probe(v) for v in video];width,height=configs[0]['width'],configs[0]['height'];duration=configs[0]['duration_seconds']
for probe in probes:
 v=next(s for s in probe['streams'] if s['codec_type']=='video');au=next(s for s in probe['streams'] if s['codec_type']=='audio')
 assert (v['width'],v['height'],v['r_frame_rate'],int(v['nb_frames']))==(width,height,'24/1',duration*24)
 assert abs(float(v['duration'])-duration)<1e-6 and abs(float(au['duration'])-duration)<1e-6
 assert au['sample_rate']=='32000' and au['channels']==2
result=dict(baseline=str(a.baseline),candidate=str(a.candidate),inputs_and_teacher_exact=True,source_files_verified=len(pre['source_sha256']),policy=allocation,strict_decode=True,note='Layer/latent/pixel errors are not perceptual scores. The reference source predates exact lifecycle changes; candidate has explicit added base-weight quantization. Audio guidance is unchanged.',video=c.compare_video(*video,width*height*3),audio=c.compare_audio(*video),latents=c.compare_latents(*(r/'publication-latents.safetensors' for r in roots)))
arrays=[load_file(str(r/'publication-latents.safetensors')) for r in roots]
for name in arrays[0]:
 x,y=[d[name].astype(np.float64) for d in arrays];nx=np.linalg.norm(x);ny=np.linalg.norm(y)
 result['latents'][name].update(reference_rms=float(np.sqrt(np.mean(x*x))),candidate_rms=float(np.sqrt(np.mean(y*y))),cosine=float(np.dot(x.ravel(),y.ravel())/(nx*ny)),mean_absolute_error=float(np.mean(np.abs(y-x))))
result['environment_differences']={k:[receipts[0]['environment'].get(k),receipts[1]['environment'].get(k)] for k in set(receipts[0]['environment'])|set(receipts[1]['environment']) if receipts[0]['environment'].get(k)!=receipts[1]['environment'].get(k)}
result['changed_source_files']=[k for k in receipts[1]['source_sha256'] if receipts[0].get('source_sha256',{}).get(k)!=receipts[1]['source_sha256'][k]]
selected=[0,60,118,123,180,duration*24-1];frames=[]
for part in range(2):
 canvas=Image.new('RGB',(480,3*450),(255,255,255));draw=ImageDraw.Draw(canvas)
 for row,frame in enumerate(selected[part*3:part*3+3]):
  for col,path in enumerate(video):
   data=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(path),'-vf',f'select=eq(n\\,{frame})','-frames:v','1','-pix_fmt','rgb24','-f','rawvideo','-']);assert len(data)==width*height*3
   image=Image.frombytes('RGB',(width,height),data);image.thumbnail((240,432));canvas.paste(image,(col*240,row*450+18));draw.text((col*240+3,row*450+2),f'{"Reference" if col==0 else "Quant candidate"} frame {frame}',fill=(0,0,0))
 canvas.save(a.output/f'contact-{part+1}.jpg',quality=95)
result['contacts']=[str(a.output/f'contact-{i}.jpg') for i in [1,2]]
(a.output/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['video','audio','latents','contacts']},indent=2))
