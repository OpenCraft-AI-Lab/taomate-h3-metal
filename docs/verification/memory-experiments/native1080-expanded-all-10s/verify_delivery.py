import argparse,json,subprocess,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('media',type=Path);p.add_argument('--seconds',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
video=a.media/'video.mp4';probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)]));v=next(s for s in probe['streams'] if s['codec_type']=='video');au=next(s for s in probe['streams'] if s['codec_type']=='audio');assert (v['width'],v['height'],v['r_frame_rate'],int(v['nb_frames']))==(1088,1920,'24/1',24*a.seconds);assert abs(float(v['duration'])-a.seconds)<1e-6;assert au['sample_rate']=='32000' and au['channels']==2 and abs(float(au['duration'])-a.seconds)<1e-6
subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-err_detect','explode','-i',str(video),'-map','0:v:0','-map','0:a:0','-f','null','-'],check=True)
audio=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(video),'-map','0:a:0','-f','f32le','-acodec','pcm_f32le','-']);arr=np.frombuffer(audio,np.float32).reshape(-1,2);assert np.isfinite(arr).all()
result=dict(strict_full_decode=True,video={k:v[k] for k in ['codec_name','width','height','pix_fmt','r_frame_rate','nb_frames','duration']},audio={k:au[k] for k in ['codec_name','sample_rate','channels','duration','duration_ts']},audio_finite=True,audio_decoded_samples_per_channel=len(arr),audio_rms=np.sqrt(np.mean(arr.astype(np.float64)**2,axis=0)).tolist(),audio_abs_peak=float(np.abs(arr).max()),aac_padding_note='decoded samples may include padding; stream presentation duration is authoritative')
from safetensors import safe_open
with safe_open(str(a.media/'publication-latents.safetensors'),framework='numpy') as f:
 result['latents']={}
 for name in f.keys():
  t=f.get_tensor(name);assert t.dtype==np.float32 and np.isfinite(t).all();result['latents'][name]=dict(shape=list(t.shape),finite=True,minimum=float(t.min()),maximum=float(t.max()));del t
frames=sorted(set([0,60,118,121,123,125,128,24*a.seconds-1]+[min(24*a.seconds-1,120*i) for i in range(2,a.seconds//5)]))
frames=frames[:25];images=[]
for n in frames:
 raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(video),'-vf',f'select=eq(n\\,{n}),scale=216:382','-vsync','0','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-']);im=Image.frombytes('RGB',(216,382),raw);tile=Image.new('RGB',(216,406),'white');tile.paste(im,(0,24));ImageDraw.Draw(tile).text((5,5),f'frame {n}, {n/24:.3f}s',fill='black');images.append(tile)
cols=5;canvas=Image.new('RGB',(216*cols,406*((len(images)+cols-1)//cols)),(230,230,230))
for i,im in enumerate(images):canvas.paste(im,((i%cols)*216,(i//cols)*406))
canvas.save(a.output/'contact.jpg',quality=90);result['contact_frames']=frames
(a.output/'delivery-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
