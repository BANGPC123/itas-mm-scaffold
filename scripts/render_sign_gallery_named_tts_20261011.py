"""Repair the prerecorded sign gallery: Vietnamese Unicode names and per-scene Edge TTS.

Precomputed 24 DEVELOPMENT still images, NOT a continuous driving video or
human ground-truth. Every audible sentence says AI predicts and is unverified.
The audio for each scene is aligned to its actual on-screen prediction.
"""
from pathlib import Path
import json,cv2,numpy as np,subprocess,wave,math,time
from PIL import Image,ImageDraw,ImageFont
from src.pipeline.orchestrator import Pipeline
from src.interaction.ai_observation import ai_observation
from src.perception.sign_catalog import NAMES_VI
from src.utils.config import load_config
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'output/review/field_review_pack_20261011'
OUT=ROOT/'output/capstone_sign_gallery_named_tts_20261011'
if (OUT/'summary.json').exists():raise FileExistsError('Will not overwrite gallery')
OUT.mkdir(parents=True,exist_ok=True)
doc=json.loads((SOURCE/'review_manifest.json').read_text(encoding='utf-8'))
p=Pipeline()
tts=p.tts_engine
font=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',25)
small=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',19)
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
scenes=[];audios={};t0=time.perf_counter()
def decode(path):
    proc=subprocess.run([ffmpeg,'-v','error','-i',str(path),'-vn','-ac','1','-ar','16000',
        '-f','s16le','-acodec','pcm_s16le','pipe:1'],capture_output=True,timeout=45)
    if proc.returncode!=0:raise RuntimeError('cannot decode TTS audio')
    return np.frombuffer(proc.stdout,dtype='<i2').astype(np.int32)

for j,f in enumerate(doc['frames']):
    im=cv2.imread(str(SOURCE/f['image']))
    if im is None:raise RuntimeError('Missing development frame')
    im=cv2.resize(im,(1280,720))
    r=p.run(im,10.7769,106.7009,False,annotate=False,road_bottom=.65)
    # Only recognized codes with real names appear on screen; unknown stays unknown.
    sign_details=[(s.label,NAMES_VI.get(s.label,'Chưa rõ mã biển'),
                   [int(z) for z in s.bbox_xyxy],s.confidence,s.recognition_confidence)
                  for s in r.perception.signs]
    text=ai_observation(r.perception.signs,video=False)
    audio=None;err=None
    if text:
        if text in audios:
            audio=audios[text]
        else:
            try:
                filename=tts.synthesize(text)
                audio=decode(filename)
                audios[text]=audio
            except Exception as exc:
                err=type(exc).__name__+': '+str(exc)[:90]
    dur=max(1.7,min(9.0,(len(audio)/16000+.7) if audio is not None else 1.7))
    # Preview labels have Unicode Vietnamese diacritics, unlike cv2.putText.
    frame=im.copy()
    for code,name,box,dc,rc in sign_details:
        a,b,c,d=box
        cv2.rectangle(frame,(a,b),(c,d),(0,210,255),3)
    cv2.rectangle(frame,(0,0),(1280,145),(17,25,38),-1)
    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    draw=ImageDraw.Draw(pil)
    draw.text((20,10),'ITAS-MM — Nhận dạng biển báo & tên tiếng Việt (AI dự đoán)',font=font,
              fill=(255,255,255))
    lines=[code+' — '+name for code,name,*_ in sign_details if code!='unknown'][:2]
    if not lines:lines=['Chưa xác định mã biển rõ trong khung hình này']
    for n,line in enumerate(lines):
        draw.text((20,55+33*n),line,font=font,fill=(255,210,78))
    draw.rectangle((0,666,1280,720),fill=(15,23,34))
    draw.text((18,679),
        'Cảnh '+str(j+1)+'/24 • Ảnh PHÁT TRIỂN • AI có thể sai • Không phải chỉ dẫn pháp luật',
        font=small,fill=(219,233,245))
    frame=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
    scenes.append({'frame':frame,'duration':dur,'audio':audio,'time_s':None,
      'scene':f['id'],'labels':[(code,name) for code,name,*_ in sign_details],
      'speech':text,'tts_error':err,'references':len(r.reasoning.references)})
    print('GALLERY_SCENE',j+1,'labels',lines,'voice',len(audio) if audio is not None else 0,
          'tts_error',err,flush=True)
fps=12
video=OUT/'video_named_silent.mp4'
writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,720))
if not writer.isOpened():raise RuntimeError('Cannot open video writer')
total=0
for row in scenes:
    row['time_s']=round(total/fps,3)
    count=max(1,math.ceil(row['duration']*fps))
    for _ in range(count):writer.write(row['frame'])
    total+=count
writer.release()
duration=total/fps
# Mix exact scene audio start times into a PCM mono WAV. No fictional voice
# if source OCR/recognition is unknown, or TTS network failed.
track=np.zeros(int((duration+.25)*16000),dtype=np.float32)
spoken=0
for row in scenes:
    if row['audio'] is None:continue
    offset=int((row['time_s']+.12)*16000)
    n=min(len(row['audio']),len(track)-offset)
    if n>0:
        track[offset:offset+n]+=row['audio'][:n].astype(np.float32)
        spoken+=1
track=np.clip(track,-32768,32767).astype('<i2')
wav=OUT/'aligned_narration.wav'
with wave.open(str(wav),'wb') as w:
    w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(track.tobytes())
dest=OUT/'video_named_with_tts.mp4'
proc=subprocess.run([ffmpeg,'-y','-loglevel','error','-i',str(video),'-i',str(wav),
    '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-t',str(duration),
    '-movflags','+faststart',str(dest)],capture_output=True,text=True,timeout=150)
if proc.returncode:raise RuntimeError('Audio/video mux failed: '+proc.stderr[:800])
check=cv2.VideoCapture(str(dest))
assert check.isOpened() and check.get(cv2.CAP_PROP_FRAME_COUNT)>=total*.98
check.release()
with (OUT/'scene_events.jsonl').open('w',encoding='utf-8') as f:
    for scene in scenes:
        row={k:v for k,v in scene.items() if k not in ('frame','audio')}
        f.write(json.dumps(row,ensure_ascii=False)+'\n')
summary={'source':'24 non-continuous development frames (not ground truth)',
 'video':str(dest),'resolution':[1280,720],'fps':fps,'duration':round(duration,2),
 'scenes':len(scenes),'spoken_scenes':spoken,
 'unique_voice_lines_generated_or_cached':len(audios),
 'source_frame_gt_verified':False,
 'unknown_and_low_confidence_may_be_silent':True,
 'scene_audio_aligned':True,'model_checkpoints_unchanged':True,
 'elapsed_wall_seconds':round(time.perf_counter()-t0,2)}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('NAMED_SIGN_VIDEO_WITH_REAL_TTS_READY',json.dumps(summary,ensure_ascii=False),flush=True)
