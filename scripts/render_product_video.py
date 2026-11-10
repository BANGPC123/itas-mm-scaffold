"""Native-frame replay with sampled inference, Vietnamese labels and cached Edge TTS."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
import wave

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline.orchestrator import Pipeline
from src.perception.sign_catalog import display_name


def mix_speech(events,engine,out,duration,*,max_failures=3):
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    rate=24000;audio=np.zeros(round(duration*rate),dtype=np.int16);busy_until=0;spoken=[];failures=[]
    for event in events:
        if event['timestamp']<busy_until:continue
        try:
            path=engine.synthesize(event['text'])
            raw=subprocess.check_output([ffmpeg,'-v','error','-i',path,'-f','s16le','-ac','1','-ar',str(rate),'-'])
            samples=np.frombuffer(raw,dtype=np.int16);start=round(event['timestamp']*rate)
            end=min(len(audio),start+len(samples));audio[start:end]=samples[:end-start]
            busy_until=end/rate+1;spoken.append({**event,'end':end/rate,'audio':path})
        except Exception as exc:
            failures.append({**event,'error':str(exc)})
        if max_failures is not None and len(failures)>=max_failures:break
    with wave.open(str(out/'speech.wav'),'wb') as stream:
        stream.setnchannels(1);stream.setsampwidth(2);stream.setframerate(rate);stream.writeframes(audio.tobytes())
    partial=out/'with_audio.partial.mp4'
    subprocess.run([ffmpeg,'-y','-v','error','-i',str(out/'silent.mp4'),'-i',str(out/'speech.wav'),
        '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','96k','-movflags','+faststart',str(partial)],check=True)
    partial.replace(out/'test_video_annotated.mp4')
    return spoken,failures


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'output/product_20261010')
    parser.add_argument('--seconds',type=float)
    parser.add_argument('--inference-fps',type=float,default=10)
    parser.add_argument('--audio-only',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.inference_fps<=0:parser.error('--inference-fps must be positive')
    if args.audio_only:
        from src.interaction.sign_speech import sign_speech
        from src.interaction.tts_engine import TtsEngine
        from src.utils.config import load_config
        from src.utils.types import SignDetection
        summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
        events=[];last_spoken={}
        with (out/'predictions.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);ts=row['timestamp']
                text=sign_speech([SignDetection(**s) for s in row['signs']])
                if text and ts-last_spoken.get(text,-100)>20:
                    events.append({'timestamp':ts,'text':text});last_spoken[text]=ts
        spoken,failures=mix_speech(events,TtsEngine(load_config('interaction')),out,summary['output_frames']/summary['output_fps'])
        summary.update(speech_events=spoken,tts_failures=failures,speech_order_normalized=True)
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'speech_events':len(spoken),'failures':len(failures)}),flush=True)
        return
    cap=cv2.VideoCapture(str(args.source));fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps<=0:raise RuntimeError('Cannot decode source video')
    source_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sw,sh=int(cap.get(3)),int(cap.get(4));w,h=1280,round(sh*1280/sw)//2*2
    step=max(1,round(fps/args.inference_fps));ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    pipe=Pipeline();cv2.setNumThreads(2)
    encoder=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','23','-threads','2',
        '-pix_fmt','yuv420p',str(out/'silent.mp4')],stdin=subprocess.PIPE)
    times=[];events=[];last_spoken={};counts={'known':0,'unknown':0};index=0;started=time.perf_counter()
    overlay=np.zeros((h,w,3),np.uint8);mask=np.zeros((h,w),np.uint8)
    try:
        with (out/'predictions.jsonl').open('w',encoding='utf-8') as log:
            while True:
                ok,frame=cap.read()
                if not ok:break
                ts=index/fps
                if args.seconds and ts>=args.seconds:break
                if index%step==0:
                    result=pipe.run(frame,10.7769,106.7009,False,road_bottom=.65,annotate=False,stream_id='product-render',frame_timestamp=ts)
                    times.append(result.timings_ms['total'])
                    signs=result.perception.signs
                    counts['known']+=sum(s.label!='unknown' for s in signs)
                    counts['unknown']+=sum(s.label=='unknown' for s in signs)
                    log.write(json.dumps({'timestamp':ts,'signs':[asdict(s) for s in signs],'speech':result.reasoning.speech_text,'timings_ms':result.timings_ms},ensure_ascii=False)+'\n')
                    text=result.reasoning.speech_text
                    if text and ts-last_spoken.get(text,-100)>20:
                        events.append({'timestamp':ts,'text':text});last_spoken[text]=ts
                    overlay[:]=0
                    for points in result.perception.lane.polylines:
                        pts=np.array([(round(x*w/sw),round(y*h/sh)) for x,y in points],np.int32)
                        cv2.polylines(overlay,[pts],False,(70,200,70),2)
                    labels=[]
                    for sign in signs:
                        x1,y1,x2,y2=[round(v*(w/sw if i%2==0 else h/sh)) for i,v in enumerate(sign.bbox_xyxy)]
                        color=(150,150,150) if sign.applicability=='ignored' else (0,200,255)
                        cv2.rectangle(overlay,(x1,y1),(x2,y2),color,2)
                        label=display_name(sign.label) if sign.label!='unknown' else 'Đang xác minh biển'
                        labels.append((x1,max(70,y1-23),label,color[::-1]))
                    pil=Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(pil)
                    draw.rectangle((0,0,w,62),fill=(20,20,20))
                    draw.text((10,4),f'{ts/60:.2f} phút | Video {fps:g} FPS | Nhận diện {fps/step:g} Hz | Ô tô',font=font,fill='white')
                    draw.text((10,30),'Tên biển đã xác nhận qua nhiều frame; điều kiện làn/biển phụ chưa đủ bằng chứng.',font=font,fill='white')
                    for x,y,label,color in labels:
                        x=max(0,min(x,w-int(draw.textlength(label,font=font))-8))
                        box=draw.textbbox((x,y),label,font=font)
                        draw.rectangle((box[0]-2,box[1]-2,box[2]+2,box[3]+2),fill=(20,20,20))
                        draw.text((x,y),label,font=font,fill=color)
                    overlay=cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR);mask=np.any(overlay!=0,axis=2).astype(np.uint8)*255
                rendered=cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA)
                cv2.copyTo(overlay,mask,rendered);encoder.stdin.write(rendered.tobytes())
                if index in (189,600,1800):cv2.imwrite(str(out/f'preview_{index}.jpg'),rendered)
                index+=1
                if index%900==0:
                    progress={'frames':index,'video_seconds':index/fps,'elapsed_seconds':round(time.perf_counter()-started,1)}
                    (out/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
    finally:
        cap.release();encoder.stdin.close();code=encoder.wait(timeout=60)
    if code:raise RuntimeError('Video encoder failed')
    spoken,failures=mix_speech(events,pipe.tts_engine,out,index/fps)
    report={'source':str(args.source),'source_frames':source_frames,'output_frames':index,'output_fps':fps,'inference_hz':fps/step,
        'complete':index==source_frames,'analysis_size':[sw,sh],'counts':counts,'speech_events':spoken,'tts_failures':failures,
        'processing_ms':{'mean':float(np.mean(times[10:])),'p95':float(np.percentile(times[10:],95))},
        'limits':['Development video; no human-verified ground truth.','30 primary sign classes only; supplementary semantics and overhead mounting not automatically verified.',
                  'Uncertain applicability is announced as an observation, not an instruction.','Original soundtrack replaced by Edge TTS. Offline export TTS is aligned to detection time, not measured live latency.'],
        'wall_seconds':time.perf_counter()-started}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('speech_events','tts_failures')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
