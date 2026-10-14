"""Reclassify cached detections and repair text overlays without repeating YOLO/UFLD."""
import json
from pathlib import Path
import subprocess
import sys
import time
import cv2
import imageio_ffmpeg
from PIL import ImageFont
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.perception.sign_recognizer import SignRecognizer
from src.perception.sign_catalog import display_name,NAMES_VI
from src.perception.temporal_signs import TemporalSigns
from src.interaction.sign_speech import sign_speech
from src.interaction.tts_engine import TtsEngine
from src.utils.config import load_config
from src.utils.types import SignDetection
from render_product_video import mix_speech
from dataclasses import asdict


def stamp(seconds):
    value=round(seconds*100)
    return f'{value//360000}:{value//6000%60:02}:{value//100%60:02}.{value%100:02}'


def main():
    out=ROOT/'output/product_20261010';summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (out/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    (out/'predictions_before_motion_fix.jsonl').write_bytes((out/'predictions.jsonl').read_bytes())
    (out/'summary_before_motion_fix.json').write_bytes((out/'summary.json').read_bytes())
    torch.set_num_threads(2);cv2.setNumThreads(2)
    rec=SignRecognizer(load_config('perception'));tracker=TemporalSigns()
    cap=cv2.VideoCapture(summary['source']);fps=cap.get(cv2.CAP_PROP_FPS);frame_id=-1
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    ass=['[Script Info]','ScriptType: v4.00+','PlayResX: 1280','PlayResY: 720','WrapStyle: 2',
         '[V4+ Styles]','Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
         'Style: Label,Arial,18,&H0000C8FF,&H0000C8FF,&H00141414,&H00141414,0,0,0,0,100,100,0,0,3,3,0,7,0,0,0,1',
         '[Events]','Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text']
    events=[];spoken_at={};counts={'known':0,'unknown':0};changes=0;started=time.perf_counter()
    with (out/'predictions_repaired.jsonl').open('w',encoding='utf-8') as log:
        for n,row in enumerate(rows):
            ts=row['timestamp'];target=round(ts*fps)
            while frame_id<target:
                if not cap.grab():raise RuntimeError(f'Source ended at {frame_id}')
                frame_id+=1
            ok,image=cap.retrieve()
            if not ok:raise RuntimeError('Cannot read source frame')
            road=image[:int(image.shape[0]*.65)]
            signs=[SignDetection('unknown',s['confidence'],tuple(s['bbox_xyxy'])) for s in row['signs']]
            signs=rec.recognize(road,signs,min_short_side=16)
            # The cached log already excluded confident background detections.
            signs=tracker.update(road,signs,ts)
            for sign in signs:
                sign.name_vi=NAMES_VI.get(sign.label)
            counts['known']+=sum(s.label!='unknown' for s in signs);counts['unknown']+=sum(s.label=='unknown' for s in signs)
            speech=sign_speech(signs)
            if speech and ts-spoken_at.get(speech,-100)>20:
                events.append({'timestamp':ts,'text':speech});spoken_at[speech]=ts
            log.write(json.dumps({**row,'signs':[asdict(s) for s in signs],'speech':speech},ensure_ascii=False)+'\n')
            lookup={tuple(s.bbox_xyxy):s for s in signs}
            for old in row['signs']:
                new=lookup.get(tuple(old['bbox_xyxy']))
                if new is not None and new.label==old['label']:continue
                changes+=1
                x1,y1,_,_=[round(v*2/3) for v in old['bbox_xyxy']]
                old_text=display_name(old['label']) if old['label']!='unknown' else 'Đang xác minh biển'
                label=display_name(new.label) if new is not None and new.label!='unknown' else 'Chưa đọc rõ biển'
                y=max(70,y1-23);old_x=max(0,min(x1,1280-int(font.getlength(old_text))-8))
                bx1,by1,bx2,by2=font.getbbox(old_text)
                left,top,right,bottom=old_x+bx1-3,y+by1-3,old_x+bx2+3,y+by2+3
                start,end=stamp(ts),stamp(min(summary['output_frames']/fps,ts+1/summary['inference_hz']))
                rectangle=f'{{\\an7\\pos(0,0)\\p1\\bord0\\shad0\\c&H141414&}}m {left} {top} l {right} {top} {right} {bottom} {left} {bottom}'
                ass.append(f'Dialogue: 0,{start},{end},Label,,0,0,0,,{rectangle}')
                x=max(3,min(x1,1280-int(font.getlength(label))-8))
                ass.append(f'Dialogue: 1,{start},{end},Label,,0,0,0,,{{\\pos({x},{y+3})}}{label}')
            if n%1000==0:print(json.dumps({'samples':n,'seconds':ts,'elapsed':round(time.perf_counter()-started,1)}),flush=True)
    cap.release()
    (out/'motion_labels.ass').write_text('\n'.join(ass)+'\n',encoding='utf-8-sig')
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg,'-y','-v','error','-i','silent.mp4','-vf','ass=motion_labels.ass','-an','-c:v','h264_nvenc',
        '-preset','p4','-cq','23','-b:v','0','-pix_fmt','yuv420p','silent_repaired.mp4'],cwd=out,check=True)
    (out/'silent.mp4').replace(out/'silent_before_motion_fix.mp4')
    (out/'silent_repaired.mp4').replace(out/'silent.mp4')
    spoken,failures=mix_speech(events,TtsEngine(load_config('interaction')),out,summary['output_frames']/fps)
    (out/'predictions_repaired.jsonl').replace(out/'predictions.jsonl')
    summary.update(counts=counts,speech_events=spoken,tts_failures=failures,motion_matching=True,label_corrections=changes,
                   repair_wall_seconds=time.perf_counter()-started)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'counts':counts,'label_changes':changes,'tts_failures':failures,'speech_events':len(spoken)}),flush=True)


if __name__=='__main__':main()
