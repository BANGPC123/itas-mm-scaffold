"""Render an MP4 in slow, offline, accuracy-first simulation mode.

Processes actual frames in timestamp order; never uses realtime camera.
Provides exhaustive native-FPS video export and independent decode verification.
No human lane GT claim, no automatic legal certainty from 59-class YOLO.
"""
from __future__ import annotations
import argparse,hashlib,json,math,subprocess,sys,time
from dataclasses import asdict
from datetime import datetime,timedelta
from pathlib import Path
import cv2,imageio_ffmpeg,numpy as np
from PIL import ImageFont,Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline.orchestrator import Pipeline
from src.perception.overlay_motion import OverlayMotion
from scripts.complete_test_video import draw_frame


def file_sha256(path):
    hasher=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def render(source:Path,out:Path,*,analysis_hz=6.,start_second=0.,
           max_seconds=None,recorded_at=None,include_auxiliary=False,
           pipeline=None,highres_rescue=False):
    if not source.is_file():raise FileNotFoundError(source)
    if out.exists():raise FileExistsError('Output must be new: '+str(out))
    if not (1.<=analysis_hz<=12.) or start_second<0:
        raise ValueError('analysis_hz must be 1 to 12 and start>=0')
    if max_seconds is not None and not 1.<=max_seconds<=7200:
        raise ValueError('max_seconds must be 1 to 7200')
    if recorded_at is not None and (recorded_at.tzinfo is None or
                                    recorded_at.utcoffset() is None):
        raise ValueError('Video date needs explicit timezone')
    cap=cv2.VideoCapture(str(source))
    fps=float(cap.get(cv2.CAP_PROP_FPS));total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sw,sh=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened() or not 5<=fps<=120 or sw<256 or sh<144:
        cap.release();raise RuntimeError('Unsupported video')
    skip=int(start_second*fps)
    if skip>=total:cap.release();raise ValueError('Start is after end')
    frames=min(total-skip,int(round(max_seconds*fps)) if max_seconds is not None else total-skip)
    step=max(1,round(fps/analysis_hz))
    w=1280 if sw>=1280 else sw
    h=round(sh*w/sw)//2*2
    assert w%2==0 and h%2==0
    out.mkdir(parents=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES,skip)
    pipe=pipeline if pipeline is not None else Pipeline()
    pipeline_id='offline_simulation_'+hashlib.sha256(str(out).encode()).hexdigest()[:18]
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    encoder=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
        '-s',f'{w}x{h}','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast',
        '-crf','21','-threads','2','-pix_fmt','yuv420p',str(out/'silent.mp4')],
        stdin=subprocess.PIPE)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    motion=OverlayMotion();aux_motion=OverlayMotion();candidate_motion=OverlayMotion()
    last_payload=None;lines=[];boxes=[];aux_boxes=[];aux_labels=[];candidate_lines=[]
    counters={'samples':0,'known_signs':0,'uncertain_signs':0,
              'auxiliary_review_boxes':0,'lanes_displayed':0,
              'legal_speech_samples':0,'warnings':0,
              'temporal_ufld_candidate_lines':0}
    started=time.perf_counter();failures=[];index=0
    try:
        with (out/'predictions.jsonl').open('w',encoding='utf8') as jsonl:
            for index in range(frames):
                okay,frame=cap.read()
                if not okay:raise RuntimeError(f'Input decode failed at {skip+index}')
                absolute=(skip+index)/fps
                canvas=cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA)
                if index%step==0:
                    observed=recorded_at+timedelta(seconds=absolute) if recorded_at else None
                    result=pipe.run(frame,10.7769,106.7009,False,
                        road_bottom=.70,annotate=False,stream_id=pipeline_id,
                        frame_timestamp=absolute,observed_at=observed,
                        lane_visual_roi=True,lane_display_mode='precision',
                        offline_video_quality=True,offline_highres_rescue=highres_rescue,
                        auxiliary_sign_review=include_auxiliary)
                    signs=result.perception.signs;lane=result.perception.lane
                    lines=[[(round(x*w/sw),round(y*h/sh)) for x,y in p]
                            for p in (lane.display_polylines or [])]
                    boxes=[[v*(w/sw if k%2==0 else h/sh) for k,v in enumerate(s.bbox_xyxy)]
                           for s in signs]
                    motion.reset(canvas,lines,boxes)
                    candidate_lines=[[(round(x*w/sw),round(y*h/sh)) for x,y in p]
                        for p in lane.candidate_polylines]
                    candidate_motion.reset(canvas,candidate_lines,[])
                    aux_labels=result.perception.auxiliary_sign_candidates
                    aux_boxes=[[v*(w/sw if k%2==0 else h/sh) for k,v in enumerate(e['bbox_xyxy'])]
                        for e in aux_labels]
                    aux_motion.reset(canvas,[],aux_boxes)
                    last_payload={'signs':[asdict(s) for s in signs],
                                  'lane':asdict(lane)}
                    counters['samples']+=1
                    counters['known_signs']+=sum(s.label!='unknown' for s in signs)
                    counters['uncertain_signs']+=sum(s.label=='unknown' for s in signs)
                    counters['auxiliary_review_boxes']+=len(result.perception.auxiliary_sign_candidates)
                    counters['lanes_displayed']+=len(lines)
                    counters['temporal_ufld_candidate_lines']+=len(candidate_lines)
                    counters['legal_speech_samples']+=int(bool(result.reasoning.speech_text))
                    counters['warnings']+=len(result.warnings)
                    jsonl.write(json.dumps({
                        'timestamp':absolute,'source_frame':skip+index,
                        'signs':last_payload['signs'],'lane':last_payload['lane'],
                        'supplementary':result.perception.supplementary_candidates,
                        'auxiliary':result.perception.auxiliary_sign_candidates,
                        'context':result.reasoning.car_summary,
                        'speech':result.reasoning.speech_text,
                        'latency_ms':result.timings_ms,
                    },ensure_ascii=False)+'\n')
                else:
                    lines,boxes=motion.advance(canvas)
                    _,aux_boxes=aux_motion.advance(canvas)
                    candidate_lines,_=candidate_motion.advance(canvas)
                rendered=draw_frame(canvas,last_payload,lines,boxes,
                                    absolute,font,fps,fps/step)
                if candidate_lines:
                    for poly in candidate_lines:
                        if len(poly)>=2:
                            cv2.polylines(rendered,[np.asarray(poly,np.int32)],False,(0,155,255),2)
                    cv2.putText(rendered,'UFLD candidates (temporal, not verified)',
                        (14,min(h-24,84)),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,155,255),1,cv2.LINE_AA)
                if include_auxiliary and aux_labels:
                    # Purple candidates explicitly remain unverified; never
                    # render them as released/legal sign warnings.
                    pil=Image.fromarray(cv2.cvtColor(rendered,cv2.COLOR_BGR2RGB))
                    painter=ImageDraw.Draw(pil)
                    for e,box in zip(aux_labels,aux_boxes):
                        if box is None or len(box)!=4 or min(box)<0:continue
                        x1,y1,x2,y2=[round(v) for v in box]
                        if x1>=x2 or y1>=y2:continue
                        painter.rectangle((x1,y1,x2,y2),outline=(235,100,245),width=2)
                        text=('YOLO11 gợi ý (chưa xác minh): '+e['label_vi'])[:75]
                        tx=max(5,min(x1,w-round(painter.textlength(text,font=font))-8))
                        ty=max(61,y1-22)
                        bounds=painter.textbbox((tx,ty),text,font=font)
                        painter.rectangle((bounds[0]-2,bounds[1]-2,bounds[2]+2,bounds[3]+2),fill=(35,25,50))
                        painter.text((tx,ty),text,font=font,fill=(245,200,255))
                    rendered=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
                encoder.stdin.write(rendered.tobytes())
                if (index+1)%300==0:
                    progress={'encoded_source_frames':index+1,'expected':frames,
                        'samples':counters['samples'],'elapsed_seconds':round(time.perf_counter()-started,1)}
                    (out/'progress.json').write_text(json.dumps(progress))
                    print('OFFLINE_VIDEO_PROGRESS',json.dumps(progress),flush=True)
    except Exception as exc:
        failures.append(str(exc))
        raise
    finally:
        cap.release()
        encoder.stdin.close()
        exitcode=encoder.wait(timeout=120)
        if pipeline is None and getattr(pipe,'supplementary_reader',None) is not None:
            pipe.supplementary_reader.close()
    if exitcode!=0:raise RuntimeError('FFmpeg video encoding failed')
    # Preserve source audio if supplied, aligning original source offset.
    video=out/'test_video_offline_max.mp4'
    mux=subprocess.run([ffmpeg,'-y','-loglevel','error','-i',str(out/'silent.mp4'),
        '-ss',str(start_second),'-t',str(frames/fps),'-i',str(source),
        '-map','0:v:0','-map','1:a:0?','-c:v','copy','-c:a','aac',
        '-b:a','128k',str(video)],capture_output=True,text=True)
    if mux.returncode:
        raise RuntimeError('Audio mux failed: '+mux.stderr[-400:])
    decode=subprocess.run([ffmpeg,'-v','error','-i',str(video),'-map','0:v:0',
        '-progress','pipe:1','-nostats','-f','null','-'],capture_output=True,text=True)
    values=[int(x.split('=',1)[1]) for x in decode.stdout.splitlines() if x.startswith('frame=')]
    decoded=values[-1] if values else 0
    verified=(decode.returncode==0 and not decode.stderr.strip() and decoded==frames)
    scan={'source_file':str(source),'source_sha256':file_sha256(source),
        'start_sec':start_second,'source_frames':total,'output_frames':frames,
        'decoded_frames':decoded,'output_size':[w,h],
        'fps':fps,'analysis_hz':fps/step,'source_frame_interval':step,
        'sign_mode':('960+640+1280' if highres_rescue else '960+640')+', three-frame confirmation',
        'models_finetuned_during_run':False,'highres_1280_enabled':bool(highres_rescue),'input_recording_datetime_known':recorded_at is not None,
        'auxiliary_research_enabled':include_auxiliary,'counts':counters,
        'elapsed_seconds':time.perf_counter()-started,'failures':failures,
        'video_sha256':file_sha256(video),'verified_full_decode':verified,
        'not_a_measured_field_accuracy':True}
    (out/'summary.json').write_text(json.dumps(scan,ensure_ascii=False,indent=2),encoding='utf8')
    print('OFFLINE_VIDEO_RESULT',json.dumps(scan,ensure_ascii=False),flush=True)
    if not verified:raise RuntimeError('Video full-decode check failed')
    return scan


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('source',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--analysis-hz',type=float,default=6.)
    ap.add_argument('--start-second',type=float,default=0.)
    ap.add_argument('--max-seconds',type=float)
    ap.add_argument('--recorded-at',type=str,help='ISO-8601 with timezone')
    ap.add_argument('--auxiliary-review',action='store_true')
    ap.add_argument('--highres-rescue',action='store_true',
        help='Experimental third detector pass; TEST static F1 is slightly worse')
    a=ap.parse_args()
    date=datetime.fromisoformat(a.recorded_at) if a.recorded_at else None
    render(a.source,a.output,analysis_hz=a.analysis_hz,
           start_second=a.start_second,max_seconds=a.max_seconds,
           recorded_at=date,include_auxiliary=a.auxiliary_review,
           highres_rescue=a.highres_rescue)

if __name__=='__main__':main()
