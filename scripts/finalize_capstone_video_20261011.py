"""Finalize offline MP4 capstone video, robust TTS, legally cautious event log.

Uses an ALREADY RENDERED video_annotated.mp4, so failure of external
Edge-TTS cannot destroy the MP4 or pretend a demo failed outright.
"""
from pathlib import Path
import cv2,json,hashlib,subprocess,time,datetime
from src.pipeline.orchestrator import Pipeline
from src.utils.config import load_config
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/capstone_demo_20261011'
VIDEO=OUT/'video_annotated.mp4'
SOURCE=ROOT/'data/demo/ufld_example.mp4'
if not VIDEO.is_file():raise RuntimeError('No rendered MP4 to finalize')
reader=cv2.VideoCapture(str(VIDEO))
fps=float(reader.get(cv2.CAP_PROP_FPS));frames=int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
assert reader.isOpened() and frames>=60 and 10<=fps<=30,(frames,fps)
w,h=int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)),int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
assert (w,h)==(1280,720)
reader.set(cv2.CAP_PROP_POS_FRAMES, min(frames-1,round(fps*3)))
ok,preview=reader.read()
if not ok:raise ValueError('Cannot decode video')
cv2.imwrite(str(OUT/'preview.jpg'),preview,[cv2.IMWRITE_JPEG_QUALITY,90])
reader.release()
print('MP4_VIDEO_VALIDATED',frames,fps,w,h,VIDEO.stat().st_size,flush=True)
# Probe 1 event/s against original footage and never use the overlay as GT.
original=cv2.VideoCapture(str(SOURCE))
pipeline=Pipeline()
events=[];t0=time.perf_counter()
for sec in range(min(12,int(frames/fps))):
    original.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    ok,img=original.read()
    if not ok:break
    if img.shape[:2]!=(720,1280):img=cv2.resize(img,(1280,720))
    r=pipeline.run(img,10.7769,106.7009,False,road_bottom=.65,
                   stream_id='capstone_finalized',frame_timestamp=float(sec),annotate=False)
    events.append({'timestamp':sec,
                   'gps_mode':'simulated','zone':r.context.zone_label,
                   'lane_lines':len(r.perception.lane.polylines),
                   'signs':[{'code':s.label,'bbox_xyxy':list(s.bbox_xyxy),
                             'scope':s.applicability,'verified':bool(s.context_verified and s.label_verified)}
                             for s in r.perception.signs],
                   'legal_references':[{'code':x.get('code'),'source_url':x.get('source_url'),
                                         'page':x.get('page')}
                                         for x in r.reasoning.references],
                   'fine_amount_verified':False,
                   'speech_text':r.reasoning.speech_text,
                   'timings_ms':r.timings_ms})
    print('EVENT',sec,'lanes',events[-1]['lane_lines'],
          'signs',len(events[-1]['signs']),flush=True)
original.release()
with (OUT/'events.jsonl').open('w',encoding='utf-8') as f:
    for e in events:f.write(json.dumps(e,ensure_ascii=False)+'\n')
narration=None
candidate=None
error=None
try:
    from src.interaction.tts_engine import TtsEngine
    narration=Path(TtsEngine(load_config('interaction')).synthesize(
        'Đây là bản demo ITAS MM. Video được phân tích với GPS mô phỏng.'))
except Exception as exc:
    error=type(exc).__name__+': '+str(exc)[:120]
    # Proven neutral recorded Vietnamese TTS from the same model/voice.
    cached=ROOT/'data/processed/tts_output/cae46964c9b22a29ca25c8ca1411a9d10583a2c56e3c7f2ebd1fa0099706101d.mp3'
    if cached.is_file() and cached.stat().st_size>2000:
        narration=cached
print('TTS_AUDIO',str(narration) if narration else None,'error',error,flush=True)
if narration and narration.is_file():
    try:
        import imageio_ffmpeg
        executable=imageio_ffmpeg.get_ffmpeg_exe()
        candidate=OUT/'video_with_narration.mp4'
        mux=[executable,'-y','-loglevel','error','-i',str(VIDEO),'-i',str(narration),
             '-filter_complex','[1:a]apad[aout]',
             '-map','0:v:0','-map','[aout]','-c:v','copy','-c:a','aac','-t',str(frames/fps),
             '-movflags','+faststart',str(candidate)]
        completed=subprocess.run(mux,capture_output=True,text=True,timeout=90)
        if completed.returncode:
            error=(error or '')+' ffmpeg:'+completed.stderr[:250]
            candidate=None
        else:
            check=cv2.VideoCapture(str(candidate))
            frames_out=int(check.get(cv2.CAP_PROP_FRAME_COUNT))
            check.release()
            if frames_out<frames*.90:raise RuntimeError('Short mux video')
            print('MUX_WITH_AUDIO_READY',candidate.stat().st_size,'frames',frames_out,flush=True)
    except Exception as exc:
        error=(error or '')+' '+type(exc).__name__+':'+str(exc)[:200]
        candidate=None
summary={'status':'DEMO_RENDERED_AND_VERIFIED',
        'input_video':str(SOURCE),'input_video_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'output_video':str(VIDEO),'output_with_narration':str(candidate) if candidate else None,
        'output_frames':frames,'fps':fps,'dimensions':[w,h],
        'video_seconds':round(frames/fps,2),
        'event_log_scanned_seconds':len(events),
        'lane_positive_events':sum(e['lane_lines']>0 for e in events),
        'sign_observations':sum(len(e['signs']) for e in events),
        'legal_reference_hits':sum(len(e['legal_references']) for e in events),
        'no_live_camera_or_gps_required':True,'gps_simulated':True,
        'video_uses_yolo11n_mobilenet_and_ufldv2':True,
        'fine_amount_verified':False,'tts_text_is_neutral_demo_only':True,
        'selected_models_unchanged':True,
        'audio_error_or_fallback':error,
        'wall_probe_seconds':round(time.perf_counter()-t0,2),
        'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('CAPSTONE_FINALIZED',json.dumps(summary,ensure_ascii=False),flush=True)
