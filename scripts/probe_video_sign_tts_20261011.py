"""Check two-frame temporal sign confirmation and optional demo narration with real models."""
from pathlib import Path
import json,cv2,time
from src.pipeline.orchestrator import Pipeline
ROOT=Path(__file__).resolve().parents[1]
img=cv2.imread(str(ROOT/'output/review/field_review_pack_20261011/frames/005_0.jpg'))
assert img is not None
p=Pipeline()
for t in (1.,1.22,1.44):
    result=p.run(img,10.7769,106.7009,False,annotate=False,
         road_bottom=.65,stream_id='user_video_voice_probe',frame_timestamp=t,
         lane_visual_roi=False)
    print('ACTUAL_VIDEO_FRAME',t,
        [(s.label,s.recognition_source,round(s.recognition_confidence or 0,3),
          s.name_vi) for s in result.perception.signs],
        'VOICE=',result.reasoning.observation_speech_text,
        'VERIFIED_VOICE=',result.reasoning.speech_text,flush=True)
if result.reasoning.observation_speech_text:
    started=time.perf_counter()
    try:
        path=Path(p.tts_engine.synthesize(result.reasoning.observation_speech_text))
        print('VOICE_MP3_OK',path.is_file(),path.stat().st_size,
              's',round(time.perf_counter()-started,2),flush=True)
    except Exception as exc:
        print('VOICE_NETWORK_ERROR',type(exc).__name__,str(exc)[:120],flush=True)
else:
    print('NO_CONFIRMED_TEMPORAL_SIGN_ON_REPEATED_TEST_FRAME',flush=True)
