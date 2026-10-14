"""Verify reproducible evidence of a full development-video replay."""
from pathlib import Path
import json
import cv2

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'output/video_tests/test_video_20261010_after_hardening'
summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
video=out/'test_video_annotated.mp4'
cap=cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise RuntimeError('Unable to open annotated video')
n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps=float(cap.get(cv2.CAP_PROP_FPS))
sampled=int(summary['sampled_frames'])
checks={}
for name,k in [('first',0),('middle',sampled//2),('last',sampled-1)]:
    cap.set(cv2.CAP_PROP_POS_FRAMES,k)
    ok,frame=cap.read()
    checks[name]={'frame':k,'decode_success':bool(ok and frame is not None and frame.size>0)}
cap.release()
pred=sum(1 for line in (out/'predictions.jsonl').open(encoding='utf-8') if line.strip())
truth={'complete_source_scan':summary['complete_source_scan'],
       'expected_output_frames':sampled,'encoded_frames':n,
       'prediction_lines':pred,'video_bytes':video.stat().st_size,
       'fps':fps,'sampled_fps_expected':summary['sampled_fps'],'samples_decode':checks}
truth['pass']=bool(truth['complete_source_scan'] and sampled==n==pred and video.stat().st_size>10000 and abs(fps-summary['sampled_fps'])<0.01 and all(v['decode_success'] for v in checks.values()))
(out/'verify_output.json').write_text(json.dumps(truth,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(truth,ensure_ascii=False,indent=2),flush=True)
if not truth['pass']:raise SystemExit(1)
