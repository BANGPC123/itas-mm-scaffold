"""M8.2 micro-benchmark: COCO tracker on the same 240 frames of the night clip.
variants: fp32 (current), fp16 (half=True), prescale (resize long side to 640 with INTER_AREA, then track, boxes scaled back)."""
import json, sys, time
from pathlib import Path
import cv2
ROOT = Path('.').resolve(); sys.path.insert(0, str(ROOT))
from src.perception.forward_objects import CocoTracker, ForwardWarnConfig
cap = cv2.VideoCapture('output/final_20261102/demo_clips/night_hanoi_YQl9c5gCjhg_30s_90s.mp4')
frames = []
for i in range(240):
    ok, f = cap.read()
    if not ok: break
    if i % 1 == 0: frames.append(f)
cap.release()
def timed(tracker, frames, prescale=False):
    tracker.reset_job()
    for f in frames[:10]: tracker.detect(f)  # warm-up
    out = []; t = time.perf_counter()
    for f in frames:
        if prescale:
            h, w = f.shape[:2]; s = 640 / max(h, w)
            g = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            dets = tracker.detect(g)
            for d in dets: d['bbox_xyxy'] = [v / s for v in d['bbox_xyxy']]
        else:
            dets = tracker.detect(f)
        out.append(len(dets))
    dt = time.perf_counter() - t
    return {'ms_per_frame': round(1000 * dt / len(frames), 2), 'total_s': round(dt, 2), 'dets_sum': sum(out)}
res = []
cfg = ForwardWarnConfig(enabled=True, half=False)
res.append({'variant': 'fp32', **timed(CocoTracker(cfg), frames)})
cfg16 = ForwardWarnConfig(enabled=True, half=True)
res.append({'variant': 'fp16', **timed(CocoTracker(cfg16), frames)})
res.append({'variant': 'fp32_prescale640', **timed(CocoTracker(cfg), frames, prescale=True)})
res.append({'variant': 'fp32_again', **timed(CocoTracker(cfg), frames)})
Path('output/demo_20261114/M8/detect/bench_detect.json').write_text(json.dumps(res, indent=2), encoding='utf8')
print(json.dumps(res, indent=2))
