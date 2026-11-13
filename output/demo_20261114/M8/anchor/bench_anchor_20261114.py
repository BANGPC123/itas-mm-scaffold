"""M8.0 micro-benchmark: alternating 960/640 on one YOLO instance vs one instance per size. Same frames, same weights."""
import json, sys, time
from pathlib import Path
import cv2
from ultralytics import YOLO
W = 'models/sign_detector/finetuned_20261026.pt'
cap = cv2.VideoCapture('output/final_20261102/demo_clips/night_hanoi_YQl9c5gCjhg_30s_90s.mp4')
frames = []
for i in range(60):
    ok, f = cap.read()
    if not ok: break
    if i % 10 == 0: frames.append(f)
cap.release()
def run(models, label):
    for m, s in [(models[960], 960), (models[640], 640)]:
        m.predict(frames[0], imgsz=s, verbose=False, device='0', conf=0.35, iou=0.45)  # warm-up
    t = time.perf_counter(); n = 0; boxes = []
    for f in frames:
        for s in (960, 640):
            r = models[s].predict(f, imgsz=s, verbose=False, device='0', conf=0.35, iou=0.45)
            boxes.append([round(v, 1) for v in r[0].boxes.xyxy.flatten().tolist()]); n += 1
    dt = time.perf_counter() - t
    return {'label': label, 'calls': n, 'total_s': round(dt, 3), 'ms_per_call': round(1000 * dt / n, 2), 'checksum': sum(map(sum, boxes))}
one = YOLO(W); single = {960: one, 640: one}
two = {960: YOLO(W), 640: YOLO(W)}
res = [run(single, 'one_instance_alternating'), run(two, 'two_instances'), run(single, 'one_instance_alternating_again')]
res.append({'boxes_identical': res[0]['checksum'] == res[1]['checksum']})
Path('output/demo_20261114/M8/anchor/bench_anchor.json').write_text(json.dumps(res, indent=2), encoding='utf8')
print(json.dumps(res, indent=2))
