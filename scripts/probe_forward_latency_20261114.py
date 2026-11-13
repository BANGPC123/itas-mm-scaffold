"""M2 gate (b): per-sample latency of the COCO tracker alone (lane off, same sampling as render: every 10th frame of 60 fps)."""
import json, sys, time
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.perception.forward_objects import CocoTracker, ForwardWarnConfig
from src.utils.config import load_config

def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    cfg = ForwardWarnConfig.from_dict(load_config('perception').get('forward_warning', {}))
    cfg.enabled = True
    tr = CocoTracker(cfg)
    cap = cv2.VideoCapture(str(src))
    times, idx, n_det, timed = [], 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok or len(times) >= 300:
            break
        if idx % 10 == 0:
            t0 = time.perf_counter()
            dets = tr.detect(frame)
            dt = (time.perf_counter() - t0) * 1000
            timed += 1
            if timed > 10:          # first 10 calls are warm-up
                times.append(dt)
            n_det += len(dets)
        idx += 1
    cap.release()
    times.sort()
    rep = {'clip': str(src), 'samples_timed': len(times), 'p50_ms': round(times[len(times)//2], 2),
           'p90_ms': round(times[int(len(times)*0.9)], 2), 'mean_ms': round(sum(times)/len(times), 2),
           'detections_total': n_det, 'weights': cfg.weights_path, 'imgsz': cfg.imgsz, 'conf': cfg.conf}
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rep, ensure_ascii=False))

if __name__ == '__main__':
    main()
