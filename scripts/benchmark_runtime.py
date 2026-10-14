"""Repeatable warm runtime and dashboard regression check on a local video."""
import argparse
import json
from pathlib import Path
import sys
import time
import psutil

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.orchestrator import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--output', required=True)
    parser.add_argument('--road-bottom', type=float, default=1.)
    parser.add_argument('--check-dashboard', action='store_true')
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.source)
    frames = []
    for seconds in [0, 10, 24, 120, 240, 360, 600, 840, 960, 1000]:
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds*1000)
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot read {seconds}s')
        frames.append((seconds, cv2.resize(image, (1280, 720))))
    cap.release()
    load_started=time.perf_counter()
    pipeline = Pipeline()
    load_seconds=time.perf_counter()-load_started
    options = {'road_bottom': args.road_bottom, 'annotate': False} if args.road_bottom < 1 else {}
    pipeline.run(frames[0][1], 10.7769, 106.7009, False, **options)
    rows = []
    for repeat in range(3):
        for seconds, frame in frames:
            result = pipeline.run(frame, 10.7769, 106.7009, False, **options)
            rows.append({'repeat': repeat, 'seconds': seconds, **result.timings_ms,
                         'dashboard_boxes': sum(s.bbox_xyxy[1] >= 720*.65 for s in result.perception.signs),
                         'dashboard_lane_points': sum(y >= 720*.65 for line in result.perception.lane.polylines for x,y in line)})
    output = {'rows': rows, 'mean_ms': {key: float(np.mean([r[key] for r in rows])) for key in result.timings_ms},
              'p95_ms': float(np.percentile([r['total'] for r in rows], 95)),
              'cuda_allocated_mb': torch.cuda.memory_allocated()/1024**2 if torch.cuda.is_available() else 0,
              'process_rss_mb':psutil.Process().memory_info().rss/1024**2,'model_load_seconds':load_seconds}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k != 'rows'}, indent=2))
    if args.check_dashboard:
        assert not any(r['dashboard_boxes'] or r['dashboard_lane_points'] for r in rows), 'Dashboard leaks into perception'


if __name__ == '__main__':
    main()
