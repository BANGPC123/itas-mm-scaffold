"""M13: cache COCO detections and ByteTrack ids for the forward-warning replay.

Makes the same detection calls as scripts/render_offline_video_max_20261014.py::render (same analysis
samples, same detect_every, same tracker), without the sign pipeline. One JSON line per detection call.
Filters are then replayed on this cache (scripts/forward_replay_20261114.py) without a GPU.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.perception.forward_objects import CocoTracker, ForwardWarnConfig  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--analysis-hz', type=float, default=6.)
    a = ap.parse_args()
    if a.out.exists():
        raise FileExistsError(str(a.out))
    cfg = ForwardWarnConfig.from_dict(load_config('perception').get('forward_warning', {}))
    cfg.enabled = True
    cap = cv2.VideoCapture(str(a.source))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / a.analysis_hz))
    tracker = CocoTracker(cfg)
    calls = 0
    with a.out.open('w', encoding='utf8') as fh:
        for index in range(0, total, step):
            # Read analysed frames only; grab() decodes the frames in between like the renderer does.
            while cap.get(cv2.CAP_PROP_POS_FRAMES) < index:
                cap.grab()
            okay, frame = cap.read()
            if not okay:
                break
            sample = index // step
            if sample % cfg.detect_every:
                continue
            dets = tracker.detect(frame)
            calls += 1
            fh.write(json.dumps({'sample': sample, 't': index / fps, 'w': frame.shape[1], 'h': frame.shape[0],
                                 'dets': dets}, ensure_ascii=False) + '\n')
            if calls % 50 == 0:
                print('DETECT_CALLS', calls, flush=True)
    cap.release()
    print('DETECT_DONE', calls, flush=True)


if __name__ == '__main__':
    main()
