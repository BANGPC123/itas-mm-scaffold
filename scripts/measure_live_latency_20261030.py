"""Measure live-mode latency of Pipeline.run on sampled frames of one demo video.

Frames are sampled every 0.2 s, JPEG round-tripped like /pipeline/run, and the
first 5 frames are discarded as warm-up. Each lane_visual_roi setting runs as its
own temporal stream so state does not carry between configurations.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_VIDEO = ROOT / "output/kiet_wp_20261030/demo/YQl9c5gCjhg_start30_60s/test_video_offline_max.mp4"
DEFAULT_OUTPUT = ROOT / "output/final_20261102/baseline_latency.json"
STEP_S = 0.2
FRAME_COUNT = 120
WARMUP_FRAMES = 5
LATITUDE, LONGITUDE = 21.0285, 105.8542
STAGE_KEYS = ("detection", "recognition", "lane", "reasoning", "tts", "total")


def sample_frames(video_path, start_s=0.0):
    """Return (timestamp_s, frame) pairs taken every STEP_S seconds from start_s, JPEG round-tripped."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    samples = []
    for index in range(FRAME_COUNT):
        timestamp = start_s + index * STEP_S
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = capture.read()
        if not ok:
            break
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("JPEG encode failed")
        decoded = cv2.imdecode(np.frombuffer(encoded.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        samples.append((timestamp, decoded))
    capture.release()
    return samples


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(rows):
    summary = {"frames_measured": len(rows)}
    for key in STAGE_KEYS:
        values = [row["timings"][key] for row in rows if key in row["timings"]]
        if not values:
            continue
        summary[key] = {
            "p50_ms": round(percentile(values, 0.5), 1),
            "p95_ms": round(percentile(values, 0.95), 1),
            "mean_ms": round(statistics.fmean(values), 1),
        }
    return summary


def run_config(pipeline, samples, lane_visual_roi, stream_id):
    rows = []
    for position, (timestamp, frame) in enumerate(samples):
        started = time.perf_counter()
        result = pipeline.run(
            frame,
            LATITUDE,
            LONGITUDE,
            True,
            road_bottom=1.0,
            annotate=True,
            stream_id=stream_id,
            frame_timestamp=timestamp,
            lane_visual_roi=lane_visual_roi,
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        if position < WARMUP_FRAMES:
            continue
        rows.append({"frame_timestamp_s": round(timestamp, 2), "wall_ms": round(wall_ms, 1),
                     "timings": {k: round(v, 1) for k, v in result.timings_ms.items()}})
    return rows


def git_head():
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=float, default=0.0, help="first sampled second of the video")
    args = parser.parse_args()

    from src.pipeline.orchestrator import Pipeline

    samples = sample_frames(args.video, args.start)
    pipeline = Pipeline()
    report = {
        "git_head": git_head(),
        "video": str(args.video.relative_to(ROOT)),
        "start_s": args.start,
        "sample_step_s": STEP_S,
        "frames_sampled": len(samples),
        "warmup_frames_discarded": WARMUP_FRAMES,
        "endpoint_defaults": {"synthesize_audio": True, "annotate": True, "road_bottom": 1.0},
        "configs": {},
    }
    for label, lane_visual_roi in (("lane_visual_roi_true", True), ("lane_visual_roi_false", False)):
        rows = run_config(pipeline, samples, lane_visual_roi, stream_id=f"latency_{label}")
        report["configs"][label] = {"summary": summarize(rows), "frames": rows}
        print(label, json.dumps(report["configs"][label]["summary"], ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
