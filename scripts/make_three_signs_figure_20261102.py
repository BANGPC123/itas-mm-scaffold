"""Before/after figure for the three-sign overlap in YQl9c5gCjhg (CP2 item C).

Draws the frame at source second T from the original video, with the B0 (old code, 0.87)
and N64 (new code, T2) boxes from the field-evaluation runs. Writes a PNG for the final report.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "output/kiet_wp_20261029/videos/YQl9c5gCjhg.mp4"
RUNS = {
    "B0 (old code, 0.87)": ROOT / "output/kiet_wp_20261029/field_eval/B_YQl9c5gCjhg_30/segment_30/predictions.jsonl",
    "N64 (new code, T2)": ROOT / "output/kiet_wp_20261029/field_eval/N64_YQl9c5gCjhg_30/segment_30/predictions.jsonl",
}
COLORS = {"B0 (old code, 0.87)": (60, 60, 230), "N64 (new code, T2)": (60, 200, 60)}


def nearest_row(path, t, tolerance=0.2):
    best = None
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if abs(row["timestamp"] - t) <= tolerance and (best is None or abs(row["timestamp"] - t) < abs(best["timestamp"] - t)):
            best = row
    if best is None:
        raise SystemExit(f"no prediction within {tolerance}s of {t}s in {path}")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=float, default=33.0)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/final_report/figures/fig_three_signs_b0_vs_n64.png")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(VIDEO))
    capture.set(cv2.CAP_PROP_POS_MSEC, args.t * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit("could not read frame")

    canvas = frame.copy()
    legend = []
    for name, path in RUNS.items():
        row = nearest_row(path, args.t)
        color = COLORS[name]
        coded = [s for s in row["signs"] if s["label"] != "unknown"]
        for s in row["signs"]:
            x1, y1, x2, y2 = (int(v) for v in s["bbox_xyxy"])
            if s["label"] == "unknown":
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), 1)
                continue
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
            cv2.putText(canvas, s["label"], (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        legend.append(f"{name}: {len(coded)} coded box(es) at t={row['timestamp']:.2f}s")

    banner_h = 70
    out = cv2.copyMakeBorder(canvas, banner_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for i, text in enumerate(legend):
        cv2.putText(out, text, (14, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[list(RUNS)[i]], 2, cv2.LINE_AA)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), out)
    for text in legend:
        print(text)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
