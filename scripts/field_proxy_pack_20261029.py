"""Proxy-recall pack (issue #2, WP3 part 4.4): one frame per 5 s of every EVAL segment, detector proposals as crops.

The reviewer reads a QCVN code (or not_sign / unreadable) for each numbered crop without any recognizer output.
Proposals use the pipeline detector at its video setting and threshold, so signs the detector never proposes are not
counted: proxy recall is conditional on detection and is not a standard recall. Labels are AI-reviewed.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STEP_S = 5.0
LENGTH_S = 60
MIN_SIDE_PX = 24
MIN_CONF = 0.35


def proxy_times(start, length_s=LENGTH_S, step_s=STEP_S):
    """Frame times inside the segment, one per `step_s`, centred in each step so the +-2 s window stays in range."""
    return [start + step_s / 2 + k * step_s for k in range(int(length_s // step_s))]


def readable_annotations(items, labels, min_side=MIN_SIDE_PX):
    """Annotated signs that count for proxy recall: a QCVN code label and a short side of at least `min_side` px."""
    kept = []
    for item in items:
        label = labels.get(str(item["number"]))
        x1, y1, x2, y2 = item["bbox_xyxy"]
        if label and label not in ("not_sign", "unreadable") and min(x2 - x1, y2 - y1) >= min_side:
            kept.append({"number": item["number"], "video_id": item["video_id"], "start": item["start"], "t": item["t"],
                         "code": label, "bbox": item["bbox_xyxy"]})
    return kept


def build(args):
    import cv2
    import numpy as np
    from scripts.field_video_eval_20261029 import read_sources
    from scripts.harvest_gate_negatives_20261029 import crop_box, detector
    det, _ = detector()
    out = Path(args.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    items, number = [], 0
    for row in read_sources(args.sources):
        if row["role"] != "eval":
            continue
        cap = cv2.VideoCapture(str(Path(args.video_root) / f"{row['video_id']}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        for start in row["segments_used"]:
            for t in proxy_times(start):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
                ok, image = cap.read()
                if not ok:
                    continue
                for d in sorted(det.detect(image), key=lambda d: d.bbox_xyxy[0]):
                    x1, y1, x2, y2 = d.bbox_xyxy
                    if d.confidence < MIN_CONF or min(x2 - x1, y2 - y1) < MIN_SIDE_PX:
                        continue
                    number += 1
                    items.append({"number": number, "video_id": row["video_id"], "start": start, "t": round(t, 3),
                                  "frame": int(round(t * fps)), "bbox_xyxy": [round(v, 1) for v in d.bbox_xyxy],
                                  "det_conf": round(d.confidence, 3)})
                    cv2.imwrite(str(out / "crops" / f"{number:04d}.png"), crop_box(image, d.bbox_xyxy, margin=0.2))
        cap.release()
    tile, cols, rows_per = 150, 8, 6
    per_sheet = cols * rows_per
    for first in range(0, len(items), per_sheet):
        sheet = np.full((rows_per * tile, cols * tile, 3), 28, np.uint8)
        for pos, item in enumerate(items[first:first + per_sheet]):
            crop = cv2.imread(str(out / "crops" / f"{item['number']:04d}.png"))
            scale = (tile - 18) / max(crop.shape[:2])
            crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
                              interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            y0, x0 = (pos // cols) * tile, (pos % cols) * tile
            sheet[y0 + 16:y0 + 16 + crop.shape[0], x0 + 2:x0 + 2 + crop.shape[1]] = crop
            cv2.putText(sheet, str(item["number"]), (x0 + 3, y0 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imwrite(str(out / f"sheet_{first // per_sheet + 1:03d}.png"), sheet)
    (out / "items.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    print(json.dumps({"proposals": len(items), "sheets": -(-len(items) // per_sheet)}))


def recall_table(items, labels, run_dir_for, configs):
    """Proxy recall per config over every EVAL segment run; run_dir_for(config, video, start) -> predictions path."""
    from scripts.field_video_eval_20261029 import proxy_recall
    annotations = readable_annotations(items, labels)
    result = {}
    for config in configs:
        caught = 0
        details = []
        for key in sorted({(a["video_id"], a["start"]) for a in annotations}):
            frames = [json.loads(line) for line in Path(run_dir_for(config, *key)).read_text(encoding="utf-8").splitlines() if line.strip()]
            group = [a for a in annotations if (a["video_id"], a["start"]) == key]
            res = proxy_recall(group, frames, tolerance_s=2.0)
            caught += res["caught"]
            details += res["details"]
        result[config] = {"annotated": len(annotations), "caught": caught, "recall": caught / len(annotations) if annotations else None,
                          "details": details}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sources", required=True)
    b.add_argument("--video-root", required=True)
    b.add_argument("--out", required=True)
    args = parser.parse_args()
    {"build": build}[args.command](args)


if __name__ == "__main__":
    main()
