"""Collect candidate negative crops for the recognizer v2 reference gate (issue #2, WP3).

Sources: harvest YouTube videos (one frame per 0.5 s) and TRAIN images (detector boxes that overlap no labelled box).
VAL, TEST and the development video V0 are never read here. Writes only under output/. The review sheets show
index numbers only: no model prediction, source or time (blind review, labels are AI-reviewed).
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOC = ROOT / "data/processed/canonical/itas_mm_v1/localization"
SEMANTIC = ROOT / "data/audit/semantic/final_v1_2/semantic_labels_v1_2.csv"
SAMPLES = ROOT / "data/manifests/dataset_v2/samples.csv"
SEED = 20261029
SEGMENT_S = 60


def iou(a, b):
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def overlaps_labeled(box, labeled, iou_limit=0.1, cover_limit=0.3):
    """True if the detector box touches any labelled box: IoU >= 0.1 or 30% of either box covered."""
    area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
    for other in labeled:
        inter = max(0.0, min(box[2], other[2]) - max(box[0], other[0])) * max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
        if not inter:
            continue
        other_area = max(1e-9, (other[2] - other[0]) * (other[3] - other[1]))
        if iou(box, other) >= iou_limit or inter / area >= cover_limit or inter / other_area >= cover_limit:
            return True
    return False


def segment_group(video_id, timestamp, segment_s=SEGMENT_S):
    return f"{video_id}@{int(timestamp // segment_s) * segment_s}"


def temporal_thin(candidates, min_gap_s=1.0, iou_limit=0.3):
    """Drop a video candidate that repeats a kept box of the same video within `min_gap_s` (same object, next frame)."""
    kept = []
    for candidate in sorted(candidates, key=lambda c: (c["video_id"], c["t"])):
        recent = [k for k in kept[-40:] if k["video_id"] == candidate["video_id"] and candidate["t"] - k["t"] < min_gap_s]
        if not any(iou(k["bbox_xyxy"], candidate["bbox_xyxy"]) >= iou_limit for k in recent):
            kept.append(candidate)
    return kept


def balanced_sample(pools, total, seed=SEED):
    """Equal quota per pool; a pool that cannot fill its quota hands the remainder to the others."""
    rng = random.Random(seed)
    names = sorted(pools)
    shuffled = {name: rng.sample(pools[name], len(pools[name])) for name in names}
    quota = {name: total // len(names) for name in names}
    for name in names[: total - sum(quota.values())]:
        quota[name] += 1
    taken = {name: min(quota[name], len(shuffled[name])) for name in names}
    spare = total - sum(taken.values())
    while spare > 0:
        grew = False
        for name in names:
            if spare > 0 and taken[name] < len(shuffled[name]):
                taken[name] += 1
                spare -= 1
                grew = True
        if not grew:
            break
    return [item for name in names for item in shuffled[name][: taken[name]]]


def group_split(groups, heldout_fraction=0.2, seed=SEED):
    """Group ids kept out for margin selection (at least one when there are two or more groups)."""
    ordered = sorted(set(groups))
    random.Random(seed).shuffle(ordered)
    count = int(round(len(ordered) * heldout_fraction))
    if len(ordered) >= 2:
        count = max(1, count)
    return set(ordered[:count])


def detector():
    from src.perception.sign_detector import SignDetector
    from src.utils.config import load_config
    cfg = load_config("perception")
    return SignDetector(cfg), cfg["sign_detector"]


def recognizer_raw(rec, crops, batch=128):
    """Raw argmax label and confidence of recognizer v2 (no gate, no threshold) for BGR crops."""
    import torch
    from src.perception.sign_recognizer import crop_tensor
    out = []
    for start in range(0, len(crops), batch):
        x = torch.stack([crop_tensor(c, rec.size) for c in crops[start:start + batch]]).to(rec.device)
        with torch.inference_mode():
            prob = rec.model(x).softmax(1)
        conf, idx = prob.max(1)
        out += [(rec.classes[i], float(c)) for c, i in zip(conf.cpu().tolist(), idx.cpu().tolist())]
    return out


def load_recognizer():
    from src.perception.sign_recognizer import SignRecognizer
    from src.utils.config import load_config
    cfg = load_config("perception")
    cfg["sign_recognizer"].pop("reference_gate_path", None)
    return SignRecognizer(cfg)


def crop_box(image, bbox, margin=0.0):
    import math
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    a, b = max(0, math.floor(x1 - margin * bw)), max(0, math.floor(y1 - margin * bh))
    c, d = min(w, math.ceil(x2 + margin * bw)), min(h, math.ceil(y2 + margin * bh))
    return image[b:d, a:c]


def scan_video(args):
    import cv2
    det, _ = detector()
    rec = load_recognizer()
    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps * args.step))
    rows = []
    for frame_no in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, image = cap.read()
        if not ok:
            continue
        found = [d for d in det.detect(image) if d.confidence >= args.min_conf]
        crops = [crop_box(image, d.bbox_xyxy) for d in found]
        valid = [(d, c) for d, c in zip(found, crops) if c.size and min(c.shape[:2]) >= 8]
        if not valid:
            continue
        preds = recognizer_raw(rec, [c for _, c in valid])
        t = frame_no / fps
        for (d, c), (label, conf) in zip(valid, preds):
            rows.append({"source": "video", "video_id": args.video_id, "t": round(t, 3), "frame": frame_no,
                         "bbox_xyxy": [round(v, 1) for v in d.bbox_xyxy], "det_conf": round(d.confidence, 4),
                         "short_side": min(c.shape[:2]), "pred": label, "pred_conf": round(conf, 4),
                         "group": segment_group(args.video_id, t)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"video": args.video_id, "frames": total // step, "candidates": len(rows)}))


def scan_train(args):
    import cv2
    import pandas as pd
    det, _ = detector()
    rec = load_recognizer()
    labels = pd.read_csv(SEMANTIC, keep_default_na=False, usecols=["image_path", "split", "bbox_xyxy"])
    labels = labels[labels.split == "TRAIN"]
    boxes = {}
    for row in labels.itertuples():
        boxes.setdefault(row.image_path, []).append(json.loads(row.bbox_xyxy))
    train = pd.read_csv(SAMPLES, keep_default_na=False)
    paths = sorted(train[train.split == "TRAIN"].image_relpath)
    rows = []
    for index, path in enumerate(paths):
        image = cv2.imread(str(LOC / path))
        if image is None:
            continue
        found = [d for d in det.detect(image) if d.confidence >= args.min_conf
                 and not overlaps_labeled(d.bbox_xyxy, boxes.get(path, []))]
        crops = [crop_box(image, d.bbox_xyxy) for d in found]
        valid = [(d, c) for d, c in zip(found, crops) if c.size and min(c.shape[:2]) >= 8]
        if valid:
            preds = recognizer_raw(rec, [c for _, c in valid])
            for (d, c), (label, conf) in zip(valid, preds):
                rows.append({"source": "train", "image_path": path, "bbox_xyxy": [round(v, 1) for v in d.bbox_xyxy],
                             "det_conf": round(d.confidence, 4), "short_side": min(c.shape[:2]), "pred": label,
                             "pred_conf": round(conf, 4), "group": path})
        if index % 1000 == 0:
            print(index, len(paths), len(rows), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"train_images": len(paths), "candidates": len(rows)}))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_review(args):
    """Sample candidates (seeded, balanced video/TRAIN), cut crops and write blind numbered contact sheets."""
    import cv2
    import numpy as np
    video_rows = []
    for path in args.video_candidates:
        video_rows += read_jsonl(path)
    thinned = temporal_thin(video_rows)
    train_rows = read_jsonl(args.train_candidates)
    pools = {"train": train_rows, "video": thinned}
    chosen = balanced_sample(pools, args.max_crops, SEED)
    random.Random(SEED + 1).shuffle(chosen)
    out = Path(args.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    caps = {}
    items = []
    for number, row in enumerate(chosen, 1):
        if row["source"] == "video":
            video = Path(args.video_root) / f"{row['video_id']}.mp4"
            cap = caps.setdefault(row["video_id"], cv2.VideoCapture(str(video)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
            ok, image = cap.read()
        else:
            image = cv2.imread(str(LOC / row["image_path"]))
            ok = image is not None
        if not ok:
            continue
        crop = crop_box(image, row["bbox_xyxy"], margin=0.15)
        if crop.size == 0:
            continue
        cv2.imwrite(str(out / "crops" / f"{number:04d}.png"), crop)
        items.append({"number": number, **row})
    tile, cols, rows_per = 150, 8, 6
    per_sheet = cols * rows_per
    for start in range(0, len(items), per_sheet):
        sheet = np.full((rows_per * tile, cols * tile, 3), 28, np.uint8)
        for position, item in enumerate(items[start:start + per_sheet]):
            crop = cv2.imread(str(out / "crops" / f"{item['number']:04d}.png"))
            scale = (tile - 18) / max(crop.shape[:2])
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))), interpolation=interp)
            y0, x0 = (position // cols) * tile, (position % cols) * tile
            sheet[y0 + 16:y0 + 16 + crop.shape[0], x0 + 2:x0 + 2 + crop.shape[1]] = crop
            cv2.putText(sheet, str(item["number"]), (x0 + 3, y0 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imwrite(str(out / f"sheet_{start // per_sheet + 1:03d}.png"), sheet)
    (out / "private_mapping.json").write_text(json.dumps({"seed": SEED, "items": items, "thinned_video": len(thinned),
        "train_candidates": len(train_rows), "video_candidates_raw": len(video_rows)}, indent=1), encoding="utf-8")
    print(json.dumps({"sampled": len(items), "video": sum(i["source"] == "video" for i in items),
                      "train": sum(i["source"] == "train" for i in items), "sheets": -(-len(items) // per_sheet)}))


def assign_splits(records, heldout_fraction=0.2, seed=SEED):
    """Mark reviewed not_a_sign records 'fit' or 'heldout' by group (video + 60 s segment, or TRAIN image), per source."""
    for source in ("video", "train"):
        groups = [r["group"] for r in records if r["source"] == source and r["label"] == "not_a_sign"]
        heldout = group_split(groups, heldout_fraction, seed)
        for r in records:
            if r["source"] == source and r["label"] == "not_a_sign":
                r["split"] = "heldout" if r["group"] in heldout else "fit"
    for r in records:
        r.setdefault("split", "unused")
    return records


def manifest_record(item, label):
    """One manifest line: source reference and box only, never image bytes."""
    record = {"id": item["number"], "source": item["source"], "bbox_xyxy": item["bbox_xyxy"], "label": label,
              "group": item["group"], "provenance": "ai_reviewed", "det_conf": item["det_conf"]}
    if item["source"] == "video":
        record.update(video_id=item["video_id"], t=item["t"], frame=item["frame"])
    else:
        record.update(image_path=item["image_path"], image_sha256=Path(item["image_path"]).stem)
    return record


def finalize(args):
    mapping = json.loads((Path(args.review) / "private_mapping.json").read_text(encoding="utf-8"))
    labels = json.loads((Path(args.review) / "blind_labels.json").read_text(encoding="utf-8"))["labels"]
    records = assign_splits([manifest_record(i, labels[str(i["number"])]) for i in mapping["items"]])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    summary = {}
    for r in records:
        key = f"{r['source']}:{r['label'] if not r['label'].startswith('qcvn_sign:') else 'qcvn_sign'}"
        summary[key] = summary.get(key, 0) + 1
    negatives = [r for r in records if r["label"] == "not_a_sign"]
    summary["not_a_sign_fit"] = sum(r["split"] == "fit" for r in negatives)
    summary["not_a_sign_heldout"] = sum(r["split"] == "heldout" for r in negatives)
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--review", required=True)
    f.add_argument("--out", required=True)
    v = sub.add_parser("scan-video")
    v.add_argument("--video", required=True)
    v.add_argument("--video-id", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--step", type=float, default=0.5)
    v.add_argument("--min-conf", type=float, default=0.5)
    t = sub.add_parser("scan-train")
    t.add_argument("--out", required=True)
    t.add_argument("--min-conf", type=float, default=0.5)
    r = sub.add_parser("build-review")
    r.add_argument("--video-candidates", nargs="+", required=True)
    r.add_argument("--train-candidates", required=True)
    r.add_argument("--video-root", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-crops", type=int, default=1500)
    args = parser.parse_args()
    {"scan-video": scan_video, "scan-train": scan_train, "build-review": build_review, "finalize": finalize}[args.command](args)


if __name__ == "__main__":
    main()
