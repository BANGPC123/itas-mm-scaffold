"""Blind review pack for confirmed sign labels of field-evaluation runs (issue #2, WP3).

Evidence shared by several configurations (same video, segment, track and best crop) is reviewed once; the
reviewer sees only numbered crops, never a configuration name or a predicted code. Reviewer labels are
AI-reviewed (provenance ai_reviewed), not human ground truth.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN_NAME = re.compile(r"^(?P<config>[A-Z])_(?P<video>[A-Za-z0-9_-]{11})_(?P<start>\d+)$")
SEED = 20261029


def confirmed_items(frames):
    """Last temporally confirmed (track, label) per run with its best crop frame, as in the V0 review pack."""
    by_time = {round(frame["timestamp"], 4): frame for frame in frames}
    last, unresolved = {}, []
    for frame in frames:
        for sign in frame["signs"]:
            if sign.get("recognition_source") == "temporal_best_crop" and sign.get("label") != "unknown":
                last[(sign["track_id"], sign["label"])] = (frame, sign)
    items = []
    for (track, label), (frame, sign) in sorted(last.items()):
        age = sign.get("crop_age_ms")
        if age is None:
            unresolved.append({"track": track, "label": label, "reason": "missing crop age"})
            continue
        best_time = round(frame["timestamp"] - age / 1000, 4)
        best = by_time.get(best_time)
        matches = [s for s in best["signs"] if s.get("track_id") == track] if best else []
        if len(matches) != 1:
            unresolved.append({"track": track, "label": label, "reason": "best frame or track unavailable"})
            continue
        items.append({"track": track, "prediction": label, "best_time": best_time, "source_frame": best["source_frame"],
                      "bbox_xyxy": matches[0]["bbox_xyxy"]})
    return items, unresolved


def evidence_key(item):
    return (item["video_id"], item["start"], item["track"], item["source_frame"], tuple(round(v) for v in item["bbox_xyxy"]))


def key_string(record):
    """Stable text key of one piece of evidence, shared by every configuration that produced it."""
    x1, y1, x2, y2 = (round(v) for v in record["bbox_xyxy"])
    return f"{record['video_id']}|{record['start']}|{record['track']}|{record['source_frame']}|{x1},{y1},{x2},{y2}"


def grade_members(records, store):
    """One graded row per (evidence, configuration) from the blind-label store; unlabeled evidence is an error."""
    from scripts.field_video_eval_20261029 import verdict
    rows = []
    for record in records:
        key = key_string(record)
        if key not in store:
            raise KeyError(f"No blind label for evidence {key}")
        for member in record["members"]:
            rows.append({"config": member["config"], "video_id": record["video_id"], "start": record["start"],
                         "track": record["track"], "best_time": record["best_time"], "prediction": member["prediction"],
                         "blind_label": store[key], "verdict": verdict(member["prediction"], store[key])})
    return rows


def merge_evidence(runs):
    """Merge per-run items into evidence records: {evidence fields, members: [{config, prediction}]}."""
    merged = {}
    for item in runs:
        record = merged.setdefault(evidence_key(item), {k: item[k] for k in ("video_id", "start", "track", "best_time",
                                                                              "source_frame", "bbox_xyxy")} | {"members": []})
        member = {"config": item["config"], "prediction": item["prediction"]}
        if member not in record["members"]:
            record["members"].append(member)
    return list(merged.values())


def collect_runs(field_eval_dir, configs=None):
    items, unresolved, runs = [], [], []
    for folder in sorted(Path(field_eval_dir).iterdir()):
        match = RUN_NAME.match(folder.name)
        if not match or not folder.is_dir() or not (folder / "temporal_benchmark.json").is_file():
            continue
        if configs and match["config"] not in configs:
            continue
        start = int(match["start"])
        frames = [json.loads(line) for line in (folder / f"segment_{start}/predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        found, lost = confirmed_items(frames)
        runs.append(folder.name)
        for item in found:
            items.append({**item, "config": match["config"], "video_id": match["video"], "start": start})
        unresolved += [{**u, "run": folder.name} for u in lost]
    return items, unresolved, runs


def build_pack(args):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    runs, unresolved, names = collect_runs(args.field_eval, args.configs)
    records = merge_evidence(runs)
    known = json.loads(Path(args.known).read_text(encoding="utf-8")) if args.known else {}
    records = [r for r in records if key_string(r) not in known]          # already reviewed: not shown again
    records.sort(key=key_string)
    random.Random(SEED + len(known)).shuffle(records)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    (out / "crops").mkdir(parents=True)
    caps, thumbs = {}, []
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    for number, record in enumerate(records, 1):
        cap = caps.setdefault(record["video_id"], cv2.VideoCapture(str(Path(args.video_root) / f"{record['video_id']}.mp4")))
        cap.set(cv2.CAP_PROP_POS_FRAMES, record["source_frame"])
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Frame decode failed for item {number}")
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = record["bbox_xyxy"]
        crop = frame[max(0, math.floor(y1)):min(h, math.ceil(y2)), max(0, math.floor(x1)):min(w, math.ceil(x2))]
        cv2.imwrite(str(out / "crops" / f"{number:04d}.png"), crop)
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        scale = min(205 / image.width, 170 / image.height)
        thumbs.append(image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.NEAREST))
        record["number"] = number
    for start in range(0, len(thumbs), 20):
        sheet = Image.new("RGB", (1100, 800), (30, 30, 30))
        draw = ImageDraw.Draw(sheet)
        for position, image in enumerate(thumbs[start:start + 20]):
            x, y = (position % 5) * 220, (position // 5) * 200
            draw.text((x + 5, y + 1), str(start + position + 1), font=font, fill="white")
            sheet.paste(image, (x + (220 - image.width) // 2, y + 25 + (170 - image.height) // 2))
        sheet.save(out / f"sheet_{start // 20 + 1:03d}.png")
    (out / "private_mapping.json").write_text(json.dumps({"seed": SEED, "runs": names, "items": records, "unresolved": unresolved}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(names), "evidence_items": len(records), "member_labels": sum(len(r["members"]) for r in records), "unresolved": len(unresolved)}))


def merge_labels(args):
    """Fold a reviewed pack into the evidence-keyed blind-label store."""
    mapping = json.loads((Path(args.pack) / "private_mapping.json").read_text(encoding="utf-8"))
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))["labels"]
    store_path = Path(args.store)
    store = json.loads(store_path.read_text(encoding="utf-8")) if store_path.is_file() else {}
    added = 0
    for record in mapping["items"]:
        label = labels[str(record["number"])]
        key = key_string(record)
        if key in store and store[key] != label:
            raise ValueError(f"Conflicting labels for {key}")
        added += key not in store
        store[key] = label
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(store, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"added": added, "store_size": len(store)}))


def track_rows(run_dir, start, track):
    rows = []
    for line in (Path(run_dir) / f"segment_{start}/predictions.jsonl").read_text(encoding="utf-8").splitlines():
        frame = json.loads(line)
        for sign in frame["signs"]:
            if sign.get("track_id") == track:
                x1, y1, x2, y2 = sign["bbox_xyxy"]
                rows.append((min(x2 - x1, y2 - y1), frame["timestamp"], frame["source_frame"], sign["bbox_xyxy"]))
    return rows


def pick_context_frames(rows, best_time, count=5, near_s=3.0, spacing_s=0.45):
    """Largest track frames within `near_s` of the best crop, at least `spacing_s` apart, in time order."""
    chosen = []
    for row in sorted(rows, key=lambda r: -r[0]):
        if abs(row[1] - best_time) < 0.3 or abs(row[1] - best_time) > near_s:
            continue
        if all(abs(row[1] - c[1]) >= spacing_s for c in chosen):
            chosen.append(row)
        if len(chosen) >= count:
            break
    return sorted(chosen, key=lambda r: r[1])


def build_montage(args):
    """Per-track context strips for ambiguous items (pixels, sizes and times only; no predictions)."""
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    mapping = json.loads((Path(args.pack) / "private_mapping.json").read_text(encoding="utf-8"))
    by_number = {r["number"]: r for r in mapping["items"]}
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    tile = 230
    strips = []
    caps = {}

    def grab(video_id, frame_no, bbox, margin=0.25):
        cap = caps.setdefault(video_id, cv2.VideoCapture(str(Path(args.video_root) / f"{video_id}.mp4")))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, img = cap.read()
        if not ok:
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        a, b, c, d = int(max(0, cx - side / 2)), int(max(0, cy - side / 2)), int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        crop = img[b:d, a:c].copy()
        cv2.rectangle(crop, (int(x1 - a), int(y1 - b)), (int(x2 - a), int(y2 - b)), (0, 255, 0), 1)
        im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        s = tile / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile, tile), (30, 30, 30))
        canvas.paste(im, ((tile - im.width) // 2, (tile - im.height) // 2))
        return canvas

    for number in args.numbers:
        record = by_number[number]
        run_dir = next(Path(args.field_eval).glob(f"?_{record['video_id']}_{record['start']}"))
        rows = track_rows(run_dir, record["start"], record["track"])
        x1, y1, x2, y2 = record["bbox_xyxy"]
        tiles = [grab(record["video_id"], record["source_frame"], record["bbox_xyxy"])]
        labels = [f"best t={record['best_time']:.2f} {x2 - x1:.0f}x{y2 - y1:.0f}"]
        for size, t, frame_no, bbox in pick_context_frames(rows, record["best_time"]):
            tiles.append(grab(record["video_id"], frame_no, bbox))
            labels.append(f"t={t:.2f} {bbox[2] - bbox[0]:.0f}x{bbox[3] - bbox[1]:.0f}")
        strip = Image.new("RGB", (tile * 6, tile + 26), (20, 20, 20))
        draw = ImageDraw.Draw(strip)
        draw.text((4, 2), f"#{number}  track frames={len(rows)}", font=font, fill="yellow")
        for i, (im, label) in enumerate(zip(tiles, labels)):
            if im is not None:
                strip.paste(im, (i * tile, 26))
            draw.text((i * tile + 4, 28), label, font=font, fill="white")
        strips.append(strip)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(strips), 4):
        chunk = strips[i:i + 4]
        sheet = Image.new("RGB", (tile * 6, (tile + 26) * len(chunk)), (0, 0, 0))
        for j, s in enumerate(chunk):
            sheet.paste(s, (0, j * (tile + 26)))
        name = f"montage_{args.numbers[i]:04d}.jpg"
        sheet.save(out / name, quality=90)
        print(name, args.numbers[i:i + 4])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--field-eval", required=True)
    b.add_argument("--video-root", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--configs", nargs="*")
    b.add_argument("--known", help="blind-label store; evidence already in it is left out of the pack")
    g = sub.add_parser("merge")
    g.add_argument("--pack", required=True)
    g.add_argument("--labels", required=True)
    g.add_argument("--store", required=True)
    m = sub.add_parser("montage")
    m.add_argument("--pack", required=True)
    m.add_argument("--field-eval", required=True)
    m.add_argument("--video-root", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--numbers", type=int, nargs="+", required=True)
    args = parser.parse_args()
    {"build": build_pack, "montage": build_montage, "merge": merge_labels}[args.command](args)


if __name__ == "__main__":
    main()
