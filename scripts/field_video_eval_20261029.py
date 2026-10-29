"""Field-video evaluation helpers (issue #2, WP3): sources table, segment rule, blind-review tallies, proxy recall.

Pure functions only; the heavy steps (download, detection, gate building, benchmark runs) live in the
other scripts. Labels are AI-reviewed (provenance ai_reviewed), never human ground truth.
"""
from __future__ import annotations
import csv
import math
import random
from pathlib import Path

VERDICTS = ("correct", "wrong_code", "not_a_sign", "uncertain")
ROLES = ("harvest", "eval", "rejected")
SOURCE_FIELDS = ["video_id", "url", "channel", "upload_date", "duration_s", "resolution", "fps", "sha256", "role",
                 "city", "day_night", "verdict_a_view", "verdict_b_speed", "verdict_c_resolution",
                 "verdict_d_overlay", "verdict_e_signs", "verdict_f_duration", "verdict_notes",
                 "near_duplicate_rate", "segments_used"]
NUMERIC = ("duration_s", "fps", "near_duplicate_rate")


def read_sources(path):
    """Read sources.csv: numbers typed, segments_used as a list of start seconds, role validated."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row = dict(raw)
            if row["role"] not in ROLES:
                raise ValueError(f"Unknown role {row['role']!r} for {row['video_id']}")
            for key in NUMERIC:
                row[key] = float(row[key]) if row.get(key) not in (None, "") else None
            row["segments_used"] = [int(x) for x in row["segments_used"].split(";") if x.strip()]
            rows.append(row)
    return rows


def select_segments(duration_s, count=5, length_s=60, trim_s=30):
    """Start seconds of `count` evenly spaced `length_s` segments inside [trim_s, duration_s - trim_s]."""
    usable = duration_s - 2 * trim_s
    if usable < length_s:
        raise ValueError("Video too short for the segment rule")
    step = (usable - length_s) / (count - 1)
    return [int(math.floor(trim_s + index * step)) for index in range(count)]


CRITERIA = (("verdict_a_view", "a_view"), ("verdict_b_speed", "b_speed"), ("verdict_c_resolution", "c_resolution"),
            ("verdict_d_overlay", "d_overlay"), ("verdict_e_signs", "e_signs"), ("verdict_f_duration", "f_duration"))


def measure_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": round(fps, 3), "frames": frames, "duration_s": round(frames / fps, 1)}
    finally:
        cap.release()


def build_source_row(meta, screening, sha256, dedup, measured):
    """One sources.csv row from yt-dlp metadata, the screening verdicts, file hash, pHash result and measured media facts."""
    role = screening["role"]
    row = {field: "" for field in SOURCE_FIELDS}
    row.update(video_id=meta["id"], url=meta["webpage_url"], channel=(meta.get("channel") or "").strip(),
               upload_date=meta["upload_date"], duration_s=measured["duration_s"],
               resolution=f"{measured['width']}x{measured['height']}", fps=measured["fps"], sha256=sha256, role=role,
               city=screening["city"], day_night=screening["day_night"], verdict_notes=screening.get("notes", ""),
               near_duplicate_rate=round(dedup["rate_any"], 6))
    for column, key in CRITERIA:
        row[column] = screening["criteria"][key]
    if role == "eval":
        row["segments_used"] = ";".join(str(s) for s in select_segments(measured["duration_s"]))
    return row


def write_sources(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def verdict(prediction, blind_label):
    """Compare a confirmed prediction with the blind reading of the crop."""
    if blind_label in ("uncertain", "not_a_sign"):
        return blind_label
    return "correct" if blind_label == prediction else "wrong_code"


def tally(verdicts):
    counts = {name: 0 for name in VERDICTS}
    for item in verdicts:
        if item not in counts:
            raise ValueError(f"Unknown verdict {item!r}")
        counts[item] += 1
    counts["n"] = sum(counts[name] for name in VERDICTS)
    return counts


def decidable(counts):
    return counts["correct"] + counts["wrong_code"] + counts["not_a_sign"]


def precision(counts):
    """correct / (correct + wrong_code + not_a_sign); uncertain labels are left out."""
    total = decidable(counts)
    return counts["correct"] / total if total else None


def bootstrap_precision(rows, n_boot=2000, seed=20261029, alpha=0.05):
    """Percentile CI of precision resampling whole clusters (rows are (cluster, verdict))."""
    clusters = {}
    for cluster, item in rows:
        clusters.setdefault(cluster, []).append(item)
    names = sorted(clusters)
    if not names or precision(tally(item for _, item in rows)) is None:
        return None, None
    per_cluster = {name: tally(clusters[name]) for name in names}
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        picked = [per_cluster[rng.choice(names)] for _ in names]
        correct = sum(c["correct"] for c in picked)
        total = sum(decidable(c) for c in picked)
        if total:
            values.append(correct / total)
    if not values:
        return None, None
    values.sort()
    low = values[int(math.floor(alpha / 2 * (len(values) - 1)))]
    high = values[int(math.ceil((1 - alpha / 2) * (len(values) - 1)))]
    return low, high


def iou(a, b):
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def proxy_recall(annotations, frames, tolerance_s=2.0, min_iou=0.2):
    """Share of AI-annotated signs whose track was confirmed with the annotated code within the tolerance.

    annotations: {t, code, bbox}; frames: {timestamp, signs: [{track_id, bbox_xyxy, label}]}.
    A sign is caught if some track overlapping the annotation (IoU >= min_iou in a frame within the
    tolerance) shows the code as its confirmed label in any frame within the tolerance.
    """
    details = []
    for annotation in annotations:
        window = [f for f in frames if abs(f["timestamp"] - annotation["t"]) <= tolerance_s]
        tracks = {s["track_id"] for f in window for s in f["signs"]
                  if iou(s["bbox_xyxy"], annotation["bbox"]) >= min_iou}
        caught = any(s["track_id"] in tracks and s["label"] == annotation["code"] for f in window for s in f["signs"])
        details.append({**annotation, "caught": caught})
    caught_total = sum(d["caught"] for d in details)
    return {"annotated": len(details), "caught": caught_total,
            "recall": caught_total / len(details) if details else None, "details": details}


def summarize_review(items, labels, cluster_key, n_boot=2000, seed=20261029):
    """Per-config tallies from the unlocked mapping and the blind labels written before it was opened.

    items: dicts with number, config, prediction and the fields named by cluster_key (a callable).
    labels: {str(number): blind label}. Returns {config: {...counts, precision, ci_low, ci_high}} plus 'ALL'.
    """
    missing = [item["number"] for item in items if str(item["number"]) not in labels]
    if missing:
        raise ValueError(f"No blind label for items {missing[:5]}")
    graded = []
    for item in items:
        graded.append({**item, "verdict": verdict(item["prediction"], labels[str(item["number"])]),
                       "blind_label": labels[str(item["number"])]})
    summary = {}
    for name in sorted({g["config"] for g in graded}) + ["ALL"]:
        chosen = [g for g in graded if name == "ALL" or g["config"] == name]
        counts = tally(g["verdict"] for g in chosen)
        low, high = bootstrap_precision([(cluster_key(g), g["verdict"]) for g in chosen], n_boot, seed)
        summary[name] = {**counts, "decidable": decidable(counts), "precision": precision(counts),
                         "ci_low": low, "ci_high": high}
    return summary, graded


def error_count(counts):
    return counts["wrong_code"] + counts["not_a_sign"]


def decide_recommendations(table, min_labels=30, gate_pairs=None, val_macro_f1_drop=None):
    """Pre-registered rules of the owner brief (Part 6); `table` maps config -> counts + proxy_recall."""
    result = {"t2": {"decision": "insufficient_evidence", "checks": {}}, "gate": {}}
    if all(name in table for name in "ABC") and all(decidable(table[name]) >= min_labels for name in "ABC"):
        p_b, p_c = precision(table["B"]), precision(table["C"])
        recall_b, recall_c = table["B"].get("proxy_recall"), table["C"].get("proxy_recall")
        checks = {"precision_within_0.02": p_c >= p_b - 0.02,
                  "errors_not_above_A": error_count(table["C"]) <= error_count(table["A"]),
                  "proxy_recall_above_B": recall_b is not None and recall_c is not None and recall_c > recall_b}
        result["t2"] = {"decision": "recommend" if all(checks.values()) else "not_recommended", "checks": checks,
                        "values": {"precision_B": p_b, "precision_C": p_c, "errors_A": error_count(table["A"]),
                                   "errors_C": error_count(table["C"]), "proxy_recall_B": recall_b,
                                   "proxy_recall_C": recall_c}}
    for gated, base in (gate_pairs or {}).items():
        if gated not in table or base not in table or min(decidable(table[gated]), decidable(table[base])) < min_labels:
            result["gate"][gated] = {"decision": "insufficient_evidence", "checks": {}}
            continue
        errors_base, errors_gated = error_count(table[base]), error_count(table[gated])
        correct_base, correct_gated = table[base]["correct"], table[gated]["correct"]
        drop = (val_macro_f1_drop or {}).get(gated)
        checks = {"errors_drop_30pct": errors_base > 0 and (errors_base - errors_gated) / errors_base >= 0.30,
                  "correct_loss_within_5pct": correct_base > 0 and (correct_base - correct_gated) / correct_base <= 0.05,
                  "val_f1_drop_within_0.005": drop is not None and drop <= 0.005}
        result["gate"][gated] = {"decision": "recommend" if all(checks.values()) else "not_recommended",
                                 "checks": checks, "values": {"errors_base": errors_base, "errors_gated": errors_gated,
                                                              "correct_base": correct_base, "correct_gated": correct_gated,
                                                              "val_macro_f1_drop": drop}}
    return result


def make_sources(args):
    import json
    meta_dir, dedup_dir = Path(args.meta_dir), Path(args.dedup_dir)
    screening = json.loads(Path(args.screening).read_text(encoding="utf-8"))
    hashes = {}
    for line in Path(args.sha256).read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        hashes[Path(name.strip().lstrip("*")).stem] = digest
    rows = []
    for video_id, verdicts in screening.items():
        meta = json.loads((meta_dir / f"{video_id}.json").read_text(encoding="utf-8"))
        dedup = json.loads((dedup_dir / f"{video_id}_dedup.json").read_text(encoding="utf-8"))
        measured = measure_video(Path(args.video_root) / f"{video_id}.mp4")
        rows.append(build_source_row(meta, verdicts, hashes[video_id], dedup, measured))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_sources(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make-sources")
    for name in ("meta-dir", "screening", "sha256", "dedup-dir", "video-root", "out"):
        make.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    {"make-sources": make_sources}[args.command](args)


if __name__ == "__main__":
    main()
