"""Summarise configurations B0, N87, N75, N64 on the field segments and export the blind review sheets (T8).

Reads output/kiet_wp_20261029/field_eval/<config>_<video>_<start>/ (B = old code at 0.87, reused from 2026-10-29).
Writes output/final_20261102/eval_summary.md and output/final_20261102/review/<config>/index.html.
Grades nothing: the review sheets are for the owner's blind reading.
"""
from __future__ import annotations
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.field_video_eval_20261029 import read_sources
from scripts.run_field_eval_20261029 import build_jobs
from src.perception.sign_catalog import NAMES_VI

FIELD = ROOT / "output/kiet_wp_20261029/field_eval"
VIDEOS = ROOT / "output/kiet_wp_20261029/videos"
OUT = ROOT / "output/final_20261102"
SOURCES = ROOT / "data/manifests/field_video_20261029/sources.csv"
CONFIGS = ["B", "N87", "N75", "N64"]
LABELS = {"B": "B0 (old code, 0.87)", "N87": "N87 (new code, 0.87)", "N75": "N75 (new code, 0.75)", "N64": "N64 (new code, T2)"}
MATCH_SECONDS = 2.0
MATCH_IOU = 0.2


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_run(run_dir):
    """Return (frames, report). frames: list of (timestamp, sign dict)."""
    report = json.loads((run_dir / "temporal_benchmark.json").read_text(encoding="utf-8"))
    frames = []
    for path in run_dir.rglob("predictions.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for sign in row["signs"]:
                frames.append((row["timestamp"], sign))
    return frames, report


def episodes(frames):
    """Consecutive coded observations of one track with the same code form one episode."""
    by_track = defaultdict(list)
    for t, sign in frames:
        by_track[sign["track_id"]].append((t, sign))
    out = []
    for track_id, items in by_track.items():
        items.sort(key=lambda x: x[0])
        current = None
        for t, sign in items:
            label = sign["label"]
            if label == "unknown":
                if current:
                    out.append(current)
                current = None
                continue
            if current and current["label"] == label:
                current["obs"].append((t, sign))
            else:
                if current:
                    out.append(current)
                current = {"track_id": track_id, "label": label, "obs": [(t, sign)]}
        if current:
            out.append(current)
    for ep in out:
        ep["t0"] = ep["obs"][0][0]
        ep["t1"] = ep["obs"][-1][0]
        ep["best"] = max(ep["obs"], key=lambda x: (x[1].get("raw_score") or 0))
        ep["confirmation_ms"] = next((s.get("confirmation_delay_ms") for _, s in ep["obs"]
                                      if s.get("confirmation_delay_ms") is not None), None)
    return out


def main():
    sources = [r for r in read_sources(SOURCES) if r["role"] == "eval"]
    table = {name: {} for name in CONFIGS}
    jobs = build_jobs(sources, CONFIGS, table)
    OUT.mkdir(parents=True, exist_ok=True)
    per_config = {name: {"episodes": [], "tracks": set(), "codes": Counter(), "runs": 0,
                         "fps": [], "render_s": [], "confirm_ms": [], "wiring": Counter(), "missing": []} for name in CONFIGS}
    by_job = {}
    for job in jobs:
        run_dir = FIELD / job["name"]
        if not (run_dir / "temporal_benchmark.json").is_file():
            per_config[job["config"]]["missing"].append(job["name"])
            continue
        frames, report = load_run(run_dir)
        video = job["video_id"]
        start = job["start"]
        stats = per_config[job["config"]]
        stats["runs"] += 1
        seg = report["segments"][0]
        stats["fps"].append(seg["inference_fps"])
        stats["render_s"].append(seg["render_wall_seconds"])
        for t, sign in frames:
            if sign["label"] != "unknown":
                stats["tracks"].add((video, start, sign["track_id"]))
        # Frame-level acceptance (before temporal confirmation): temporal copies a track label onto
        # frames whose own score is lower, so the per-box raw score alone cannot test the threshold.
        for events in run_dir.glob("events_*.jsonl"):
            for line in events.read_text(encoding="utf-8").splitlines():
                for sign in json.loads(line).get("pre_temporal_signs", []):
                    if sign["label"] != "unknown" and 0.75 <= (sign.get("recognition_confidence") or 0) < 0.87:
                        stats["wiring"]["frame_accepted_0.75_0.87"] += 1
        eps = episodes(frames)
        for ep in eps:
            ep["video"], ep["start"] = video, start
            stats["codes"][ep["label"]] += 1
            if ep["confirmation_ms"] is not None:
                stats["confirm_ms"].append(ep["confirmation_ms"])
        stats["episodes"].extend(eps)
        by_job[(job["config"], video, start)] = (frames, eps)

    def new_vs_b0(ep):
        b0_frames, _ = by_job.get(("B", ep["video"], ep["start"]), ([], []))
        bbox = ep["best"][1]["bbox_xyxy"]
        for t, sign in b0_frames:
            if sign["label"] == ep["label"] and abs(t - ep["t0"]) <= MATCH_SECONDS and iou(sign["bbox_xyxy"], bbox) >= MATCH_IOU:
                return False
        return True

    lines = ["# Field segment evaluation, configurations B0 / N87 / N75 / N64 (T8)", "",
             "Same 15 Hanoi segments (3 eval videos x 5 x 60 s, original field videos, unannotated), 6 Hz, same detector and lanes.",
             "B0 is the 2026-10-29 run of the old code at 0.87 (`output/kiet_wp_20261029/field_eval/B_*`), reused, not re-run.",
             "N87/N75/N64 are new code (T5-T7) at 0.87 / 0.75 / T2 = 0.6413 (`output/kiet_wp_20261029/field_eval/N*_*`).",
             "No correctness grading here; review sheets are in `output/final_20261102/review/`.", "",
             "| Config | Runs | Physical tracks with a code | Distinct codes | Median confirmation (ms) | Inference FPS (median) | Render wall time (s, sum) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in CONFIGS:
        s = per_config[name]
        conf = statistics.median(s["confirm_ms"]) if s["confirm_ms"] else float("nan")
        fps = statistics.median(s["fps"]) if s["fps"] else float("nan")
        lines.append(f"| {LABELS[name]} | {s['runs']} | {len(s['tracks'])} | {len(s['codes'])} | {conf:.0f} | {fps:.2f} | {sum(s['render_s']):.0f} |")
    lines += ["", "Missing runs: " + (", ".join(f"{n}: {len(per_config[n]['missing'])}" for n in CONFIGS) or "none"), "",
              "## Wiring check (F)", "",
              "| Config | Frame-level coded decisions with recognition_confidence in [0.75, 0.87) |", "|---|---:|"]
    for name in CONFIGS:
        lines.append(f"| {LABELS[name]} | {per_config[name]['wiring']['frame_accepted_0.75_0.87']} |")
    lines += ["", "N87 must be 0 (threshold 0.87 rejects that range); N75 must be > 0, otherwise the config did not take effect.", ""]

    review_root = OUT / "review"
    for name in ["N75", "N64"]:
        new = [ep for ep in per_config[name]["episodes"] if new_vs_b0(ep)]
        per_config[name]["new"] = new
        lines.append(f"- {LABELS[name]}: {len(new)} coded episodes not matched by B0 (same code, within {MATCH_SECONDS:g} s, IoU >= {MATCH_IOU}).")
        export_review(name, new, review_root / name)
    export_demo_three_signs(review_root / "three_signs")

    (OUT / "eval_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def export_review(name, episodes_new, folder):
    folder.mkdir(parents=True, exist_ok=True)
    cards = []
    for index, ep in enumerate(sorted(episodes_new, key=lambda e: (e["video"], e["start"], e["t0"]))):
        t, sign = ep["best"]
        path = folder / f"{index:03d}_{ep['label']}_{ep['video']}_{ep['start']}_{t:.2f}.jpg"
        crop_frame(ep["video"], t, sign["bbox_xyxy"], path)
        name_vi = NAMES_VI.get(ep["label"], "")
        cards.append(f"<figure><img src='{path.name}'><figcaption><b>{ep['label']}</b> {name_vi}<br>"
                     f"raw {sign.get('raw_score') or 0:.3f} · {ep['video']} @ {ep['start']}+{t - ep['start']:.1f}s · track {ep['track_id']}</figcaption></figure>")
    (folder / "index.html").write_text(html_page(f"Review {name}", cards), encoding="utf-8")


def export_demo_three_signs(folder):
    folder.mkdir(parents=True, exist_ok=True)
    demo = ROOT / "output/kiet_wp_20261030/demo/YQl9c5gCjhg_start30_60s"
    frames = []
    for line in (demo / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for sign in row["signs"]:
            frames.append((row["timestamp"], sign))
    cards = []
    for ep in episodes(frames):
        if ep["track_id"] not in (3, 4):
            continue
        t, sign = ep["best"]
        path = folder / f"track{ep['track_id']}_{ep['label']}_{t:.2f}.jpg"
        crop_frame("YQl9c5gCjhg", t, sign["bbox_xyxy"], path)
        cards.append(f"<figure><img src='{path.name}'><figcaption><b>{ep['label']}</b> {NAMES_VI.get(ep['label'], '')}<br>"
                     f"track {ep['track_id']} · raw {sign.get('raw_score') or 0:.3f}</figcaption></figure>")
    (folder / "index.html").write_text(html_page("Three clear signs (YQl9c5gCjhg demo)", cards), encoding="utf-8")


def crop_frame(video_id, t, bbox, path):
    capture = cv2.VideoCapture(str(VIDEOS / f"{video_id}.mp4"))
    capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return
    x1, y1, x2, y2 = (int(v) for v in bbox)
    pad = 4
    crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
    side = min(crop.shape[:2])
    if side and side < 96:
        scale = 96 / side
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(path), crop)


def html_page(title, cards):
    body = "\n".join(cards) or "<p>No items.</p>"
    return (f"<!doctype html><meta charset='utf-8'><title>{title}</title>"
            "<style>body{font-family:sans-serif;margin:16px}"
            "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}"
            "figure{margin:0;border:1px solid #ccc;padding:8px}img{width:100%;image-rendering:auto}"
            "figcaption{font-size:13px;margin-top:6px}</style>"
            f"<h1>{title}</h1><main>{body}</main>")


if __name__ == "__main__":
    main()
