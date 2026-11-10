"""Run the recognizer configurations A-E on the field-evaluation segments (issue #2, WP3).

Jobs are ordered segment by segment so that every finished segment has all requested configurations.
Outputs go to output/kiet_wp_20261029/field_eval/<config>_<video>_<start>/ (new directories only).
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.field_video_eval_20261029 import read_sources

OUT = ROOT / "output/kiet_wp_20261029/field_eval"
VIDEO_ROOT = ROOT / "output/kiet_wp_20261029/videos"          # downloaded media: inside the project, ignored by Git
TEMP_DIR = ROOT / "output/kiet_wp_20261029/tmp"               # keep every temporary file off drive C
OLD_MODEL = "models/sign_recognizer/retrained_20261009.pt"
NEW_MODEL = "models/sign_recognizer/retrained_20261028.pt"
OLD_GATE = "models/sign_recognizer/reference_gate_20261011_final.npz"


def configs(t2, gate_v2=None, gate_margin=None):
    table = {"A": {"model": OLD_MODEL, "threshold": 0.63, "gate": OLD_GATE},
             "B": {"model": NEW_MODEL, "threshold": 0.87, "gate": None},
             "C": {"model": NEW_MODEL, "threshold": t2, "gate": None}}
    if gate_v2:
        table["D"] = {"model": NEW_MODEL, "threshold": t2, "gate": gate_v2, "margin": gate_margin}
        table["E"] = {"model": NEW_MODEL, "threshold": 0.87, "gate": gate_v2, "margin": gate_margin}
    return table


def build_jobs(sources, names, available):
    """One job per (segment, config); segments in video order, all configs of a segment together."""
    jobs = []
    for row in sources:
        if row["role"] != "eval":
            continue
        for start in row["segments_used"]:
            for name in names:
                if name not in available:
                    raise ValueError(f"Configuration {name} is not available")
                jobs.append({"config": name, "video_id": row["video_id"], "start": start, "sha256": row["sha256"],
                             "name": f"{name}_{row['video_id']}_{start}"})
    return jobs


def run_job(job, table, video_root, out=OUT):
    target = out / job["name"]
    if (target / "temporal_benchmark.json").is_file():
        return {**job, "status": "skipped_done"}
    config = table[job["config"]]
    command = [sys.executable, "-X", "utf8", str(ROOT / "scripts/benchmark_temporal_video_20261028.py"),
               "--source", str(Path(video_root) / f"{job['video_id']}.mp4"), "--output", str(target),
               "--source-sha256", job["sha256"], "--recognizer", config["model"], "--threshold", repr(config["threshold"]),
               "--starts", str(job["start"]), "--duration", "60", "--trace-raw-recognition"]
    if config["gate"]:
        command += ["--reference-gate", config["gate"]]
        if config.get("margin") is not None:
            command += ["--reference-gate-margin", repr(config["margin"])]
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT), "YOLO_OFFLINE": "true",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TEMP": str(TEMP_DIR), "TMP": str(TEMP_DIR), "TMPDIR": str(TEMP_DIR)}
    out.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (out / f"{job['name']}.log").open("w", encoding="utf-8") as log:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    for video in list(target.rglob("*.mp4")):          # media is not needed after verified decode
        video.unlink()
    return {**job, "status": "ok" if code == 0 else "failed", "exit": code, "seconds": round(time.time() - started, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "data/manifests/field_video_20261029/sources.csv")
    parser.add_argument("--video-root", type=Path, default=VIDEO_ROOT)
    parser.add_argument("--configs", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--gate-v2", help="reference gate v2 for configurations D and E")
    parser.add_argument("--gate-margin", type=float, help="gate v2 margin (default: the one chosen in margin_selection.json)")
    parser.add_argument("--margin-selection", type=Path, default=ROOT / "output/kiet_wp_20261029/gate_v2/margin_selection.json")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--only-video", nargs="*", help="restrict to these video ids")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    release = json.loads((ROOT / "docs/release_gate_v2.json").read_text(encoding="utf-8"))
    t2 = release["recognizer_v2_crosscheck"]["alternative_threshold_val_only"]["threshold"]
    margin = args.gate_margin
    if args.gate_v2 and margin is None:
        margin = json.loads(args.margin_selection.read_text(encoding="utf-8"))["chosen"]["margin"]
    table = configs(t2, args.gate_v2, margin)
    sources = [r for r in read_sources(args.sources) if not args.only_video or r["video_id"] in args.only_video]
    jobs = build_jobs(sources, args.configs, table)
    print(json.dumps({"jobs": len(jobs), "t2": t2, "gate_margin": margin}), flush=True)
    if args.dry_run:
        for job in jobs:
            print(job["name"])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    status = OUT / "run_status.jsonl"
    with ThreadPoolExecutor(max(1, args.jobs)) as pool:
        for result in pool.map(lambda j: run_job(j, table, args.video_root), jobs):
            with status.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
            if result["status"] == "failed":
                print("FAILED", result["name"], flush=True)


if __name__ == "__main__":
    main()
