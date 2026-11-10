"""Run the post-T5-T7 configurations N87, N75 and N64 on the 15 field-evaluation segments (T8).

Same segments, sources and job layout as run_field_eval_20261029.py. Uses the 20261031 benchmark copy,
which applies --threshold to the video path. Outputs go to output/kiet_wp_20261029/field_eval/<config>_<video>_<start>/.
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
from scripts.run_field_eval_20261029 import build_jobs

OUT = ROOT / "output/kiet_wp_20261029/field_eval"
VIDEO_ROOT = ROOT / "output/kiet_wp_20261029/videos"
TEMP_DIR = ROOT / "output/kiet_wp_20261029/tmp"
NEW_MODEL = "models/sign_recognizer/retrained_20261028.pt"
BENCHMARK = ROOT / "scripts/benchmark_temporal_video_20261031.py"


def configs(t2):
    return {"N87": {"model": NEW_MODEL, "threshold": 0.87, "gate": None},
            "N75": {"model": NEW_MODEL, "threshold": 0.75, "gate": None},
            "N64": {"model": NEW_MODEL, "threshold": t2, "gate": None}}


def run_job(job, table, video_root, out=OUT):
    target = out / job["name"]
    if (target / "temporal_benchmark.json").is_file():
        return {**job, "status": "skipped_done"}
    config = table[job["config"]]
    command = [sys.executable, "-X", "utf8", str(BENCHMARK),
               "--source", str(Path(video_root) / f"{job['video_id']}.mp4"), "--output", str(target),
               "--source-sha256", job["sha256"], "--recognizer", config["model"], "--threshold", repr(config["threshold"]),
               "--starts", str(job["start"]), "--duration", "60", "--trace-raw-recognition"]
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT), "YOLO_OFFLINE": "true",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TEMP": str(TEMP_DIR), "TMP": str(TEMP_DIR), "TMPDIR": str(TEMP_DIR)}
    out.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (out / f"{job['name']}.log").open("w", encoding="utf-8") as log:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return {**job, "status": "ok" if code == 0 else "failed", "exit": code, "seconds": round(time.time() - started, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "data/manifests/field_video_20261029/sources.csv")
    parser.add_argument("--video-root", type=Path, default=VIDEO_ROOT)
    parser.add_argument("--configs", nargs="+", default=["N87", "N75", "N64"])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    release = json.loads((ROOT / "docs/release_gate_v2.json").read_text(encoding="utf-8"))
    t2 = release["recognizer_v2_crosscheck"]["alternative_threshold_val_only"]["threshold"]
    table = configs(t2)
    sources = read_sources(args.sources)
    jobs = build_jobs(sources, args.configs, table)
    print(json.dumps({"jobs": len(jobs), "t2": t2}), flush=True)
    if args.dry_run:
        for job in jobs:
            print(job["name"])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    status = OUT / "run_status_20261031.jsonl"
    with ThreadPoolExecutor(max(1, args.jobs)) as pool:
        for result in pool.map(lambda j: run_job(j, table, args.video_root), jobs):
            with status.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
            if result["status"] == "failed":
                print("FAILED", result["name"], flush=True)


if __name__ == "__main__":
    main()
