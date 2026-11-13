"""Resumable training queue for the 2026-11-12 benchmark (Giai doan A, items A2/A5).

Each job gets output/bench_20261112/<stage>/<job>/. A job is finished when that folder holds
results.json (success) or error.json (failure, message verbatim). Finished jobs are skipped on
re-run, so the queue can be restarted after any interruption. The queue reads queue.json on
every pick, so jobs can be appended while it runs.

Recipe (R40, shared by all detectors): 40 epochs, imgsz 640, AdamW lr0 1e-3 lrf 0.01, AMP,
seed 391, deterministic, batch 16 with fallback 8 then 4 on CUDA out-of-memory, 1 class,
configs/sign_dataset.yaml, augmentation of the final fine-tune stage (hsv 0.005/0.15/0.15,
degrees 2, translate 0.03, scale 0.15, no flips, no mosaic, no mixup).
Model selection inside a run uses VAL only (ultralytics best.pt = max VAL mAP50-95).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "output/bench_20261112"
QUEUE = BENCH / "queue.json"
WEIGHTS_DIR = BENCH / "weights"
DATA_YAML = ROOT / "configs/sign_dataset.yaml"
CURRENT_DETECTOR = ROOT / "models/sign_detector/finetuned_20261026.pt"
RTDETR_MAX_HOURS = 6.0

R40 = dict(epochs=40, imgsz=640, optimizer="AdamW", lr0=0.001, lrf=0.01, amp=True,
           seed=391, deterministic=True, device=0, workers=2, plots=False, cache=False,
           hsv_h=0.005, hsv_s=0.15, hsv_v=0.15, degrees=2.0, translate=0.03, scale=0.15,
           shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0,
           copy_paste=0.0)


def finished(job_dir: Path) -> bool:
    return (job_dir / "results.json").is_file() or (job_dir / "error.json").is_file()


def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


class TimeBudgetExceeded(RuntimeError):
    pass


def make_callbacks(state: dict, max_hours: float | None):
    import torch

    def on_train_epoch_end(trainer):
        import time as _t
        now = _t.time()
        state["epochs_seen"] = trainer.epoch + 1
        state["peak_vram_gb"] = max(state.get("peak_vram_gb", 0.0),
                                    torch.cuda.max_memory_reserved() / 1e9)
        if state.get("epoch_started"):
            state["last_epoch_seconds"] = now - state["epoch_started"]
        state["epoch_started"] = now
        if max_hours is not None and state.get("epochs_seen") == 1:
            # Measured from the first epoch only (includes validation): stop early if the whole run
            # would exceed the budget; the job is then recorded as skipped, not retried.
            estimate_h = (now - state["train_started"]) * trainer.epochs / 3600.0
            state["estimate_hours"] = estimate_h
            if estimate_h > max_hours:
                raise TimeBudgetExceeded(
                    f"Estimated {estimate_h:.2f} h for {trainer.epochs} epochs exceeds {max_hours} h budget")

    def on_train_start(trainer):
        state["train_started"] = time.time()

    return {"on_train_start": on_train_start, "on_train_epoch_end": on_train_epoch_end}


def best_epoch_from_csv(save_dir: Path) -> dict:
    import pandas as pd

    frame = pd.read_csv(save_dir / "results.csv")
    frame.columns = [c.strip() for c in frame.columns]
    index = int(frame["metrics/mAP50-95(B)"].idxmax())
    row = frame.iloc[index]
    return {"best_epoch_by_val_map50_95": int(row["epoch"]),
            "best_val_map50": float(row["metrics/mAP50(B)"]),
            "best_val_map50_95": float(row["metrics/mAP50-95(B)"]),
            "best_val_precision": float(row["metrics/precision(B)"]),
            "best_val_recall": float(row["metrics/recall(B)"]),
            "epochs_completed": int(len(frame))}


def model_cost(weights: Path) -> dict:
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    params = int(sum(p.numel() for p in model.model.parameters()))
    try:
        info = model.model.info(detailed=False, verbose=False, imgsz=640)
        gflops = float(info[-1]) if info and info[-1] is not None else None
    except Exception:  # GFLOPs are informational; a failure is recorded as null
        gflops = None
    del model
    torch.cuda.empty_cache()
    return {"params": params, "gflops_640": gflops}


def run_detector(job: dict, job_dir: Path) -> dict:
    import torch
    from ultralytics import YOLO

    data = job.get("data", str(DATA_YAML))
    batch_options = job.get("batches", [16, 8, 4])
    attempts = []
    for batch in batch_options:
        run_name = f"attempt_b{batch}"
        state: dict = {}
        started = time.time()
        torch.cuda.reset_peak_memory_stats()
        try:
            model = YOLO(job["model"])  # official .pt name (downloaded) or a model .yaml
            for event, fn in make_callbacks(state, job.get("max_hours")).items():
                model.add_callback(event, fn)
            train_args = dict(R40, data=data, batch=batch, project=str(job_dir), name=run_name,
                              exist_ok=False, epochs=job.get("epochs", R40["epochs"]))
            if job.get("pretrained"):
                # ultralytics transfers every shape-matching tensor from this checkpoint
                train_args["pretrained"] = job["pretrained"]
            train_args.update(job.get("overrides", {}))
            model.train(**train_args)
            save_dir = Path(model.trainer.save_dir)
            best = save_dir / "weights" / "best.pt"
            summary = best_epoch_from_csv(save_dir)
            cost = model_cost(best)
            result = {
                "status": "COMPLETED", "job": job["name"], "stage": job["stage"],
                "model": job["model"], "pretrained": job.get("pretrained"),
                "data": data, "batch_used": batch, "batch_attempts": [a["batch"] for a in attempts] + [batch],
                "attempts": attempts,
                "recipe": {k: train_args[k] for k in ("epochs", "imgsz", "optimizer", "lr0", "lrf",
                                                     "amp", "seed", "deterministic", "batch",
                                                     "hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
                                                     "scale", "fliplr", "flipud", "mosaic", "mixup")},
                "best_weights": str(best.relative_to(ROOT)).replace("\\", "/"),
                "train_seconds": time.time() - started,
                "peak_vram_gb_reserved": max(state.get("peak_vram_gb", 0.0), torch.cuda.max_memory_reserved() / 1e9),
                "cost": cost, **summary,
            }
            (job_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            return result
        except TimeBudgetExceeded as exc:
            result = {"status": "SKIPPED_TIME_BUDGET", "job": job["name"], "message": str(exc),
                      "estimate_hours": state.get("estimate_hours")}
            (job_dir / "error.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            return result
        except Exception as exc:  # recorded verbatim; OOM triggers the next smaller batch
            text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            attempts.append({"batch": batch, "error": text, "seconds": time.time() - started})
            torch.cuda.empty_cache()
            if is_oom(exc) and batch != batch_options[-1]:
                continue
            result = {"status": "ERROR", "job": job["name"], "attempts": attempts}
            (job_dir / "error.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            return result
    return {"status": "ERROR", "job": job["name"], "attempts": attempts}


def run_script(job: dict, job_dir: Path) -> dict:
    """Recognizer and one-stage dataset steps run as separate scripts with their own outputs."""
    import subprocess

    cmd = job["command"]
    log = job_dir / "stdout.log"
    with log.open("w", encoding="utf-8") as stream:
        code = subprocess.call(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if code == 0 and (job_dir / "results.json").is_file():
        return {"status": "COMPLETED", "job": job["name"]}
    result = {"status": "ERROR", "job": job["name"], "exit_code": code,
              "message": f"see {log.relative_to(ROOT)}"}
    if code == 0:
        result["message"] = "script exited 0 but wrote no results.json"
    (job_dir / "error.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_job(job: dict) -> dict:
    job_dir = BENCH / job["stage"] / job["name"]
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "train.log"
    stream = log_path.open("a", encoding="utf-8", buffering=1)
    stdout, stderr = os.dup(1), os.dup(2)
    os.dup2(stream.fileno(), 1)
    os.dup2(stream.fileno(), 2)
    try:
        print(json.dumps({"start": job["name"], "time": time.strftime("%Y-%m-%d %H:%M:%S")}), flush=True)
        if job["kind"] == "detector":
            return run_detector(job, job_dir)
        if job["kind"] == "script":
            return run_script(job, job_dir)
        raise ValueError(f"unknown kind {job['kind']}")
    except Exception as exc:
        result = {"status": "ERROR", "job": job["name"],
                  "message": "".join(traceback.format_exception_only(type(exc), exc)).strip()}
        (job_dir / "error.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result
    finally:
        os.dup2(stdout, 1)
        os.dup2(stderr, 2)
        stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run at most one pending job")
    args = parser.parse_args()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(WEIGHTS_DIR)  # ultralytics downloads official weights to the working directory
    while True:
        jobs = json.loads(QUEUE.read_text(encoding="utf-8"))
        pending = [j for j in jobs if not finished(BENCH / j["stage"] / j["name"])]
        if not pending:
            print(json.dumps({"queue": "empty"}), flush=True)
            break
        job = pending[0]
        outcome = run_job(job)
        with (BENCH / "queue_log.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"job": job["name"], "status": outcome.get("status"),
                                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False) + "\n")
        if args.once:
            break


if __name__ == "__main__":
    main()
