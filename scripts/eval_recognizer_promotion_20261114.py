"""Evaluate the frozen recognizer checkpoints for M3 without changing defaults."""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from hashlib import file_digest
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.models import efficientnet_b0

from scripts import analyze_field_a7_20261112 as field_a7
from src.perception.recognizer_v2_crops import exact_crop, load_cache
from src.perception.sign_detector import SignDetector
from src.perception.sign_recognizer import SignRecognizer, crop_tensor
from src.utils.config import load_config
from src.utils.types import SignDetection

SHARED = Path(r"D:\DSP391m\project\itas-mm-scaffold")
V2 = SHARED / "output/kiet_wp_20261028/recognizer_v2"
A4 = SHARED / "output/bench_20261112/A4"
A7 = SHARED / "output/bench_20261112/A7"
PACK = Path(r"D:\DSP391m\project\review_pack_20261019_239_proposals")
OUTPUT_ROOT = SHARED / "output/demo_20261114/M3"
OUT = OUTPUT_ROOT
OLD = ROOT / "models/sign_recognizer/retrained_20261009.pt"
CANDIDATE = A4 / "efficientnet_b0/candidate.pt"
LOCK = OUT.parent / "GPU_LOCK"
SEED = 20261112
DRAWS = 1000
VIDEO_THRESHOLD = 0.6413
GROUPS = {
    "all": (0, np.inf),
    "<16": (0, 16),
    "16-32": (16, 32),
    "32-64": (32, 64),
    ">=64": (64, np.inf),
    ">=32": (32, np.inf),
}


def write_json(name: str, data: dict) -> None:
    path = OUT / name
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def load_baseline(config: dict, device: str) -> SignRecognizer:
    settings = {**config["sign_recognizer"], "device": device}
    settings["weights_path"] = str(ROOT / settings["weights_path"])
    baseline = SignRecognizer({"sign_recognizer": settings})
    if baseline.model is None or baseline.reference_gate is not None:
        raise RuntimeError("Expected the deployed MobileNet without an embedding gate")
    return baseline


def load_candidate(baseline: SignRecognizer) -> SignRecognizer:
    checkpoint = torch.load(CANDIDATE, map_location="cpu", weights_only=True)
    if checkpoint["backbone"] != "efficientnet_b0" or checkpoint["classes"] != baseline.classes:
        raise ValueError("Unexpected candidate backbone or class mapping")
    # Use the same runtime crop, filtering and postprocessing with the frozen model.
    model = efficientnet_b0(weights=None, num_classes=len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    runtime = SignRecognizer({"sign_recognizer": {**baseline.config, "weights_path": ""}})
    runtime.model = model.to(baseline.device).eval()
    runtime.device = baseline.device
    runtime.classes = checkpoint["classes"]
    runtime.size = checkpoint["image_size"]
    runtime.threshold = checkpoint["confidence_threshold"]
    return runtime


def read_split(split: str, classes: list[str]):
    boxes = pd.read_csv(V2 / "boxes.csv", keep_default_na=False)
    rows = boxes[boxes.split == split]
    if split == "VAL":
        background = pd.read_csv(V2 / "background_boxes.csv", keep_default_na=False)
        columns = ["image_path", "bbox_xyxy", "short_side_px", "code", "source"]
        eligible = rows[(rows.short_side_px >= 16) & rows.code.isin(classes)]
        rows = pd.concat([eligible[columns], background.loc[background.split == split, columns]])
    suffix = "test_all" if split == "TEST" else "val"
    entries = load_cache(V2 / f"crop_cache_{suffix}.pkl")
    if len(entries) != len(rows):
        raise ValueError(f"{split} manifest and cache lengths differ")
    keep = [i for i, entry in enumerate(entries) if entry is not None]
    return rows.iloc[keep].reset_index(drop=True), [entries[i] for i in keep]


@torch.inference_mode()
def predict(recognizer: SignRecognizer, entries: list) -> tuple[np.ndarray, np.ndarray]:
    names = np.asarray(recognizer.classes)
    labels, scores = [], []
    for start in range(0, len(entries), 128):
        inputs = torch.stack([
            crop_tensor(exact_crop(entry), recognizer.size)
            for entry in entries[start:start + 128]
        ])
        confidence, indices = recognizer.model(inputs.to(recognizer.device)).softmax(1).max(1)
        labels.extend(names[indices.cpu().numpy()])
        scores.extend(confidence.cpu().tolist())
    return np.asarray(labels), np.asarray(scores)


def paired_metrics(rows: pd.DataFrame, models: dict[str, np.ndarray]) -> dict:
    truth = rows.code.to_numpy()
    labels, truth_index = np.unique(truth, return_inverse=True)
    images, image_index = np.unique(rows.image_path.to_numpy(), return_inverse=True)
    # Resample source images. Every crop from an image gets the same weight
    # for both models. Keep the observed class support fixed within each bucket.
    rng = np.random.default_rng(SEED)
    draws = np.stack([
        np.bincount(rng.integers(0, len(images), len(images)), minlength=len(images))
        for _ in range(DRAWS)
    ])
    weights = np.vstack([np.ones(len(images), dtype=int), draws])
    true_counts = np.zeros((len(images), len(labels)), dtype=np.int32)
    np.add.at(true_counts, (image_index, truth_index), 1)
    results = {}
    for name, predictions in models.items():
        correct = predictions == truth
        pred_counts = np.zeros_like(true_counts)
        tp_counts = np.zeros_like(true_counts)
        known = np.isin(predictions, labels)
        np.add.at(pred_counts, (image_index[known], labels.searchsorted(predictions[known])), 1)
        np.add.at(tp_counts, (image_index[correct], truth_index[correct]), 1)
        true_total = weights @ true_counts
        denom = true_total + weights @ pred_counts
        tp = weights @ tp_counts
        f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom != 0)
        results[name] = {"accuracy": tp.sum(axis=1) / true_total.sum(axis=1), "macro_f1": f1.mean(axis=1)}

    def estimate(values):
        return {"point": float(values[0]), "ci95": np.percentile(values[1:], [2.5, 97.5]).tolist()}

    return {
        "crops": len(rows),
        "images": len(images),
        "classes": len(labels),
        **{model: {metric: estimate(values) for metric, values in metrics.items()} for model, metrics in results.items()},
        "delta": {
            metric: estimate(results["efficientnet"][metric] - results["mobilenet"][metric])
            for metric in ("accuracy", "macro_f1")
        },
    }


def operating_point(truth, names, scores, threshold):
    accepted = (scores >= threshold) & (names != "unknown")
    count = int(accepted.sum())
    correct = int((accepted & (truth == names)).sum())
    return {
        "threshold": float(threshold), "accepted": count, "correct": correct,
        "precision": correct / count if count else None, "coverage": count / len(truth),
    }


def calibrate(truth, baseline, candidate):
    reference = operating_point(truth, *baseline, VIDEO_THRESHOLD)
    names, scores = candidate
    eligible = np.flatnonzero(names != "unknown")
    order = eligible[np.argsort(-scores[eligible], kind="stable")]
    sorted_scores = scores[order]
    if reference["precision"] is None or not len(sorted_scores):
        raise ValueError("VAL requires accepted MobileNet codes and predicted EfficientNet codes")
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    correct = np.cumsum(names[order] == truth[order])
    precision = correct[ends] / (ends + 1)
    error = np.abs(precision - reference["precision"])
    # Nearest attainable precision; for a tie, keep the highest coverage.
    closest = np.flatnonzero(np.isclose(error, error.min(), rtol=0, atol=1e-12))[-1]
    selected = operating_point(truth, names, scores, sorted_scores[ends[closest]])
    return {"n": len(truth), "mobilenet": reference, "efficientnet": selected,
            "precision_delta": selected["precision"] - reference["precision"]}


def run_crops(config):
    baseline = load_baseline(config, "cpu")
    test, entries = read_split("TEST", baseline.classes)
    old = torch.load(OLD, map_location="cpu", weights_only=True)
    mask = test.code.isin(set(old["classes"]) - {"unknown"}).to_numpy()
    test = test.loc[mask].reset_index(drop=True)
    entries = [entry for entry, keep in zip(entries, mask, strict=True) if keep]
    labels, scores = predict(baseline, entries)
    large = test.short_side_px.to_numpy() >= 32
    references = {"all": (3144, 0.9090, np.ones(len(test), dtype=bool)),
                  ">=32": (1255, 0.9697, large)}
    replay = {}
    for name, (expected_n, expected_accuracy, bucket) in references.items():
        count = int(bucket.sum())
        correct = int((labels[bucket] == test.code.to_numpy()[bucket]).sum())
        replay[name] = {"n": count, "correct": correct, "accuracy": correct / count,
                        "expected_n": expected_n, "expected_accuracy": expected_accuracy}
    reproduced = all(
        v["n"] == v["expected_n"] and abs(v["accuracy"] - v["expected_accuracy"]) <= 0.002
        for v in replay.values()
    )
    write_json("baseline.json", {"reproduced": reproduced, "tolerance": 0.002, "results": replay})
    if not reproduced:
        raise RuntimeError(f"MobileNet TEST baseline was outside the M3 tolerance: {replay}")

    candidate = load_candidate(baseline)
    val, val_entries = read_split("VAL", baseline.classes)
    val_baseline = predict(baseline, val_entries)
    val_candidate = predict(candidate, val_entries)
    calibration = calibrate(val.code.to_numpy(), val_baseline, val_candidate)
    write_json("val.json", calibration)
    print("VAL:", calibration, flush=True)

    candidate_labels, candidate_scores = predict(candidate, entries)
    results = {}
    for name, (low, high) in GROUPS.items():
        keep = (test.short_side_px >= low) & (test.short_side_px < high)
        results[name] = paired_metrics(
            test.loc[keep], {"mobilenet": labels[keep], "efficientnet": candidate_labels[keep]}
        )
    test = test[["image_path", "bbox_xyxy", "short_side_px", "code"]].assign(
        mobilenet=labels, mobilenet_score=scores,
        efficientnet=candidate_labels, efficientnet_score=candidate_scores,
    )
    test.to_csv(OUT / "test-predictions.csv", index=False, mode="x")
    inputs = [Path(baseline.config["weights_path"]), OLD, CANDIDATE, V2 / "boxes.csv",
              V2 / "background_boxes.csv", V2 / "crop_cache_test_all.pkl", V2 / "crop_cache_val.pkl"]
    write_json("test.json", {"device": "cpu", "seed": SEED, "draws": DRAWS,
                            "sampling_unit": "source image within each size bucket",
                            "input_sha256": {str(path): checksum(path) for path in inputs},
                            "buckets": results})
    print("TEST all:", results["all"], flush=True)


def run_field(config):
    field_a7.A7 = A7
    first = {(entry["frame_id"], entry["box_id"]): entry["label"]
             for entry in field_a7.load_pass("pass1").values()}
    second = {(entry["frame_id"], entry["box_id"]): entry["label"]
              for entry in field_a7.load_pass("pass2").values()}
    reference = {key: label for key, label in first.items()
                 if label == second.get(key) and label not in {"uncertain", "not_sign"}}
    if len(reference) != 65:
        raise ValueError(f"Expected 65 A7 reference boxes, found {len(reference)}")
    baseline = load_baseline(config, "cpu")
    candidate = load_candidate(baseline)
    threshold = json.loads((OUT / "val.json").read_text(encoding="utf-8"))["efficientnet"]["threshold"]
    config["sign_detector"]["device"] = "cpu"
    config["sign_detector"]["weights_path"] = str(ROOT / config["sign_detector"]["weights_path"])
    detector = SignDetector(config)
    if detector._model is None:
        raise RuntimeError("Production detector unavailable")
    manifest = json.loads((PACK / "review_manifest.json").read_text(encoding="utf-8"))

    results = []
    for frame in manifest["frames"]:
        frame_id = frame["id"]
        raw = frame["signs"]
        boxes = ast.literal_eval(raw) if isinstance(raw, str) else raw
        if not any((frame_id, box["id"]) in reference for box in boxes):
            continue
        image = cv2.imread(str(PACK / frame["image"]))
        if image is None:
            raise FileNotFoundError(PACK / frame["image"])
        detections = detector.detect(image)
        still_baseline = baseline.recognize(image, deepcopy(detections))
        still_candidate = candidate.recognize(image, deepcopy(detections))
        video_baseline = baseline.recognize(image, deepcopy(detections), threshold=VIDEO_THRESHOLD)
        video_candidate = candidate.recognize(image, deepcopy(detections), threshold=threshold)
        used = set()
        for box in boxes:
            overlaps = [(field_a7.iou(box["bbox_xyxy"], det.bbox_xyxy), index)
                        for index, det in enumerate(detections) if index not in used]
            score, index = max(overlaps, key=lambda item: item[0], default=(0.0, None))
            matched = score >= 0.5
            if matched:
                used.add(index)
            key = frame_id, box["id"]
            if key in reference:
                results.append({
                    "frame": frame_id, "box": box["id"], "reference": reference[key], "matched": matched,
                    "a7_mobilenet": still_baseline[index].label if matched else None,
                    "still_efficientnet": still_candidate[index].label if matched else None,
                    "video_threshold_mobilenet": video_baseline[index].label if matched else None,
                    "video_threshold_efficientnet": video_candidate[index].label if matched else None,
                })

    if len(results) != len(reference):
        raise ValueError(f"Matched manifest entries do not cover the A7 reference: {len(results)}")
    predictions = ("a7_mobilenet", "still_efficientnet", "video_threshold_mobilenet", "video_threshold_efficientnet")
    totals = {model: sum(row[model] == row["reference"] for row in results) for model in predictions}
    write_json("field.json", {"reference": len(reference), "matched": sum(row["matched"] for row in results),
                             "correct": totals, "baseline_threshold": baseline.threshold,
                             "candidate_still_threshold": candidate.threshold,
                             "candidate_video_threshold": threshold,
                             "detector_sha256": checksum(Path(config["sign_detector"]["weights_path"])),
                             "results": results})
    print("FIELD:", totals, flush=True)


def run_latency(config, device):
    token = os.environ.get("ITAS_M3_GPU_LOCK_TOKEN")
    if not token or LOCK.read_text(encoding="utf-8").strip() != token:
        raise RuntimeError("Latency requires ownership of the shared GPU lock")
    baseline = load_baseline(config, device)
    candidate = load_candidate(baseline)
    candidate_threshold = json.loads((OUT / "val.json").read_text(encoding="utf-8"))["efficientnet"]["threshold"]
    _, val_entries = read_split("VAL", baseline.classes)
    crops = []
    for entry in val_entries:
        crop = exact_crop(entry)
        if min(crop.shape[:2]) >= 32:
            crops.append(crop)
        if len(crops) == 8:
            break
    if len(crops) != 8:
        raise ValueError("Need eight representative VAL crops for the runtime benchmark")
    widths = [crop.shape[1] for crop in crops]
    canvas = np.zeros((max(crop.shape[0] for crop in crops), sum(widths), 3), dtype=np.uint8)
    boxes = []
    start = 0
    for crop, width in zip(crops, widths, strict=True):
        height = crop.shape[0]
        canvas[:height, start:start + width] = crop
        boxes.append((start, 0, start + width, height))
        start += width

    def timed(recognizer, threshold, batch):
        signs = [SignDetection("sign", 1.0, box) for box in boxes[:batch]]
        start_ns = time.perf_counter_ns()
        recognizer.recognize(canvas, signs, threshold=threshold)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        return (time.perf_counter_ns() - start_ns) / 1e6

    pairs = {"mobilenet": (baseline, VIDEO_THRESHOLD), "efficientnet": (candidate, candidate_threshold)}
    results = {}
    for batch in (1, 8):
        for _ in range(20):
            for recognizer, threshold in pairs.values():
                timed(recognizer, threshold, batch)
        readings = {name: [] for name in pairs}
        for iteration in range(200):
            order = tuple(pairs) if iteration % 2 == 0 else tuple(reversed(pairs))
            for name in order:
                readings[name].append(timed(*pairs[name], batch))
        results[str(batch)] = {
            name: {"batch_p50_ms": float(np.median(values)), "batch_p95_ms": float(np.percentile(values, 95)),
                   "amortized_crop_p50_ms": float(np.median(values) / batch), "raw_batch_ms": values}
            for name, values in readings.items()
        }
    write_json(f"latency-{device.split(':')[0]}.json",
               {"device": device, "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
                "torch": torch.__version__, "cpu_threads": torch.get_num_threads(), "warmup": 20,
                "iterations": 200, "included": "crop, preprocessing, model, softmax, postprocessing and sync",
                "batch": results})


def main():
    global OUT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("crops", "field", "latency-cpu", "latency-gpu"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--run-name", help="write to a separate subdirectory under M3")
    args = parser.parse_args()
    if args.run_name:
        run_name = Path(args.run_name)
        if run_name.is_absolute() or len(run_name.parts) != 1 or run_name.name in {"", ".."}:
            parser.error("--run-name must be a single directory name under M3")
        OUT = OUTPUT_ROOT / run_name
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config("perception")
    if args.stage != "crops":
        baseline = json.loads((OUT / "baseline.json").read_text(encoding="utf-8"))
        if not baseline["reproduced"]:
            raise RuntimeError("M3 baseline did not reproduce")
    if args.stage == "crops":
        run_crops(config)
    elif args.stage == "field":
        run_field(config)
    else:
        run_latency(config, "cpu" if args.stage == "latency-cpu" else "cuda:0")


if __name__ == "__main__":
    main()
