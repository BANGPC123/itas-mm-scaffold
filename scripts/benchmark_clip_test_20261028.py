"""Compare frozen detectors on selected canonical TEST clip frames."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from src.evaluation.field_metrics import iou

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def match_boxes(source, predictions, threshold=0.5):
    if not source or not predictions:
        return []
    scores = np.asarray([[iou(g["bbox_xyxy"], p["bbox_xyxy"]) for p in predictions] for g in source])
    left, right = linear_sum_assignment(-np.where(scores >= threshold, 1 + scores / 1000, 0))
    return [(int(i), int(j)) for i, j in zip(left, right) if scores[i, j] >= threshold]


def score_frame(source, review, predictions):
    truth = source["signs"]
    pairs = match_boxes(truth, predictions)
    labels = {item["id"]: item for item in review["signs"]}
    eligible = {index: labels[sign["id"]]["label"] for index, sign in enumerate(truth)
                if sign["id"] in labels and labels[sign["id"]].get("judgment") == "sign"
                and labels[sign["id"]].get("label") not in (None, "", "unknown")}
    matched = [(i, j) for i, j in pairs if i in eligible]
    return {"tp": len(pairs), "fp": len(predictions) - len(pairs), "fn": len(truth) - len(pairs),
            "code_correct": sum(predictions[j].get("label") == eligible[i] for i, j in matched),
            "code_matched": len(matched), "code_eligible": len(eligible),
            "lane_proposed": review["lane_status"] == "annotated"}


def bootstrap(groups, iterations=1000, seed=20261028):
    """Paired percentile intervals, resampling entire clip groups."""
    names = sorted(groups)
    if not names or iterations < 1:
        raise ValueError("Clip groups and iterations are required")
    rng = np.random.default_rng(seed)
    values = {f"{model}_{metric}": [] for model in ("baseline", "candidate")
              for metric in ("precision", "recall", "recognizer")}
    values.update({f"{metric}_delta": [] for metric in ("precision", "recall", "recognizer")})
    for _ in range(iterations):
        chosen = rng.integers(0, len(names), size=len(names))
        totals = {model: {key: sum(groups[names[i]][model].get(key, 0) for i in chosen)
                          for key in ("tp", "fp", "fn", "code_correct", "code_matched")}
                  for model in ("baseline", "candidate")}
        for metric, numerator, denominator in (("precision", "tp", "fp"), ("recall", "tp", "fn"),
                                                ("recognizer", "code_correct", "code_matched")):
            scores = {}
            for model, counts in totals.items():
                divisor = counts["code_matched"] if metric == "recognizer" else counts[numerator] + counts[denominator]
                if divisor:
                    scores[model] = counts[numerator] / divisor
                    values[f"{model}_{metric}"].append(scores[model])
            if len(scores) == 2:
                values[f"{metric}_delta"].append(scores["candidate"] - scores["baseline"])
    return {"iterations": iterations, "seed": seed, "clip_groups": len(names),
            **{name + "_95ci": [float(v) for v in np.percentile(samples, [2.5, 97.5])] if samples else None
               for name, samples in values.items()},
            "valid_resamples": {name: len(samples) for name, samples in values.items()}}


def summarize(records):
    keys = ("tp", "fp", "fn", "code_correct", "code_matched", "code_eligible")
    total = {key: sum(record[key] for record in records) for key in keys}
    total["precision"] = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else None
    total["recall"] = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else None
    total["recognizer_accuracy_vs_ai_labels_on_matched_boxes"] = (
        total["code_correct"] / total["code_matched"] if total["code_matched"] else None)
    total["recognizer_matching_coverage"] = (
        total["code_matched"] / total["code_eligible"] if total["code_eligible"] else None)
    return total


def run(pack, review_path, output):
    import cv2
    import torch
    import ultralytics
    from src.evaluation.label_review_schema import validate_review
    from src.perception.sign_detector import SignDetector
    from src.perception.sign_recognizer import SignRecognizer
    from src.utils.config import load_config

    pack, review_path, output = Path(pack).resolve(), Path(review_path).resolve(), Path(output).resolve()
    if ROOT / "output/kiet_wp_20261028/wp3" not in output.parents or output.exists():
        raise ValueError("Output must be a new directory inside WP3 output")
    manifest_path = pack / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    validate_review(review, manifest)
    if review.get("status") != "COMPLETED" or review.get("review_method", {}).get("type") != "ai_reviewed":
        raise ValueError("Completed AI review required")
    if any(f.get("human_confirmed") is not False or f.get("review_method", {}).get("type") != "ai_reviewed"
           for f in review["frames"]):
        raise ValueError("Review frames must be AI-attributed and not human-confirmed")
    if len(review["frames"]) != len(manifest["frames"]) or review["provenance"]["input_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("AI review does not cover the input manifest")
    expected_clips = {"IMG_" + item for item in ("4017", "4022", "4024", "4032", "4047", "4071", "4094", "4112", "4135", "4186", "4189")}
    if {f["source_group"] for f in manifest["frames"]} != expected_clips:
        raise ValueError("Unexpected clip groups")
    cfg = load_config("perception")
    config_path = ROOT / "configs/perception.yaml"
    if sha256(config_path) != review["provenance"]["perception_config_sha256"]:
        raise ValueError("Config differs from candidate AI inference")
    paths = {"baseline": ROOT / "models/sign_detector/retrained_20261009.pt",
             "candidate": ROOT / "models/sign_detector/finetuned_20261026.pt",
             "recognizer": ROOT / "models/sign_recognizer/retrained_20261009.pt"}
    hashes = {key: sha256(path) for key, path in paths.items()}
    if (hashes["candidate"] != review["provenance"]["model_sha256"]["detector"] or
            hashes["recognizer"] != review["provenance"]["model_sha256"]["recognizer"]):
        raise ValueError("Checkpoint differs from candidate AI inference")
    hashes["lane"] = review["provenance"]["model_sha256"]["lane"]
    reference = cfg["sign_recognizer"].get("reference_gate_path")
    if reference:
        reference = (ROOT / reference).resolve()
        if sha256(reference) != review["provenance"]["model_sha256"].get("recognizer_reference_gate"):
            raise ValueError("Recognizer reference gate differs from AI inference")
        cfg["sign_recognizer"]["reference_gate_path"] = str(reference)
    cfg["sign_detector"]["weights_path"] = str(paths["baseline"])
    cfg["sign_recognizer"]["weights_path"] = str(paths["recognizer"])
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    detector, recognizer = SignDetector(cfg), SignRecognizer(cfg)
    if detector._model is None or recognizer.model is None:
        raise RuntimeError("Baseline detector or recognizer unavailable")
    review_by_id = {f["id"]: f for f in review["frames"]}
    if len(review_by_id) != len(review["frames"]):
        raise ValueError("Duplicate review frame ID")
    records, groups = [], defaultdict(lambda: {"baseline": [], "candidate": []})
    started = time.perf_counter()
    for index, frame in enumerate(manifest["frames"], 1):
        ai = review_by_id[frame["id"]]
        path = (pack / frame["image"]).resolve()
        if pack not in path.parents or sha256(path) != frame["image_sha256"].lower():
            raise ValueError("Image path or SHA256 mismatch")
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (frame["height"], frame["width"]):
            raise ValueError("Invalid image dimensions")
        detected = detector.detect(image)
        recognizer.recognize(image, detected)
        baseline = [{"bbox_xyxy": list(p.bbox_xyxy), "confidence": p.confidence,
                     "label": p.label, "recognition_confidence": p.recognition_confidence} for p in detected]
        candidate = ai["detector_proposals"]
        scores = {"baseline": score_frame(frame, ai, baseline), "candidate": score_frame(frame, ai, candidate)}
        for model, score in scores.items():
            groups[frame["source_group"]][model].append(score)
        records.append({"frame_id": frame["id"], "image_sha256": frame["image_sha256"],
                        "source_group": frame["source_group"], "scores": scores,
                        "baseline_predictions": baseline, "candidate_predictions": candidate})
        if index % 50 == 0 or index == len(manifest["frames"]):
            print(json.dumps({"processed": index, "total": len(manifest["frames"])}), flush=True)
    by_clip = {name: {model: summarize(values[model]) for model in ("baseline", "candidate")}
               for name, values in sorted(groups.items())}
    result = {"status": "COMPLETED", "split": "TEST", "frames": len(records), "clips": len(by_clip),
              "ground_truth_boxes": sum(len(f["signs"]) for f in manifest["frames"]),
              "metrics": {model: summarize([r["scores"][model] for r in records])
                          for model in ("baseline", "candidate")},
              "lane_proposals": {"frames_with_proposals": sum(r["scores"]["candidate"]["lane_proposed"] for r in records),
                                 "frames": len(records), "metric": "proposal rate; no human lane GT or lane F1"},
              "per_clip": by_clip, "bootstrap": bootstrap(by_clip),
              "limitations": ["Canonical boxes are source annotations, not independent human ground truth.",
                              "Recognizer accuracy compares matched detection crops against AI pseudo-labels from the same recognizer; eligible labels were selected using candidate detector agreement, so the result is circular and potentially biased.",
                              "Clip IDs are inferred from canonical image source names; original videos and timestamps were not verified.",
                              "One clip contains only one selected frame; cluster bootstrap uses only 11 clip groups."],
              "provenance": {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                             "candidate_inference_head": review["provenance"]["git_head"],
                             "manifest_sha256": sha256(manifest_path), "ai_review_sha256": sha256(review_path),
                             "config_sha256": sha256(config_path), "model_sha256": hashes,
                             "recognizer_reference_gate_sha256": sha256(reference) if reference else None,
                             "confidence": cfg["sign_detector"]["confidence_threshold"],
                             "nms_iou": cfg["sign_detector"]["iou_threshold"], "match_iou": 0.5,
                             "matching": "maximum-cardinality Hungarian, IoU tie-break",
                             "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                             "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                             "command": sys.argv, "elapsed_seconds": time.perf_counter() - started}}
    result["lane_proposals"]["rate"] = result["lane_proposals"]["frames_with_proposals"] / len(records)
    output.mkdir(parents=True)
    with (output / "frames.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    result["provenance"]["frames_jsonl_sha256"] = sha256(output / "frames.jsonl")
    (output / "benchmark.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--ai-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.pack, args.ai_review, args.output)
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
