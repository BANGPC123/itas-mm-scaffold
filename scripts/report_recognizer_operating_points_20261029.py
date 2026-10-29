"""Report thresholded TEST crop metrics; choose the alternative threshold on VAL only."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/kiet_wp_20261028/recognizer_v2"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(truth, prediction, confidence, threshold, rejected=None):
    truth, prediction, confidence = np.asarray(truth), np.asarray(prediction), np.asarray(confidence)
    if len(truth) != len(prediction) or len(truth) != len(confidence):
        raise ValueError("Input arrays must have equal length")
    rejected = np.zeros(len(truth), dtype=bool) if rejected is None else np.asarray(rejected, dtype=bool)
    if len(rejected) != len(truth):
        raise ValueError("Gate verdict count does not match predictions")
    accepted = (prediction != "unknown") & (confidence >= threshold) & ~rejected
    correct = accepted & (prediction == truth)
    n, a, c = len(truth), int(accepted.sum()), int(correct.sum())
    return {"n": n, "accepted": a, "correct": c, "coverage": a / n if n else None,
            "accepted_accuracy": c / a if a else None, "end_to_end": c / n if n else None,
            "wrong_label_rate": (a - c) / n if n else None}


def select_threshold(logits, truth, unknown_index, min_coverage=0.95, min_accuracy=0.97):
    logits, truth = np.asarray(logits, dtype=np.float64), np.asarray(truth)
    shifted = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(shifted); prob /= prob.sum(axis=1, keepdims=True)
    pred = prob.argmax(axis=1); confidence = prob.max(axis=1)
    known = truth != unknown_index; background = ~known
    if not known.any():
        raise ValueError("VAL contains no known-class samples")
    candidates = np.unique(np.concatenate(([0.0], confidence[(pred != unknown_index) & known])))
    rows = []
    for threshold in candidates:
        accepted = (pred != unknown_index) & (confidence >= threshold)
        selected = known & accepted
        n = int(selected.sum())
        rows.append({"threshold": float(threshold), "coverage": float(n / known.sum()),
                     "accepted_accuracy": float((pred[selected] == truth[selected]).mean()) if n else None,
                     "background_accepted": float(accepted[background].mean()) if background.any() else None})
    eligible = [r for r in rows if r["coverage"] >= min_coverage]
    if eligible:
        result = max(eligible, key=lambda r: r["threshold"]); result["criterion"] = "coverage_target"
    else:
        eligible = [r for r in rows if r["accepted_accuracy"] is not None and r["accepted_accuracy"] >= min_accuracy]
        if not eligible:
            return {"threshold": None, "criterion": "no_feasible_threshold", "known_samples": int(known.sum())}
        result = max(eligible, key=lambda r: (r["coverage"], r["threshold"])); result["criterion"] = "fallback_accuracy"
    result["known_samples"] = int(known.sum())
    result["background_samples"] = int(background.sum())
    return result


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=BASE / "eval/test_predictions.csv")
    parser.add_argument("--val-logits", type=Path, default=BASE / "train/val_logits.npz")
    parser.add_argument("--old-gate-verdicts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (ROOT / "output").resolve() not in args.output.resolve().parents:
        raise ValueError("Report output must be inside the repository output directory")
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(ROOT / "models/sign_recognizer/retrained_20261009.pt", map_location="cpu", weights_only=True)
    old_classes = set(checkpoint["classes"]) - {"unknown"}
    info = json.loads((BASE / "class_list.json").read_text(encoding="utf-8"))
    old_count = len(old_classes)
    if old_count != 30:
        raise ValueError("Expected 30 old known classes")
    with np.load(args.val_logits, allow_pickle=False) as val:
        sizes, logits, labels = val["short_side"], val["logits"], val["y"]
        keep = sizes >= 16
        alternative = select_threshold(logits[keep], labels[keep], info["classes"].index("unknown"))
    frame = pd.read_csv(args.predictions, keep_default_na=False)
    if frame.canonical_box_id.duplicated().any():
        raise ValueError("Duplicate box IDs")
    gate = None
    if args.old_gate_verdicts:
        verdict = pd.read_csv(args.old_gate_verdicts, keep_default_na=False)
        if verdict.canonical_box_id.duplicated().any() or set(verdict.canonical_box_id) != set(frame.canonical_box_id):
            raise ValueError("Gate verdicts do not match TEST predictions")
        frame = frame.merge(verdict[["canonical_box_id", "rejected"]], on="canonical_box_id", validate="one_to_one")
        gate = frame.rejected.astype(str).str.lower().map({"true": True, "false": False})
        if gate.isna().any():
            raise ValueError("Invalid gate verdict")
    categories = {"old30": frame.code.isin(old_classes),
                  "new_main": frame.code.isin(info["main"]) & ~frame.code.isin(old_classes),
                  "low_support": frame.code.isin(info["low_support"]) & ~frame.code.isin(old_classes)}
    sizes = {"<16": frame.short_side_px < 16, "16-32": frame.short_side_px.ge(16) & frame.short_side_px.lt(32),
             "32-64": frame.short_side_px.ge(32) & frame.short_side_px.lt(64),
             ">=64": frame.short_side_px.ge(64), ">=32": frame.short_side_px.ge(32)}
    configs = {"old_063_no_gate": ("old", 0.63, None), "old_063_gate": ("old", 0.63, gate),
               "new_087_no_gate": ("new", 0.87, None),
               "new_t2_no_gate": ("new", alternative["threshold"], None)}
    results = {}
    for name, (model, threshold, verdict) in configs.items():
        if threshold is None or (name == "old_063_gate" and verdict is None):
            results[name] = {"status": "BLOCKED", "reason": "Gate verdicts or feasible VAL threshold unavailable"}
            continue
        entries = []
        for category, catmask in categories.items():
            for size, sizemask in sizes.items():
                mask = catmask & sizemask; sub = frame[mask]
                value = metrics(sub.code, sub[f"{model}_pred"], sub[f"{model}_conf"], threshold,
                                verdict[mask] if verdict is not None else None)
                entries.append({"category": category, "size": size, **value})
        results[name] = {"status": "COMPLETED", "threshold": threshold, "rows": entries}
    output = {"selection": {"split": "VAL", "min_short_side": 16, "test_used": False, **alternative},
              "results": results,
              "limitations": ["Crop-level scores against AI-assisted labels, not human ground truth.",
                              "Raw GT crops; runtime also applies size and semantic checks not captured here."],
              "provenance": {"predictions_sha256": sha256(args.predictions), "val_logits_sha256": sha256(args.val_logits),
                             "old_gate_verdicts_sha256": sha256(args.old_gate_verdicts) if args.old_gate_verdicts else None,
                             "class_list_sha256": sha256(BASE / "class_list.json"),
                             "old_model_sha256": sha256(ROOT / "models/sign_recognizer/retrained_20261009.pt"),
                             "new_model_sha256": sha256(ROOT / "models/sign_recognizer/retrained_20261028.pt"),
                             "old_gate_sha256": (sha256(ROOT / "models/sign_recognizer/reference_gate_20261011_final.npz")
                                                 if (ROOT / "models/sign_recognizer/reference_gate_20261011_final.npz").is_file() else None),
                             "script_sha256": sha256(Path(__file__)), "command": sys.argv,
                             "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"val_selection": output["selection"], "config_status": {k: v["status"] for k, v in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
