"""A1 baseline check: re-evaluate a sign detector on VAL and TEST with the reporting protocol.

Protocol (fixed before running): confidence 0.35, NMS IoU 0.45, match IoU 0.5, one class,
image size 640, batch 1. The matching and AP code is reused unchanged from the frozen WP3
runner (output/kiet_wp_20261026/A/run_wp3_benchmarks.py) so that numbers stay comparable.

The script never writes to that runner's output folder; results go to --output.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN_RUNNER = ROOT / "output/kiet_wp_20261026/A/run_wp3_benchmarks.py"
DEFAULT_WEIGHTS = ROOT / "models/sign_detector/finetuned_20261026.pt"
DEFAULT_OUTPUT = ROOT / "output/bench_20261112/A1"
REPORTED_TEST = {"map50": 0.9357, "recall": 0.8754}
TOLERANCE = 0.002


def load_frozen_runner():
    spec = importlib.util.spec_from_file_location("wp3_frozen_runner", FROZEN_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(module, weights: Path, split: str) -> dict:
    loc = module.LOC
    samples_all = pd.read_csv(loc / "samples.csv", keep_default_na=False)
    labels_all = pd.read_csv(module.AUDIT / "semantic_labels_v1_1.csv", keep_default_na=False)
    samples = samples_all[samples_all["split"] == split].reset_index(drop=True)
    labels = labels_all[labels_all["split"] == split]
    per_image, infer_seconds = module.get_detector_predictions(samples, labels, weights)
    metrics = module.metrics_from_images(per_image)
    return {
        "split": split,
        "weights": str(weights.relative_to(ROOT)).replace("\\", "/"),
        "weights_sha256": module.sha(weights),
        "images": len(samples),
        "inference_seconds": infer_seconds,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", nargs="+", default=["VAL", "TEST"])
    args = parser.parse_args()
    weights = args.weights.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    module = load_frozen_runner()
    results = {}
    for split in args.splits:
        result = evaluate(module, weights, split)
        results[split] = result
        (output / f"detector_{weights.stem}_{split.lower()}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"split": split, "precision": result["metrics"]["precision"],
                          "recall": result["metrics"]["recall"], "map50": result["metrics"]["map50"]}), flush=True)
    check = {}
    if "TEST" in results:
        test = results["TEST"]["metrics"]
        check = {name: {"reported": ref, "measured": test[name], "abs_diff": abs(test[name] - ref),
                        "within_tolerance": abs(test[name] - ref) <= TOLERANCE}
                 for name, ref in REPORTED_TEST.items()}
        check["passed"] = all(v["within_tolerance"] for v in check.values() if isinstance(v, dict))
    (output / "a1_check.json").write_text(json.dumps({"tolerance": TOLERANCE, "test_check": check},
                                                     indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"test_check": check}, indent=2), flush=True)
    return 0 if check.get("passed", True) else 2


if __name__ == "__main__":
    sys.exit(main())
