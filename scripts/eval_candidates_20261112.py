"""A2 evaluation of one trained candidate against the current detector, under the frozen reporting protocol.

For each model: VAL and TEST metrics at confidence 0.35 / NMS 0.45 / match IoU 0.5, recall by GT short side
(<16, 16-32, 32-64, >=64), mAP50 and mAP50-95, and the paired image-level bootstrap (1000 resamples, seed
20261112) of the candidate minus the current detector on TEST for recall and mAP50. The current detector is
always evaluated in the same run, so the pair shares one code path.

Selection is NOT done here. Timing is NOT done here (GPU shared with the training queue); see the timing script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN_RUNNER = ROOT / "output/kiet_wp_20261026/A/run_wp3_benchmarks.py"
CURRENT = ROOT / "models/sign_detector/finetuned_20261026.pt"
OUT = ROOT / "output/bench_20261112/eval_candidates"
BOOT_ITER = 1000
BOOT_SEED = 20261112


def load_frozen_runner():
    spec = importlib.util.spec_from_file_location("wp3_frozen_runner", FROZEN_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_records(module, weights: Path, split: str):
    samples = pd.read_csv(module.LOC / "samples.csv", keep_default_na=False)
    labels = pd.read_csv(module.AUDIT / "semantic_labels_v1_1.csv", keep_default_na=False)
    samples = samples[samples["split"] == split].reset_index(drop=True)
    labels = labels[labels["split"] == split]
    per_image, _ = module.get_detector_predictions(samples, labels, weights)
    return per_image


def paired_bootstrap(module, base, cand):
    rng = np.random.default_rng(BOOT_SEED)
    count = len(base)
    draws = rng.integers(0, count, size=(BOOT_ITER, count), endpoint=False)
    deltas = {"recall": [], "map50": []}
    for indices in draws:
        b = module.metrics_from_images([base[int(i)] for i in indices], include_map5095=False)
        c = module.metrics_from_images([cand[int(i)] for i in indices], include_map5095=False)
        deltas["recall"].append((c["recall"] or 0.0) - (b["recall"] or 0.0))
        deltas["map50"].append(c["map50"] - b["map50"])
    out = {}
    for key, values in deltas.items():
        lo, hi = np.quantile(np.asarray(values), [0.025, 0.975])
        out[key] = {"ci95": [float(lo), float(hi)], "ci_excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    weights = args.weights.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    module = load_frozen_runner()

    def cached(path_weights, split, tag):
        cache = OUT / f"records_{tag}_{split.lower()}.pkl"
        if cache.is_file():
            return pickle.loads(cache.read_bytes())
        records = split_records(module, path_weights, split)
        module.build_ap_records(records)
        cache.write_bytes(pickle.dumps(records))
        return records

    result = {"name": args.name, "weights": str(weights.relative_to(ROOT)).replace("\\", "/"),
              "weights_sha256": module.sha(weights), "protocol": "confidence 0.35, NMS 0.45, match IoU 0.5"}
    for split in ("VAL", "TEST"):
        records = cached(weights, split, args.name)
        metrics = module.metrics_from_images(records)
        result[split] = {k: metrics[k] for k in ("precision", "recall", "map50", "map50_95", "tp", "fp", "fn")}
        result[split]["recall_by_gt_size"] = {k: v["recall_by_gt_size"] for k, v in metrics["size_groups"].items()}
    base_records = cached(CURRENT, "TEST", "current_finetune_20261026")
    base_metrics = module.metrics_from_images(base_records)
    result["current_TEST"] = {k: base_metrics[k] for k in ("precision", "recall", "map50", "tp", "fp", "fn")}
    cand_records = cached(weights, "TEST", args.name)
    result["paired_bootstrap_TEST_candidate_minus_current"] = paired_bootstrap(module, base_records, cand_records)
    (OUT / f"{args.name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("name", "VAL", "TEST")}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())
