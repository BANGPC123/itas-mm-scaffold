"""A6: evaluate sign-aware retrieval variants on the sign-definition set and the 24 original cases.

Selection uses DEV only (boost grid, best dev Recall@1, tie-break dev MRR). HOLDOUT is reported
for every variant once, with the boost chosen on dev; it is not used to choose anything.
Variants are opt-in flags in src/reasoning/sign_retrieval_20261112.py; defaults stay off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.reasoning.sign_retrieval_20261112 import SignAwareLegalIndex  # noqa: E402

CORPUS = ROOT / "data/regulations/corpus.json"
SIGN_SET = ROOT / "data/evaluation/legal_retrieval_signs_20261112.json"
ORIGINAL = ROOT / "data/evaluation/legal_retrieval.json"
OUT = ROOT / "output/bench_20261112/A6/retrieval_eval.json"
BOOSTS = [0.25, 0.5, 1.0, 2.0]


def run_cases(index, cases):
    rows = []
    for case in cases:
        expected = {(r["document_id"], r["locator"]) for r in case["expected"]}
        hits = index.search(case["query"], 4)
        rank = next((i for i, h in enumerate(hits, 1) if (h["document_id"], h["locator"]) in expected), None)
        rows.append({"id": case["id"], "split": case.get("split", "original"), "rank": rank})
    return rows


def metrics(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    out = {"n": n}
    for k in (1, 3, 4):
        out[f"recall_at_{k}"] = sum(r["rank"] is not None and r["rank"] <= k for r in rows) / n
    out["mrr"] = sum(1 / r["rank"] if r["rank"] else 0.0 for r in rows) / n
    return out


def build(variant: str, boost: float):
    flags = {"R0": {}, "R1": {"code_boost": True}, "R2": {"definition_chunks": True},
             "R3": {"code_boost": True, "definition_chunks": True}}[variant]
    return SignAwareLegalIndex(CORPUS, boost=boost, **flags)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    signs = json.loads(SIGN_SET.read_text(encoding="utf-8"))["cases"]
    originals = json.loads(ORIGINAL.read_text(encoding="utf-8"))["cases"]
    dev = [c for c in signs if c["split"] == "dev"]
    holdout = [c for c in signs if c["split"] == "holdout"]
    results = {}
    chosen = {}
    plan = [("R0", None), ("R1", None), ("R2", None), ("R3", None)]
    for variant, _ in plan:
        grid = BOOSTS if variant in ("R1", "R3") else [None]
        scored = []
        for boost in grid:
            index = build(variant, boost if boost is not None else 0.5)
            dev_rows = run_cases(index, dev)
            scored.append((boost, metrics(dev_rows), index))
        best = max(scored, key=lambda item: (item[1]["recall_at_1"], item[1]["mrr"]))
        boost_value, dev_metrics, _ = best
        chosen[variant] = {"boost": boost_value, "dev_metrics_by_boost": {
            ("none" if b is None else str(b)): m for b, m, _ in scored}}
        index = build(variant, boost_value if boost_value is not None else 0.5)
        results[variant] = {
            "boost": boost_value,
            "dev": metrics(run_cases(index, dev)),
            "holdout": metrics(run_cases(index, holdout)),
            "original_24": metrics(run_cases(index, originals)),
            "extra_definition_chunks": getattr(index, "extra_chunk_count", 0),
        }
        print(json.dumps({variant: {k: results[variant][k] for k in ("boost", "dev", "holdout")}},
                         ensure_ascii=False), flush=True)
    payload = {
        "protocol": "Selection on DEV only; HOLDOUT reported once per variant; 24 original cases are a regression check.",
        "sign_cases": {"dev": len(dev), "holdout": len(holdout)},
        "boost_grid": BOOSTS,
        "boost_selection": chosen,
        "results": results,
        "defaults": "All variants are off by default in the application (no flags set there).",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"written": str(args.output)}))


if __name__ == "__main__":
    main()
