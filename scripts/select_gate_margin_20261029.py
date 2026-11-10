"""Choose the reference-gate margin on VAL and the held-out 20% of reviewed negatives only (issue #2, WP3).

Rule (owner brief): the margin with the highest block rate on held-out negatives such that the VAL macro-F1 over
the main classes drops by at most 0.005 at both operating points (T2 and 0.87). TEST is never read here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MIN_SIMILARITY = 0.80
MAX_F1_DROP = 0.005


def gate_scores(embeddings, labels, bank):
    """Best negative and best same-label positive similarity per sample, as FrozenReferenceGate.verdict computes them.

    Returns (negative, positive) float arrays; NaN where the gate abstains (label 'unknown' or no references).
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    negative_bank, positive_bank = np.asarray(bank["negative"], dtype=np.float32), np.asarray(bank["positive"], dtype=np.float32)
    bank_labels, targets = np.asarray(bank["labels"]), np.asarray(bank["negative_targets"])
    labels = np.asarray(labels)
    neg = np.full(len(labels), np.nan, dtype=np.float32)
    pos = np.full(len(labels), np.nan, dtype=np.float32)
    for label in np.unique(labels):
        if label == "unknown":
            continue
        selected = np.flatnonzero(bank_labels == label)
        usable = np.flatnonzero((targets == "*") | (targets == label))
        rows = np.flatnonzero(labels == label)
        if not len(selected) or not len(usable) or not len(rows):
            continue
        neg[rows] = (embeddings[rows] @ negative_bank[usable].T).max(1)
        pos[rows] = (embeddings[rows] @ positive_bank[selected].T).max(1)
    return neg, pos


def rejected(neg, pos, margin, min_similarity=MIN_SIMILARITY):
    with np.errstate(invalid="ignore"):
        return (neg >= min_similarity) & (neg > pos + margin)


def macro_f1(y, pred, main_index):
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, labels=main_index, average="macro", zero_division=0))


def operating_point(logits, y, neg, pos, margin, threshold, unknown_index, main_index):
    """VAL macro-F1 and coverage at one threshold with the gate at `margin` (None = no gate)."""
    shifted = logits - logits.max(1, keepdims=True)
    prob = np.exp(shifted) / np.exp(shifted).sum(1, keepdims=True)
    pred, conf = prob.argmax(1), prob.max(1)
    veto = np.zeros(len(y), dtype=bool) if margin is None else rejected(neg, pos, margin)
    accepted = (pred != unknown_index) & (conf >= threshold) & ~veto
    out = np.where(accepted, pred, unknown_index)
    known = y != unknown_index
    return {"macro_f1_main": macro_f1(y, out, main_index), "coverage_known": float(accepted[known].mean()),
            "accepted_accuracy_known": float((out[known & accepted] == y[known & accepted]).mean()) if (known & accepted).any() else None}


def sweep(val, held, thresholds, unknown_index, main_index, margins):
    """Per margin: F1 drop at each threshold and block rates on the held-out negatives."""
    base = {name: operating_point(val["logits"], val["y"], val["neg"], val["pos"], None, t, unknown_index, main_index)
            for name, t in thresholds.items()}
    rows = []
    for margin in margins:
        row = {"margin": round(float(margin), 4), "points": {}}
        for name, t in thresholds.items():
            gated = operating_point(val["logits"], val["y"], val["neg"], val["pos"], margin, t, unknown_index, main_index)
            row["points"][name] = {**gated, "f1_drop": base[name]["macro_f1_main"] - gated["macro_f1_main"]}
        veto = rejected(held["neg"], held["pos"], margin)
        row["block_rate_heldout"] = float(veto.mean()) if len(veto) else None
        for name, t in thresholds.items():
            accepted = (held["pred"] != unknown_index) & (held["conf"] >= t)
            row.setdefault("blocked_of_accepted_heldout", {})[name] = (float((veto & accepted).sum() / accepted.sum()) if accepted.any() else None,
                                                                         int(accepted.sum()))
        rows.append(row)
    return base, rows


def choose_margin(rows, max_drop=MAX_F1_DROP):
    """Highest held-out block rate whose F1 drop is within `max_drop` at every operating point; ties go to the larger margin."""
    feasible = [r for r in rows if all(p["f1_drop"] <= max_drop + 1e-12 for p in r["points"].values())
                and r["block_rate_heldout"] is not None]
    if not feasible:
        return None
    return max(feasible, key=lambda r: (r["block_rate_heldout"], r["margin"]))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import torch
    from scripts.build_sign_reference_gate import negative_crop
    from scripts.train_recognizer_v2_20261028 import OUT as V2, load_split
    from src.perception.recognizer_v2_crops import exact_crop
    from src.perception.sign_recognizer import crop_tensor
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recognizer", type=Path, default=ROOT / "models/sign_recognizer/retrained_20261028.pt")
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--negatives-jsonl", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--t2", type=float)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    release = json.loads((ROOT / "docs/release_gate_v2.json").read_text(encoding="utf-8"))
    t2 = args.t2 or release["recognizer_v2_crosscheck"]["alternative_threshold_val_only"]["threshold"]
    checkpoint = torch.load(args.recognizer, map_location="cpu", weights_only=True)
    classes, main_classes = checkpoint["classes"], checkpoint["main_classes"]
    from torchvision.models import mobilenet_v3_small
    model = mobilenet_v3_small(num_classes=len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    size = checkpoint.get("image_size", 128)

    def embed_logits(crops, batch=128):
        emb, logit = [], []
        for start in range(0, len(crops), batch):
            x = torch.stack([crop_tensor(c, size) for c in crops[start:start + batch]]).to(device)
            with torch.inference_mode():
                feat = model.avgpool(model.features(x)).flatten(1)
                pre = model.classifier[:-1](feat)
                logit.append(model.classifier[-1](pre).float().cpu())
                emb.append(torch.nn.functional.normalize(pre, dim=1).cpu())
        return torch.cat(emb).numpy(), torch.cat(logit).numpy()

    boxes = pd.read_csv(V2 / "boxes.csv", keep_default_na=False)
    background = pd.read_csv(V2 / "background_boxes.csv", keep_default_na=False)
    rows, entries = load_split(V2, "VAL", classes, boxes, background)          # VAL crops >= 16 px plus VAL background
    val_emb, val_logits = embed_logits([exact_crop(e) for e in entries])
    y = rows.code.map(classes.index).to_numpy()
    bank = dict(np.load(args.gate, allow_pickle=False))
    unknown = classes.index("unknown")
    main_index = [classes.index(c) for c in main_classes]
    pred_labels = np.array(classes)[val_logits.argmax(1)]
    val_neg, val_pos = gate_scores(val_emb, pred_labels, bank)
    held_records = [json.loads(l) for l in args.negatives_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    held_records = [r for r in held_records if r["label"] == "not_a_sign" and r["split"] == "heldout"]
    held_emb, held_logits = embed_logits([negative_crop(r, args.video_root) for r in held_records])
    held_pred = held_logits.argmax(1)
    shifted = held_logits - held_logits.max(1, keepdims=True)
    held_conf = (np.exp(shifted) / np.exp(shifted).sum(1, keepdims=True)).max(1)
    held_neg, held_pos = gate_scores(held_emb, np.array(classes)[held_pred], bank)
    thresholds = {"T2": t2, "0.87": 0.87}
    margins = np.round(np.arange(0.0, 0.3001, 0.005), 4)
    base, table = sweep({"logits": val_logits, "y": y, "neg": val_neg, "pos": val_pos},
                        {"neg": held_neg, "pos": held_pos, "pred": held_pred, "conf": held_conf}, thresholds, unknown, main_index, margins)
    chosen = choose_margin(table)
    report = {"split": "VAL", "heldout_negatives": len(held_records), "val_crops": int(len(y)), "thresholds": thresholds,
              "min_similarity": MIN_SIMILARITY, "max_f1_drop": MAX_F1_DROP, "no_gate": base, "chosen": chosen, "sweep": table,
              "test_used": False, "provenance": {"gate_sha256": sha256_file(args.gate), "recognizer_sha256": sha256_file(args.recognizer),
                                                 "negatives_sha256": sha256_file(args.negatives_jsonl)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"no_gate": base, "chosen": chosen}, indent=1))


if __name__ == "__main__":
    main()
