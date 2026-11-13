"""A7 analysis: two blind labeling passes, agreement, and static-pipeline evaluation on the 239-frame pack.

Reference codes = boxes where both independent passes gave the same sign code (AI-reviewed reference, not
human ground truth). Detection reference = the 327 pack boxes (AI-proposed, circular for the detector family
that proposed them; reported with that caveat).

Pipeline = production still-image path: SignDetector (configs/perception.yaml) then SignRecognizer, with the
configured thresholds. Nothing is tuned on these frames.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
import sys
from collections import Counter

import cv2
import numpy as np
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PACK = Path("D:/DSP391m/project/review_pack_20261019_239_proposals")
A7 = ROOT / "output/bench_20261112/A7"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])  # noqa: E731
    return inter / max(area(a) + area(b) - inter, 1e-12)


def bucket(side):
    if side < 32:
        return "16-32"
    if side < 64:
        return "32-64"
    return ">=64"


def load_pass(name):
    rows = {r["id"]: r for r in csv.DictReader((A7 / f"{name}_labels.csv").open(encoding="utf-8"))}
    key = json.loads((A7 / "sealed" / f"{name}_key.json").read_text(encoding="utf-8"))
    return {key_id: {**key[key_id], "label": rows[key_id]["label"]} for key_id in key}


def main() -> None:
    p1 = load_pass("pass1")
    p2 = load_pass("pass2")
    # Match passes by the shared box identity (frame_id + box_id)
    by_box = {}
    for side, data in (("p1", p1), ("p2", p2)):
        for entry in data.values():
            by_box.setdefault((entry["frame_id"], entry["box_id"]), {})[side] = entry
    pairs = [(k, v["p1"], v["p2"]) for k, v in by_box.items() if "p1" in v and "p2" in v]
    sizes = {k: min(e["bbox_xyxy"][2] - e["bbox_xyxy"][0], e["bbox_xyxy"][3] - e["bbox_xyxy"][1]) for k, e, _ in pairs}
    report = {"boxes_labeled_by_both_passes": len(pairs)}
    report["exact_label_agreement"] = sum(e["label"] == f["label"] for _, e, f in pairs) / len(pairs)
    code_like = lambda s: s not in ("uncertain", "not_sign")  # noqa: E731
    report["both_code_and_equal"] = sum(code_like(e["label"]) and e["label"] == f["label"] for _, e, f in pairs)
    report["kappa_exact_labels_by_size"] = {}
    for b in ("16-32", "32-64", ">=64"):
        subset = [(e, f) for k, e, f in pairs if bucket(sizes[k]) == b]
        if subset:
            a = [e["label"] for e, _ in subset]
            c = [f["label"] for _, f in subset]
            report["kappa_exact_labels_by_size"][b] = {
                "n": len(subset), "agreement": sum(x == y for x, y in zip(a, c)) / len(subset),
                "kappa": float(cohen_kappa_score(a, c)) if len(set(a + c)) > 1 else None}
    reference = {k: e["label"] for k, e, f in pairs if code_like(e["label"]) and e["label"] == f["label"]}
    report["reference_code_boxes"] = len(reference)
    report["reference_code_distribution"] = dict(Counter(reference.values()))

    # Pack boxes (detection reference) and frames
    manifest = json.loads((PACK / "review_manifest.json").read_text(encoding="utf-8"))
    boxes_by_frame = {}
    for frame in manifest["frames"]:
        signs = frame["signs"]
        signs = ast.literal_eval(signs) if isinstance(signs, str) else signs
        boxes_by_frame[frame["id"]] = (frame["image"], [(b["id"], [float(v) for v in b["bbox_xyxy"]]) for b in signs])

    from src.utils.config import load_config
    from src.perception.sign_detector import SignDetector
    from src.perception.sign_recognizer import SignRecognizer

    cfg = load_config("perception")
    detector = SignDetector(cfg)
    recognizer = SignRecognizer(cfg)
    if detector._model is None or recognizer.model is None:
        raise RuntimeError("production detector or recognizer unavailable")
    tp = fp = fn = 0
    code_total = code_correct = code_matched = 0
    per_size = {}
    for frame_id, (image_rel, pack_boxes) in boxes_by_frame.items():
        image = cv2.imread(str(PACK / image_rel))
        detections = detector.detect(image)
        recognizer.recognize(image, detections)
        preds = [d for d in detections]
        used = set()
        matched_pairs = []
        for bid, box in pack_boxes:
            best, best_j = 0.0, None
            for j, d in enumerate(preds):
                if j in used:
                    continue
                score = iou(box, list(d.bbox_xyxy))
                if score > best:
                    best, best_j = score, j
            if best_j is not None and best >= 0.5:
                used.add(best_j)
                tp += 1
                matched_pairs.append((bid, preds[best_j]))
            else:
                fn += 1
        fp += len(preds) - len(used)
        for bid, det in matched_pairs:
            if (frame_id, bid) in reference:
                code_matched += 1
                side = bucket(sizes.get((frame_id, bid), 0) or 0)
                per_size.setdefault(side, [0, 0])
                per_size[side][1] += 1
                if det.label == reference[(frame_id, bid)]:
                    code_correct += 1
                    per_size[side][0] += 1
    code_total = len(reference)
    report["detection_vs_pack_boxes"] = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "pipeline": "SignDetector (configs/perception.yaml: finetuned_20261026.pt, conf 0.35, NMS 0.45, IoU 0.5 match)",
        "reference_caveat": "pack boxes are AI proposals from detector-era review; circular for that detector family"}
    report["end_to_end_code_accuracy"] = {
        "reference_code_boxes": code_total, "matched_by_detector": code_matched,
        "correct_code": code_correct,
        "accuracy_over_reference_boxes": code_correct / code_total if code_total else None,
        "recognizer": "SignRecognizer (retrained_20261028.pt, production threshold; no video threshold)",
        "by_size_matched": {k: {"correct": v[0], "matched": v[1]} for k, v in per_size.items()}}
    report["scope_note"] = ("AI-reviewed reference from two rule-free passes by the same reviewer; "
                            "not human ground truth; candidate detector comparison deferred until A2 finishes.")
    (A7 / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
