"""Render Report 3 figures from the local 2026-10-26 revalidation artifacts.

Inputs are local JSON/CSV files under output/ (ignored by Git); see
docs/report3/REPORT3_MODEL_RESULTS.md for their SHA256 references.
Usage: python scripts/make_report3_figures.py
"""
from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "output/kiet_wp_20261026/A"
D = ROOT / "output/kiet_wp_20261026/D"
TRAIN_CSV = ROOT / "output/training/detector_finetuned/results.csv"
OUT = ROOT / "docs/report3/figures"
MODELS = (("baseline", "Baseline (retrained_20261009)", "#8c8c8c"),
          ("best_selected", "Fine-tuned (finetuned_20261026)", "#1f6fb2"))
SIZES = ("<16", "16-32", "32-64", ">=64")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def detector_metrics():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    names = ("precision", "recall", "map50", "map50_95")
    labels = ("Precision", "Recall", "mAP50", "mAP50-95")
    x = np.arange(len(names))
    for ax, split in zip(axes, ("val", "test")):
        for i, (key, label, color) in enumerate(MODELS):
            ev = load(A / f"detector_{key}_{split}.json")["evaluation"]
            vals = [ev[n] for n in names]
            bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, label=label, color=color)
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)
        ax.set_xticks(x, labels)
        ax.set_ylim(0.6, 1.0)
        ax.set_title(f"{split.upper()} (conf 0.35, IoU 0.50)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8, loc="lower left")
    save(fig, "fig1_detector_baseline_vs_finetuned.png")


def detector_recall_by_size():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(SIZES))
    for i, (key, label, color) in enumerate(MODELS):
        groups = load(A / f"detector_{key}_test.json")["evaluation"]["size_groups"]
        vals = [groups[s]["recall_by_gt_size"] for s in SIZES]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)
    counts = load(A / "detector_baseline_test.json")["evaluation"]["size_groups"]
    ax.set_xticks(x, [f"{s} px\n(n={counts[s]['gt_count']})" for s in SIZES])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Recall")
    ax.set_xlabel("GT box short side")
    ax.set_title("Detector recall by sign size - TEST")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig2_detector_recall_by_size.png")


def detector_ap_by_iou():
    report = load(A / "detector_map50_95.json")["splits"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    styles = {"VAL": "--", "TEST": "-"}
    for split in ("VAL", "TEST"):
        for key, label, color in MODELS:
            ap = report[split]["models"][key]["ap_by_iou"]
            ious = [float(k) for k in ap]
            ax.plot(ious, list(ap.values()), styles[split], marker="o", ms=3,
                    color=color, label=f"{label.split(' (')[0]} - {split}")
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("AP")
    ax.set_title("Detector AP across IoU thresholds")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "fig3_detector_ap_by_iou.png")


def training_curve():
    with TRAIN_CSV.open(encoding="utf-8") as stream:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(stream)]
    epochs = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for col, label in (("metrics/mAP50-95(B)", "VAL mAP50-95"),
                       ("metrics/recall(B)", "VAL recall"),
                       ("metrics/precision(B)", "VAL precision")):
        ax.plot(epochs, [float(r[col]) for r in rows], marker="o", ms=3, label=label)
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Score (Ultralytics val)")
    ax.set_title("Detector fine-tuning on VAL (10 epochs, AdamW lr0=1e-3)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "fig4_detector_finetuning_curve.png")


def recognizer_by_size():
    res = load(D / "recognizer_size_groups.json")["results"]
    groups = [("VAL", ">=32")] + [("TEST", s) for s in SIZES]
    labels = [f"{sp} {g}\n(n={res[sp][g]['crop_count']})" for sp, g in groups]
    acc = [res[sp][g]["accuracy"] for sp, g in groups]
    f1 = [res[sp][g]["macro_f1_supported_classes"] for sp, g in groups]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, (vals, label, color) in enumerate(((acc, "Accuracy", "#1f6fb2"),
                                              (f1, "Macro-F1", "#e08a2c"))):
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_title("Recognizer on GT crops by crop short side (px)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig5_recognizer_by_crop_size.png")


def recognizer_confusion():
    with (D / "recognizer_test_ge32_confusion_matrix.csv").open(encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    cols = rows[0][1:]
    names = [r[0] for r in rows[1:]]
    mat = np.array([[int(v) for v in r[1:]] for r in rows[1:]], dtype=float)
    keep = [i for i, n in enumerate(names) if mat[i].sum() > 0]
    mat = mat[keep]
    names = [names[i] for i in keep]
    norm = mat / mat.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title("Recognizer confusion matrix, TEST crops >=32 px (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.04)
    save(fig, "fig6_recognizer_confusion_test_ge32.png")


def final_evidence():
    evidence = ROOT / "output/kiet_wp_20261030"
    review = load(evidence / "second_review/results.json")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    modes = (("claude", "Claude"), ("sol", "Sol"), ("consensus", "Consensus subset"))
    colors = ("#666666", "#2267a5", "#ba731d")
    for ax, (population, title) in zip(axes, (("hanoi", "Hanoi"), ("v0", "V0"), ("combined", "Combined"))):
        for offset, ((mode, label), color) in enumerate(zip(modes, colors)):
            values = [review["field_precision"][population][mode][c] for c in "ABC"]
            y = np.array([r["precision"] for r in values])
            bounds = np.array([r["ci95"] for r in values]).T
            ax.errorbar(np.arange(3) + (offset-1)*.16, y,
                        yerr=np.maximum(0, np.vstack((y-bounds[0], bounds[1]-y))),
                        fmt="o", capsize=3, color=color, label=label)
        ax.set_xticks(range(3), ["A: old .63", "B: v2 .87", "C: v2 T2"], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
        ax.set_ylim(.4, 1.04)
    axes[0].set_ylabel("Precision among decidable AI labels")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Field emitted-sign precision, 95% segment bootstrap CI; consensus is selected")
    save(fig, "fig7_final_field_precision.png")

    operating = load(evidence / "operating_points.json")["results"]
    names = list(operating)
    rows = [next(r for r in operating[c]["rows"] if r["category"] == "old30" and r["size"] == ">=32") for c in names]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (key, label) in enumerate((("coverage", "Coverage"), ("accepted_accuracy", "Correct / accepted"), ("end_to_end", "Correct / all crops"))):
        bars = ax.bar(np.arange(len(rows)) + (i-1)*.24, [r[key] for r in rows], .24, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(range(len(names)), ["Old .63", "Old .63 + gate", "V2 .87", "V2 T2"])
    ax.set_ylim(0, 1.13)
    ax.set_title(f"Operating points: old 30 codes, TEST >=32 px, n={rows[0]['n']}")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig8_final_operating_points.png")

    sources = list("abcde")
    rows = [review["agreement_by_source_size"][s]["all"] for s in sources]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (key, label) in enumerate((("agreement", "Exact label agreement"), ("kappa", "Cohen kappa"))):
        bars = ax.bar(np.arange(5) + (i-.5)*.36, [r[key] for r in rows], .36, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(range(5), [f"{s}: {title}\n(n={r['n']})" for s,title,r in zip(sources, ("Hanoi", "V0", "Extension TEST", "TRAIN/VAL", "Owner pack"), rows)], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_title("Claude-Sol agreement; uncertain is a category, labels are AI review")
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig9_final_reviewer_agreement.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-evidence", action="store_true", help="Generate only the three 2026-10-30 final-evidence figures")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.final_evidence:
        final_evidence()
        print("Generated final-evidence figures 7-9 from JSON")
        return
    detector_metrics()
    detector_recall_by_size()
    detector_ap_by_iou()
    training_curve()
    recognizer_by_size()
    recognizer_confusion()
    print("\n".join(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
