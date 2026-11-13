"""A5: one-stage 88-class detection labels (87 sign codes + other_sign), split kept as in the canonical set.

Source: data/audit/semantic/final_v1_1/semantic_labels_v1_1.csv, the table the recognizer-v2 manifest was
built from. A box with a code in the 87-class recognizer list gets that code. A box with no code, or a code
outside the 87-class list, gets other_sign (index 87). Images are hard-linked (not copied) into the output
folder; nothing is written under D:\\DSP391m\\datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOC = ROOT / "data/processed/canonical/itas_mm_v1/localization"
LABELS = ROOT / "data/audit/semantic/final_v1_1/semantic_labels_v1_1.csv"
CLASS_LIST = ROOT / "output/kiet_wp_20261028/recognizer_v2/class_list.json"
DEFAULT_OUT = Path("D:/DSP391m/output/bench_onestage_20261112")
MANIFEST = ROOT / "output/bench_20261112/A5/dataset_manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out
    if "datasets" in str(out).lower():
        raise SystemExit("Refusing to write under a datasets folder")
    classes = [c for c in json.loads(CLASS_LIST.read_text(encoding="utf-8"))["classes"] if c != "unknown"]
    if len(classes) != 87:
        raise SystemExit(f"expected 87 sign codes, found {len(classes)}")
    names = classes + ["other_sign"]
    index = {c: i for i, c in enumerate(names)}
    boxes = pd.read_csv(LABELS, keep_default_na=False)
    boxes = boxes[boxes.split.isin(["TRAIN", "VAL", "TEST"])].reset_index(drop=True)
    counts = {"TRAIN": 0, "VAL": 0, "TEST": 0}
    manifest_rows = []
    for split, group in boxes.groupby("split"):
        split_dir = split.lower()
        if split == "TRAIN":
            split_dir = "train"
        elif split == "VAL":
            split_dir = "val"
        else:
            split_dir = "test"
        (out / "images" / split_dir).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split_dir).mkdir(parents=True, exist_ok=True)
        for image_rel, rows in group.groupby("image_path", sort=False):
            src = (LOC / image_rel).resolve()
            name = Path(image_rel).name
            dst = out / "images" / split_dir / name
            if not dst.exists():
                os.link(src, dst)
            lines = []
            for row in rows.itertuples():
                code = row.canonical_sign_code if row.canonical_sign_code in index else ""
                cls = index[code] if code else index["other_sign"]
                x1, y1, x2, y2 = json.loads(row.bbox_xyxy)
                w_img, h_img = float(row.width_px), float(row.height_px)
                cx, cy = (x1 + x2) / 2 / w_img, (y1 + y2) / 2 / h_img
                bw, bh = (x2 - x1) / w_img, (y2 - y1) / h_img
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                manifest_rows.append({"split": split, "image": image_rel, "canonical_box_id": row.canonical_box_id,
                                      "code": row.canonical_sign_code or "", "class_index": cls,
                                      "class_name": names[cls], "short_side_px": row.short_side_px})
                counts[split] += 1
            (out / "labels" / split_dir / (Path(name).stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(names)}\nnames: {json.dumps(names, ensure_ascii=False)}\n", encoding="utf-8")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {"out": str(out), "classes": len(names), "boxes_per_split": counts,
               "other_sign_boxes": sum(1 for r in manifest_rows if r["class_name"] == "other_sign"),
               "codes_outside_87_mapped_to_other_sign": int(sum(
                   1 for r in manifest_rows if r["code"] and r["code"] not in index))}
    (ROOT / "output/bench_20261112/A5/dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
