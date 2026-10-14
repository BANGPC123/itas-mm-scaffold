"""Verify the frozen dataset; optionally restore YOLO labels from boxes.csv."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"
LOCALIZATION = ROOT / "processed/canonical/itas_mm_v1/localization"
RELEASE = ROOT / "audit/semantic/final_v1_1"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def tree_hash(root):
    records = []
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix().encode()):
        if path.is_file():
            records.append(f"{path.relative_to(root).as_posix()}|{path.stat().st_size}|{sha256(path).lower()}")
    return hashlib.sha256(("\n".join(records) + "\n").encode()).hexdigest().upper()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-labels", action="store_true")
    args = parser.parse_args()
    release = json.loads((RELEASE / "FINAL_MANIFEST.json").read_text(encoding="utf-8"))
    samples = read_csv(LOCALIZATION / "samples.csv")
    if args.restore_labels:
        grouped = defaultdict(list)
        for box in read_csv(LOCALIZATION / "boxes.csv"):
            grouped[box["sample_id"]].append(box)
        for sample in samples:
            path = LOCALIZATION / sample["label_path"]
            content = "\n".join(
                box["canonical_yolo_line"]
                for box in sorted(grouped[sample["sample_id"]], key=lambda box: box["canonical_box_id"])
            ) + "\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != content.encode("ascii"):
                raise ValueError(f"Existing label differs from frozen boxes.csv: {path}")
            if not path.exists():
                path.write_bytes(content.encode("ascii"))
    artifacts = {
        name: (RELEASE / name).is_file() and sha256(RELEASE / name) == entry["sha256"]
        for name, entry in release["artifacts"].items()
    }
    metadata = {
        name: sha256(LOCALIZATION / name) == release["baseline_expected_hashes"]["localization/" + name]
        for name in ("samples.csv", "boxes.csv")
    }
    missing_images = []
    wrong_images = []
    missing_labels = []
    for sample in samples:
        image = LOCALIZATION / sample["image_path"]
        if not image.is_file():
            missing_images.append(sample["image_path"])
        elif sha256(image) != sample["image_sha256"].upper():
            wrong_images.append(sample["image_path"])
        if not (LOCALIZATION / sample["label_path"]).is_file():
            missing_labels.append(sample["label_path"])
    labels_match = tree_hash(LOCALIZATION / "labels") == release["baseline_trees"]["labels_tree_sha256"]
    recognition = read_csv(RELEASE / "recognition_manifest_v1.csv")
    report = {
        "source": "https://drive.google.com/drive/folders/1lusdJKxeES-gXswoCeCXozIY3Xw4guI0",
        "release_artifact_hashes_match": artifacts,
        "canonical_metadata_hashes_match": metadata,
        "samples": len(samples),
        "sample_splits": dict(Counter(row["split"] for row in samples)),
        "recognition_rows": len(recognition),
        "recognition_splits": dict(Counter(row["split"] for row in recognition)),
        "recognition_classes": len(read_csv(RELEASE / "recognition_classes_v1.csv")),
        "labels_tree_hash_matches": labels_match,
        "missing_images": missing_images,
        "wrong_image_hashes": wrong_images,
        "missing_labels": missing_labels,
    }
    report["training_ready"] = (
        all(artifacts.values()) and all(metadata.values()) and labels_match
        and not missing_images and not wrong_images and not missing_labels
    )
    (ROOT / "local-import-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: len(v) if k in {"missing_images", "wrong_image_hashes", "missing_labels"} else v
                      for k, v in report.items()}, indent=2))
    if not report["training_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
