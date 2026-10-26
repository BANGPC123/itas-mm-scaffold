"""Build the versioned dataset_v2 reference manifest (WP1).

dataset_v2 reuses the hash-verified canonical V1 localization pixels and adds
the owner-reviewed field pilot frames. Only non-sensitive metadata (IDs,
SHA256, splits, provenance, label status) is written; no image or video bytes.

Usage:
  build:  python scripts/build_dataset_v2_manifest.py --dataset-root <ITAS-MM root>
              [--pilot-pack <review pack dir>] [--pilot-review <review JSON>]
  verify: python scripts/build_dataset_v2_manifest.py --verify
              [--dataset-root <root> [--rehash]] [--pilot-pack <dir>]
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/manifests/dataset_v2"
VERSION = "dataset_v2.0"
LOC = Path("processed/canonical/itas_mm_v1/localization")
SPLITS = ("TRAIN", "VAL", "TEST")
SAMPLE_FIELDS = ("sample_key", "image_sha256", "split", "split_group_id",
                 "source_ids", "primary_source_id", "source_original_split",
                 "is_augmented_derivative", "image_relpath", "width_px", "height_px",
                 "box_count", "box_label_status", "code_label_status", "human_review_status",
                 "timestamp_seconds")
RIGHTS_NOTE = ("Reported license recorded as published; license review skipped by owner "
               "decision (local research project). Do not redistribute media publicly.")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def split_digest(rows):
    """Order-independent fingerprint of a split: sorted key|sha lines."""
    lines = sorted(f"{r['sample_key']}|{r['image_sha256']}" for r in rows)
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def leak_report(rows):
    """Return leakage findings; an empty dict means no leakage."""
    findings = {}
    for column in ("split_group_id", "image_sha256", "sample_key"):
        splits = defaultdict(set)
        for row in rows:
            splits[row[column]].add(row["split"])
        leaked = sorted(key for key, value in splits.items() if len(value) > 1)
        if leaked:
            findings[column] = leaked[:20]
    seen = defaultdict(int)
    for row in rows:
        seen[row["image_sha256"]] += 1
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    if duplicates:
        findings["duplicate_image_sha256"] = duplicates[:20]
    return findings


def verify(dataset_root, pilot_pack, rehash):
    """Re-check a committed manifest against the local dataset; return a report."""
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    rows = read_csv(OUT / "samples.csv")
    errors = []
    for name, digest in manifest["outputs_sha256"].items():
        if sha256_file(OUT / name) != digest:
            errors.append(f"output hash mismatch: {name}")
    for split, info in manifest["splits"].items():
        part = [r for r in rows if r["split"] == split]
        if len(part) != info["samples"] or split_digest(part) != info["frozen_sha256"]:
            errors.append(f"frozen split mismatch: {split}")
    leaks = leak_report(rows)
    if leaks:
        errors.append(f"leakage: {sorted(leaks)}")
    checked = missing = mismatched = 0
    if dataset_root:
        for row in rows:
            if not row["image_relpath"]:
                continue
            path = dataset_root / LOC / row["image_relpath"]
            checked += 1
            if not path.is_file():
                missing += 1
            elif rehash and sha256_file(path) != row["image_sha256"]:
                mismatched += 1
    if pilot_pack:
        frames = json.loads((pilot_pack / "review_manifest.json").read_text(encoding="utf-8"))["frames"]
        expected = {r["image_sha256"] for r in rows if r["primary_source_id"] == "field_video_dev_20261015"}
        actual = {sha256_file(pilot_pack / f["image"]) for f in frames}
        if expected != actual:
            errors.append("pilot frame hashes differ from manifest")
    if missing or mismatched:
        errors.append(f"dataset files missing={missing} hash_mismatch={mismatched}")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors,
            "canonical_files_checked": checked, "rehash": bool(rehash and dataset_root),
            "pilot_checked": bool(pilot_pack)}


def canonical_rows(dataset_root):
    samples = read_csv(dataset_root / LOC / "samples.csv")
    rows = []
    for s in samples:
        provenance = json.loads(s["source_occurrence_provenance_json"] or "[]")
        sources = sorted({p["source"] for p in provenance} | {s["representative_source"]})
        rows.append({
            "sample_key": "canon:" + s["image_sha256"].lower()[:16],
            "image_sha256": s["image_sha256"].lower(),
            "split": s["split"],
            "split_group_id": s["split_constraint_group_id"],
            "source_ids": ";".join(sources),
            "primary_source_id": s["representative_source"],
            "source_original_split": s["representative_original_split"],
            "is_augmented_derivative": s["representative_is_augmented_derivative"] or "unknown",
            "image_relpath": s["image_path"],
            "width_px": s["image_width_px"],
            "height_px": s["image_height_px"],
            "box_count": s["canonical_box_count"],
            "box_label_status": "source_annotation_canonicalized",
            "code_label_status": "ai_assisted_semantic_v1_1",
            "human_review_status": "none",
            "timestamp_seconds": "",
        })
    return rows


def pilot_rows(pack, review_path):
    manifest = json.loads((pack / "review_manifest.json").read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    reviewed = {f["id"]: f for f in review["frames"]}
    video = manifest["source_video_sha256"].lower()
    rows = []
    for frame in manifest["frames"]:
        image = pack / frame["image"]
        digest = sha256_file(image)
        if digest != frame["image_sha256"].lower():
            raise ValueError(f"Pilot frame hash mismatch: {frame['id']}")
        r = reviewed.get(frame["id"], {})
        confirmed = r.get("human_confirmed") is True
        rows.append({
            "sample_key": f"field:{video[:16]}:{frame['id']}",
            "image_sha256": digest,
            "split": "TEST",
            "split_group_id": "video:" + video,
            "source_ids": "field_video_dev_20261015",
            "primary_source_id": "field_video_dev_20261015",
            "source_original_split": "",
            "is_augmented_derivative": "false",
            "image_relpath": "",
            "width_px": frame["width"],
            "height_px": frame["height"],
            "box_count": len(r.get("signs", frame["signs"])),
            "box_label_status": ("owner_bulk_approved_detector_proposals" if confirmed
                                 else "detector_proposal_unreviewed"),
            "code_label_status": "not_reviewed",
            "human_review_status": ("owner_confirmed_exhaustive"
                                    if confirmed and r.get("all_visible_signs_annotated") is True
                                    else ("owner_confirmed" if confirmed else "none")),
            "timestamp_seconds": frame.get("timestamp_seconds", ""),
        })
    return rows, {"source_video_sha256": video, "frames": len(rows),
                  "review_method": review.get("review_method", {})}


def source_rows(dataset_root, pilot_info):
    sources = json.loads((dataset_root / "sources.json").read_text(encoding="utf-8"))["sources"]
    archives = {r["source_id"]: r for r in read_csv(dataset_root / "download-manifest.csv")}
    rows = []
    for s in sources:
        a = archives.get(s["id"], {})
        rows.append({
            "source_id": s["id"], "kind": "public_dataset_archive", "uri": s.get("url", ""),
            "version": s.get("version", ""),
            "reported_license": s.get("kaggle_reported_license", a.get("reported_license", "")),
            "rights_note": RIGHTS_NOTE, "archive_sha256": a.get("archive_sha256", "").lower(),
            "archive_bytes": a.get("archive_bytes", ""), "images": a.get("images", ""),
            "annotations": a.get("annotations", ""), "annotation_format": a.get("annotation_format", ""),
            "captured_date": "2026-10-02",
        })
    if pilot_info:
        rows.append({
            "source_id": "field_video_dev_20261015", "kind": "user_supplied_video",
            "uri": "private:user-supplied-footage", "version": "1",
            "reported_license": "not documented",
            "rights_note": "User-supplied driving footage; keep video and frames out of public Git.",
            "archive_sha256": pilot_info["source_video_sha256"], "archive_bytes": "",
            "images": pilot_info["frames"], "annotations": "", "annotation_format": "itasmm-field-review-v1",
            "captured_date": "",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("ITASMM_DATASET_ROOT"))
    parser.add_argument("--pilot-pack", type=Path)
    parser.add_argument("--pilot-review", type=Path)
    parser.add_argument("--verify", action="store_true",
                        help="Check the committed manifest instead of writing it")
    parser.add_argument("--rehash", action="store_true",
                        help="With --verify, recompute SHA256 of every canonical image")
    args = parser.parse_args()
    if args.verify:
        report = verify(args.dataset_root.resolve() if args.dataset_root else None,
                        args.pilot_pack.resolve() if args.pilot_pack else None, args.rehash)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 1
    if not args.dataset_root:
        parser.error("--dataset-root or ITASMM_DATASET_ROOT is required")
    root = args.dataset_root.resolve()
    rows = canonical_rows(root)
    pilot_info = None
    if args.pilot_pack:
        review = args.pilot_review or args.pilot_pack / "human_review.json"
        extra, pilot_info = pilot_rows(args.pilot_pack.resolve(), review.resolve())
        rows += extra
    leaks = leak_report(rows)
    if leaks:
        print(json.dumps({"status": "LEAKAGE", "findings": leaks}, indent=2))
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (SPLITS.index(r["split"]), r["sample_key"]))
    write_csv(OUT / "samples.csv", SAMPLE_FIELDS, rows)
    sources = source_rows(root, pilot_info)
    write_csv(OUT / "sources.csv", list(sources[0]), sources)
    by_split = {sp: [r for r in rows if r["split"] == sp] for sp in SPLITS}
    inputs = {name: sha256_file(root / name) for name in
              ("sources.json", "download-manifest.csv",
               (LOC / "samples.csv").as_posix(), (LOC / "boxes.csv").as_posix())}
    if pilot_info:
        inputs["pilot/review_manifest.json"] = sha256_file(args.pilot_pack / "review_manifest.json")
        inputs["pilot/human_review.json"] = sha256_file(review)
    manifest = {
        "version": VERSION,
        "description": "Canonical V1 localization pixels (hash-verified) plus owner-reviewed field pilot frames.",
        "inputs_sha256": inputs,
        "splits": {sp: {"samples": len(v), "split_groups": len({r["split_group_id"] for r in v}),
                        "boxes": sum(int(r["box_count"]) for r in v),
                        "frozen_sha256": split_digest(v)} for sp, v in by_split.items()},
        "sources": {s["source_id"]: sum(1 for r in rows if r["primary_source_id"] == s["source_id"])
                    for s in sources},
        "label_status": {k: sum(1 for r in rows if r["human_review_status"] == k)
                         for k in sorted({r["human_review_status"] for r in rows})},
        "pilot": pilot_info,
        "image_relpath_base": LOC.as_posix(),
        "leakage_checks": {"columns": ["split_group_id", "image_sha256", "sample_key"],
                           "note": ("V1 lineage components are exact-content groups, one per image "
                                    "SHA256; split_group_id is the stricter constraint group."),
                           "result": "PASS"},
        "frozen_test_policy": "TEST is frozen by frozen_sha256; never tune on TEST.",
        "outputs_sha256": {n: sha256_file(OUT / n) for n in ("samples.csv", "sources.csv")},
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8", newline="\n")
    print(json.dumps({"status": "WRITTEN", "splits": manifest["splits"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
