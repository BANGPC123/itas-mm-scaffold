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
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/manifests/dataset_v2"
VERSION = "dataset_v2.0"
LOC = Path("processed/canonical/itas_mm_v1/localization")
SPLITS = ("TRAIN", "VAL", "TEST")
SAMPLE_FIELDS = ("sample_key", "image_sha256", "split", "split_group_id",
                 "source_ids", "primary_source_id", "source_original_split",
                 "is_augmented_derivative", "image_relpath", "width_px", "height_px",
                 "box_count", "box_label_status", "code_label_status", "human_review_status", "prior_human_review_status",
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


def apply_ai_reviews(rows, reviews):
    """Attach AI status to existing TEST images without erasing prior human provenance."""
    updated = [dict(row, prior_human_review_status=row.get("human_review_status", "none")) for row in rows]
    by_hash = {row["image_sha256"].lower(): row for row in updated}
    if len(by_hash) != len(updated):
        raise ValueError("Duplicate image SHA256 in dataset manifest")
    seen, labels = set(), []
    sign_keys = ("id", "bbox_xyxy", "label", "judgment", "panel_role", "parent_id",
                 "recognition_confidence", "detector_confidence", "detector_match", "review_reason", "source")
    for review in reviews:
        from src.evaluation.label_review_schema import validate_review
        validate_review(review)
        frames = review.get("frames", [])
        if (review.get("status") != "COMPLETED" or review.get("review_method", {}).get("type") != "ai_reviewed"
                or review.get("provenance", {}).get("expected_frames") != len(frames)):
            raise ValueError("AI review is incomplete or has invalid provenance")
        for frame in frames:
            digest = frame.get("image_sha256", "").lower()
            if (digest in seen or digest not in by_hash or by_hash[digest]["split"] != "TEST"
                    or frame.get("human_confirmed") is not False
                    or frame.get("reviewer") != "ai-review (Codex)"
                    or frame.get("review_method", {}).get("type") != "ai_reviewed"):
                raise ValueError("AI review frame is duplicated, unmatched, non-TEST, or misattributed")
            seen.add(digest)
            row = by_hash[digest]
            row["code_label_status"] = "ai_reviewed"
            row["human_review_status"] = "ai_reviewed"
            labels.append({"sample_key": row["sample_key"], "image_sha256": digest,
                           "frame_id": frame["id"], "source_group": frame["source_group"],
                           "width": frame["width"], "height": frame["height"],
                           "frame_number": frame.get("frame_number"),
                           "timestamp_seconds": frame.get("timestamp_seconds"),
                           "reviewer": frame["reviewer"], "human_confirmed": False,
                           "review_method": frame["review_method"], "lane_status": frame["lane_status"],
                           "lanes": frame["lanes"], "detector_proposals": frame.get("detector_proposals", []),
                           "signs": [{key: sign[key] for key in sign_keys if key in sign}
                                     for sign in frame["signs"]]})
    labels.sort(key=lambda item: item["sample_key"])
    return updated, labels


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
    expected_ai = {row["sample_key"]: row for row in rows if row.get("human_review_status") == "ai_reviewed"}
    if expected_ai or manifest.get("ai_reviews"):
        label_path = OUT / "ai_review_labels.jsonl"
        try:
            labels = [json.loads(line) for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            actual = {item.get("sample_key"): item for item in labels}
            if (len(actual) != len(labels) or set(actual) != set(expected_ai)
                    or "ai_review_labels.jsonl" not in manifest["outputs_sha256"]):
                errors.append("AI review labels do not match manifest coverage")
            for key, label in actual.items():
                row = expected_ai.get(key, {})
                if (label.get("human_confirmed") is not False or label.get("reviewer") != "ai-review (Codex)"
                        or label.get("review_method", {}).get("type") != "ai_reviewed"
                        or label.get("image_sha256") != row.get("image_sha256")
                        or row.get("split") != "TEST" or row.get("code_label_status") != "ai_reviewed"):
                    errors.append("AI review attribution or sample provenance is invalid")
                    break
        except (OSError, ValueError, TypeError):
            errors.append("AI review labels are missing or invalid")
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
            "prior_human_review_status": "none",
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
            "prior_human_review_status": ("owner_confirmed_exhaustive"
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
    global OUT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-root", type=Path, default=os.environ.get("ITASMM_DATASET_ROOT"))
    parser.add_argument("--pilot-pack", type=Path)
    parser.add_argument("--pilot-review", type=Path)
    parser.add_argument("--ai-review", type=Path, action="append", default=[],
                        help="Completed AI review JSON; may be supplied more than once")
    parser.add_argument("--output-dir", type=Path, default=OUT,
                        help="Manifest destination, including for verification")
    parser.add_argument("--verify", action="store_true",
                        help="Check the committed manifest instead of writing it")
    parser.add_argument("--rehash", action="store_true",
                        help="With --verify, recompute SHA256 of every canonical image")
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if args.verify:
        report = verify(args.dataset_root.resolve() if args.dataset_root else None,
                        args.pilot_pack.resolve() if args.pilot_pack else None, args.rehash)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 1
    if not args.dataset_root:
        parser.error("--dataset-root or ITASMM_DATASET_ROOT is required")
    root = args.dataset_root.resolve()
    if OUT == root or root in OUT.parents:
        raise ValueError("Manifest output cannot be inside the dataset source")
    rows = canonical_rows(root)
    pilot_info = None
    if args.pilot_pack:
        review = args.pilot_review or args.pilot_pack / "human_review.json"
        extra, pilot_info = pilot_rows(args.pilot_pack.resolve(), review.resolve())
        rows += extra
    ai_reviews = [json.loads(path.read_text(encoding="utf-8")) for path in args.ai_review]
    ai_labels = []
    if ai_reviews:
        rows, ai_labels = apply_ai_reviews(rows, ai_reviews)
    leaks = leak_report(rows)
    if leaks:
        print(json.dumps({"status": "LEAKAGE", "findings": leaks}, indent=2))
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (SPLITS.index(r["split"]), r["sample_key"]))
    write_csv(OUT / "samples.csv", SAMPLE_FIELDS, rows)
    sources = source_rows(root, pilot_info)
    write_csv(OUT / "sources.csv", list(sources[0]), sources)
    if ai_reviews:
        with (OUT / "ai_review_labels.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for item in ai_labels:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    by_split = {sp: [r for r in rows if r["split"] == sp] for sp in SPLITS}
    inputs = {name: sha256_file(root / name) for name in
              ("sources.json", "download-manifest.csv",
               (LOC / "samples.csv").as_posix(), (LOC / "boxes.csv").as_posix())}
    if pilot_info:
        inputs["pilot/review_manifest.json"] = sha256_file(args.pilot_pack / "review_manifest.json")
        inputs["pilot/human_review.json"] = sha256_file(review)
    ai_info = []
    for path, item in zip(args.ai_review, ai_reviews):
        ai_info.append({"name": path.parent.name, "review_sha256": sha256_file(path),
                        "frames": len(item["frames"]),
                        "input_manifest_sha256": item["provenance"]["input_manifest_sha256"],
                        "perception_config_sha256": item["provenance"]["perception_config_sha256"],
                        "model_sha256": item["provenance"]["model_sha256"],
                        "method": "ai_reviewed", "human_confirmed": False})
    old_manifest = ROOT / "data/manifests/dataset_v2/MANIFEST.json"
    prior_test = None
    if ai_reviews and old_manifest.is_file():
        prior_test = json.loads(old_manifest.read_text(encoding="utf-8"))["splits"]["TEST"]["frozen_sha256"]
    manifest = {
        "version": "dataset_v2.1" if ai_reviews else VERSION,
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
        "ai_reviews": ai_info,
        "image_relpath_base": LOC.as_posix(),
        "leakage_checks": {"columns": ["split_group_id", "image_sha256", "sample_key"],
                           "note": ("V1 lineage components are exact-content groups, one per image "
                                    "SHA256; split_group_id is the stricter constraint group."),
                           "result": "PASS"},
        "frozen_test_policy": "TEST membership is frozen by frozen_sha256; never tune on TEST. AI label provenance changes are recorded separately.",
        "test_revision": {"prior_membership_sha256": prior_test,
                          "membership_changed": prior_test is not None and prior_test != split_digest(by_split["TEST"]),
                          "label_provenance_changed": bool(ai_reviews),
                          "ai_label_sha256": sha256_file(OUT / "ai_review_labels.jsonl") if ai_reviews else None},
        "outputs_sha256": {n: sha256_file(OUT / n) for n in
                           (("samples.csv", "sources.csv", "ai_review_labels.jsonl") if ai_reviews
                            else ("samples.csv", "sources.csv"))},
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8", newline="\n")
    print(json.dumps({"status": "WRITTEN", "splits": manifest["splits"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
