"""Audit portable assets after a clean clone, without synthesizing missing evidence.

This checks source availability and integrity. It does NOT certify field accuracy,
dataset legality or legal currency, and it never changes model/data assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UFLD_SHA = "c903880678454dfd9b55a63022368db05c00bc6d"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def file_record(name: str, relative: str, sha256: str | None = None,
                expected_size: int | None = None) -> dict:
    path = ROOT / relative
    found = path.is_file()
    size = path.stat().st_size if found else None
    size_ok = found and (expected_size is None or size == expected_size)
    hash_ok = bool(size_ok and (sha256 is None or digest(path) == sha256))
    return {"name": name, "path": relative, "present": found,
            "verified": hash_ok, "bytes": size}


def audit() -> dict:
    weights_manifest = json.loads(
        (ROOT / "docs/MODEL_WEIGHTS_LFS_MANIFEST_20261018.json").read_text(
            encoding="utf-8"))
    weights = [
        file_record("lfs_weight", row["path"], row["sha256"], row["size_bytes"])
        for row in weights_manifest["files"]
    ]

    official_sources = ROOT / "data/regulations/official_20261011/sources.json"
    legal = []
    if official_sources.is_file():
        snapshot = json.loads(official_sources.read_text(encoding="utf-8"))
        legal = [
            file_record("official_legal_pdf",
                        "data/regulations/official_20261011/" + item["file"],
                        item["sha256"], item["bytes"])
            for item in snapshot.get("official_documents", [])
        ]
    # Source-generated hashes alone are not an independent trust anchor.
    # Compare them to the small, versioned snapshot committed to Git.
    pin_path = ROOT / "docs/OFFICIAL_PDF_SNAPSHOT_20261022.json"
    pins_verified = False
    if pin_path.is_file() and official_sources.is_file():
        pinned = json.loads(pin_path.read_text(encoding="utf-8"))
        recorded = json.loads(official_sources.read_text(encoding="utf-8"))
        key = lambda item: (item["file"], item["sha256"], item["bytes"],
                            item["source_url"])
        pins_verified = (len(pinned.get("sources", [])) == 4
                         and sorted(map(key, pinned["sources"]))
                         == sorted(map(key, recorded.get("official_documents", []))))
    index_path = ROOT / "data/processed/legal_reference/official_corpus_20261011"
    index = {
        "present": (index_path / "manifest.json").is_file()
                   and (index_path / "chunks.jsonl").is_file(),
        "relative_path": index_path.relative_to(ROOT).as_posix(),
    }
    if index["present"]:
        data = json.loads((index_path / "manifest.json").read_text(encoding="utf-8"))
        chunks = (index_path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        index["verified"] = (
            len(chunks) == data.get("chunks", -1)
            and len(chunks) > 0
            and len(legal) == data.get("source_count", -1)
            and all(f["verified"] for f in legal)
        )
        index["chunks"] = len(chunks)
    else:
        index["verified"] = False

    vendor = ROOT / "third_party/Ultra-Fast-Lane-Detection-v2"
    vendor_sha = None
    if (vendor / "utils/loss.py").is_file():
        result = subprocess.run(
            ["git", "-C", str(vendor), "rev-parse", "HEAD"],
            text=True, capture_output=True, check=False,
        )
        if result.returncode == 0:
            vendor_sha = result.stdout.strip()

    # A field evaluation requires genuine reviewed source data, not generated
    # placeholders or simply existing inference weights.
    private_inputs = {
        "canonical_training_dataset":
            (ROOT / "data/processed/canonical/itas_mm_v1/localization/data.yaml").is_file(),
        "human_review_manifest":
            (ROOT / "data/lane_vietnam_review_20261011/manifest.json").is_file(),
        "legacy_release_gate_report":
            (ROOT / "output/evaluation/ITAS_MM_RELEASE_GATE_20261011.json").is_file(),
        "legacy_video_review":
            (ROOT / "output/video_recheck_full_20261014/test_video_annotated.mp4").is_file(),
    }
    return {
        "schema_version": 1,
        "purpose": "clean-clone reproducibility, not field-validation",
        "lfs": {"count": len(weights), "verified_count":
                sum(int(item["verified"]) for item in weights), "files": weights},
        "legal_snapshot": {
            "pinned_snapshot_verified": pins_verified,
            "count": len(legal),
            "verified_count": sum(int(item["verified"]) for item in legal),
            "files": legal, "offline_index": index,
            "legal_currency_verified": False,
        },
        "vendor_ufld": {
            "expected_commit": UFLD_SHA, "actual_commit": vendor_sha,
            "verified": vendor_sha == UFLD_SHA,
        },
        "historical_private_inputs": private_inputs,
        "clean_clone_core_assets_ready":
            len(weights) == weights_manifest["file_count"]
            and all(row["verified"] for row in weights)
            and len(legal) == 4 and all(row["verified"] for row in legal)
            and pins_verified and index["verified"] and vendor_sha == UFLD_SHA,
        "historical_full_acceptance_ready": all(private_inputs.values())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional local JSON report file")
    parser.add_argument("--strict-core", action="store_true",
                        help="Exit 1 if basic assets are not reproducibly restored")
    args = parser.parse_args()
    result = audit()
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    return int(args.strict_core and not result["clean_clone_core_assets_ready"])


if __name__ == "__main__":
    raise SystemExit(main())
