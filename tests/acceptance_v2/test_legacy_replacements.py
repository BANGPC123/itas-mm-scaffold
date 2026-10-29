"""v2 checks that replace archived 2026-10-11 artifact tests; see docs/LEGACY_TESTS_ARCHIVE.md.

They run against the 2026-10-28 evidence (dataset_v2 manifest, clip and temporal
benchmarks, V0 demo, AI-review packs). Opt in with ITASMM_ACCEPTANCE_DATASET_ROOT.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest

from src.evaluation.label_review_schema import validate_review
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/kiet_wp_20261028"
DATASET = os.environ.get("ITASMM_ACCEPTANCE_DATASET_ROOT")
pytestmark = pytest.mark.skipif(not DATASET, reason="local dataset and ignored outputs are required")

DETECTOR = "models/sign_detector/finetuned_20261026.pt"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_default_detector_is_the_benchmarked_lfs_model():
    cfg = load_config("perception")
    assert cfg["sign_detector"]["weights_path"] == DETECTOR
    assert cfg["sign_recognizer"]["weights_path"] == "models/sign_recognizer/retrained_20261028.pt"
    assert cfg["lane_detector"]["weights_path"] == "models/lane/selected_20261009.pth"
    candidate = load(OUTPUT / "wp3/clip_benchmark/benchmark.json")["provenance"]["model_sha256"]["candidate"]
    assert digest(ROOT / DETECTOR) == candidate
    manifest = load(ROOT / "docs/MODEL_WEIGHTS_LFS_MANIFEST_20261018.json")
    entry = next(row for row in manifest["files"] if row["path"] == DETECTOR)
    assert entry["sha256"] == candidate and entry["size_bytes"] == (ROOT / DETECTOR).stat().st_size
    assert manifest["file_count"] == len(manifest["files"])
    assert manifest["total_bytes"] == sum(row["size_bytes"] for row in manifest["files"])


def test_canonical_ai_review_never_claims_human_ground_truth():
    review = load(OUTPUT / "wp2/canonical_ai/ai_review.json")
    manifest = load(OUTPUT / "wp2/canonical_test_pack/review_manifest.json")
    result = validate_review(review, manifest)
    assert result["frames"] == 394
    assert result["human_confirmed"] == 0 and result["exhaustive"] == 0
    assert all(not frame["human_confirmed"] for frame in review["frames"])


def test_dataset_v2_ai_review_labels_keep_ai_provenance():
    path = ROOT / "data/manifests/dataset_v2/ai_review_labels.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 514
    assert len({row["sample_key"] for row in rows}) == len(rows)
    assert all(row["human_confirmed"] is False for row in rows)
    assert all(row["review_method"]["type"] == "ai_reviewed" for row in rows)
    assert all(row["reviewer"].startswith("ai-review") for row in rows)


def test_pilot_ai_review_keeps_owner_source_boxes():
    manifest = load(OUTPUT / "wp2/pilot_v0_copy/review_manifest.json")
    review = load(OUTPUT / "wp2/pilot_v0_ai/ai_review.json")
    assert len(manifest["frames"]) == len(review["frames"]) == 120
    count = 0
    for source, reviewed in zip(manifest["frames"], review["frames"]):
        assert source["id"] == reviewed["id"]
        by_id = {sign["id"]: sign for sign in reviewed["signs"]}
        for sign in source["signs"]:
            assert by_id[sign["id"]]["bbox_xyxy"] == sign["bbox_xyxy"]
            count += 1
    assert count > 0
    assert all(not frame["human_confirmed"] for frame in review["frames"])


def test_review_pack_images_resolve_to_recorded_hashes():
    for pack in ("wp2/canonical_test_pack", "wp2/pilot_v0_copy"):
        directory = OUTPUT / pack
        for frame in load(directory / "review_manifest.json")["frames"]:
            image = directory / frame["image"]
            assert image.is_file(), image
            assert digest(image) == frame["image_sha256"]


def test_release_gate_v2_holds_release_and_pins_current_evidence():
    gate = load(ROOT / "docs/release_gate_v2.json")
    assert gate["release_decision"] == "HOLD" and gate["approval_status"] == "NOT_APPROVED"
    by_id = {row["id"]: row for row in gate["gates"]}
    assert by_id["human_ground_truth"]["status"] == "BLOCKED"
    assert all(row.get("gaps") or row.get("evidence") for row in gate["gates"])
    provenance = gate["provenance"]
    assert digest(OUTPUT / "wp3/clip_benchmark/benchmark.json") == provenance["clip_benchmark_json_sha256"]
    assert digest(OUTPUT / "wp3/temporal_benchmark/temporal_benchmark.json") == provenance["temporal_benchmark_json_sha256"]
    assert digest(OUTPUT / "wp4/e2e_demo/demo_report.json") == provenance["demo_report_json_sha256"]
    assert digest(OUTPUT / "wp4/e2e_demo/test_video_annotated.mp4") == provenance["demo_video_sha256"]


def test_temporal_benchmark_reports_speed_without_accuracy_claims():
    report = load(OUTPUT / "wp3/temporal_benchmark/temporal_benchmark.json")
    assert report["status"] == "COMPLETED" and report["limitations"]
    assert len(report["segments"]) == 3
    for segment in report["segments"]:
        assert segment["verified_full_decode"] is True
        assert segment["inference_fps"] > 0 and segment["render_fps"] > 0
        assert not {"f1", "precision", "recall", "accuracy"} & set(segment)


def test_v0_demo_report_records_limitations_and_no_tts_failures():
    report = load(OUTPUT / "wp4/e2e_demo/demo_report.json")
    assert report["status"] == "COMPLETED" and report["limitations"]
    assert report["speech_events"] and not report["tts_failures"]
    assert report["source_sha256"] == "eb109f8c852df580611c236e2b313b01cf4a62662586ce71fa35525bdda05ac4"
