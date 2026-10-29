import importlib.util
import csv
import json
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "build_dataset_v2_manifest", ROOT / "scripts/build_dataset_v2_manifest.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(key, sha, split, group):
    return {"sample_key": key, "image_sha256": sha, "split": split, "split_group_id": group}


def test_clean_rows_have_no_leakage():
    rows = [row("a", "1", "TRAIN", "g1"), row("b", "2", "TRAIN", "g1"), row("c", "3", "TEST", "g2")]
    assert module.leak_report(rows) == {}


def test_group_across_splits_is_leakage():
    rows = [row("a", "1", "TRAIN", "g1"), row("b", "2", "TEST", "g1")]
    assert "split_group_id" in module.leak_report(rows)


def test_same_image_in_two_splits_is_leakage():
    rows = [row("a", "1", "TRAIN", "g1"), row("b", "1", "VAL", "g2")]
    findings = module.leak_report(rows)
    assert "image_sha256" in findings and "duplicate_image_sha256" in findings


def test_duplicate_image_within_split_is_reported():
    rows = [row("a", "1", "TEST", "g1"), row("b", "1", "TEST", "g2")]
    assert "duplicate_image_sha256" in module.leak_report(rows)


def test_split_digest_is_order_independent_and_content_sensitive():
    rows = [row("a", "1", "TEST", "g"), row("b", "2", "TEST", "g")]
    assert module.split_digest(rows) == module.split_digest(list(reversed(rows)))
    assert module.split_digest(rows) != module.split_digest(rows[:1])


def test_ai_review_updates_only_matching_test_rows_and_retains_prior_owner_status():
    rows = [dict(row("field:1", "a" * 64, "TEST", "g"),
                 code_label_status="not_reviewed", human_review_status="owner_confirmed_exhaustive"),
            dict(row("canon:2", "b" * 64, "VAL", "g2"),
                 code_label_status="old", human_review_status="none")]
    review = {"schema_version": "itasmm-field-review-v1", "status": "COMPLETED", "review_method": {"type": "ai_reviewed"},
              "provenance": {"expected_frames": 1}, "frames": [
                  {"id": "frame_1", "image_sha256": "a" * 64, "width": 100, "height": 100,
                   "source_group": "video", "reviewer": "ai-review (Codex)",
                   "human_confirmed": False, "review_method": {"type": "ai_reviewed"},
                   "signs": [], "lanes": [], "lane_status": "uncertain",
                   "detector_proposals": [{"bbox_xyxy": [1, 2, 3, 4], "confidence": 0.9}]}]}
    updated, labels = module.apply_ai_reviews(rows, [review])
    assert updated[0]["human_review_status"] == "ai_reviewed"
    assert updated[0]["code_label_status"] == "ai_reviewed"
    assert updated[0]["prior_human_review_status"] == "owner_confirmed_exhaustive"
    assert updated[1]["code_label_status"] == "old"
    assert rows[0]["human_review_status"] == "owner_confirmed_exhaustive"
    assert labels[0]["sample_key"] == "field:1" and labels[0]["human_confirmed"] is False
    assert labels[0]["detector_proposals"][0]["bbox_xyxy"] == [1, 2, 3, 4]
    assert (labels[0]["width"], labels[0]["height"]) == (100, 100)


def test_ai_review_rejects_partial_duplicate_or_human_confirmed_inputs():
    rows = [dict(row("canon:1", "a" * 64, "TEST", "g"),
                 code_label_status="old", human_review_status="none")]
    frame = {"id": "f", "image_sha256": "a" * 64, "width": 100, "height": 100, "reviewer": "ai-review (Codex)",
             "human_confirmed": False, "review_method": {"type": "ai_reviewed"},
             "signs": [], "lanes": [], "lane_status": "uncertain", "source_group": "IMG_4017"}
    review = {"schema_version": "itasmm-field-review-v1", "status": "COMPLETED", "review_method": {"type": "ai_reviewed"},
              "provenance": {"expected_frames": 1}, "frames": [frame]}
    with pytest.raises(ValueError):
        module.apply_ai_reviews(rows, [dict(review, status="PARTIAL")])
    with pytest.raises(ValueError):
        module.apply_ai_reviews(rows, [review, review])
    with pytest.raises(ValueError):
        module.apply_ai_reviews(rows, [dict(review, frames=[dict(frame, human_confirmed=True)])])


def test_verify_rejects_ai_label_claiming_human_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "OUT", tmp_path)
    sample = dict(row("canon:1", "a" * 64, "TEST", "g"),
                  human_review_status="ai_reviewed", code_label_status="ai_reviewed")
    with (tmp_path / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sample)); writer.writeheader(); writer.writerow(sample)
    labels = {"sample_key": "canon:1", "image_sha256": "a" * 64,
              "reviewer": "ai-review (Codex)", "human_confirmed": True,
              "review_method": {"type": "ai_reviewed"}}
    (tmp_path / "ai_review_labels.jsonl").write_text(json.dumps(labels) + "\n", encoding="utf-8")
    manifest = {"outputs_sha256": {name: module.sha256_file(tmp_path / name) for name in
                                   ("samples.csv", "ai_review_labels.jsonl")},
                "splits": {"TEST": {"samples": 1, "frozen_sha256": module.split_digest([sample])}},
                "ai_reviews": [{"frames": 1}]}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = module.verify(None, None, False)
    assert result["status"] == "FAIL"
    assert any("AI review" in error for error in result["errors"])


def test_ai_review_rejects_out_of_bounds_box():
    rows = [dict(row("canon:1", "a" * 64, "TEST", "g"),
                 code_label_status="old", human_review_status="none")]
    frame = {"id": "f", "image_sha256": "a" * 64, "width": 100, "height": 100,
             "reviewer": "ai-review (Codex)", "human_confirmed": False,
             "review_method": {"type": "ai_reviewed"}, "source_group": "IMG_4017",
             "signs": [{"id": "s", "bbox_xyxy": [-1, 0, 20, 20], "judgment": "sign",
                        "panel_role": "unknown", "label": "unknown"}],
             "lanes": [], "lane_status": "uncertain"}
    review = {"schema_version": "itasmm-field-review-v1", "status": "COMPLETED",
              "review_method": {"type": "ai_reviewed"}, "provenance": {"expected_frames": 1},
              "frames": [frame]}
    with pytest.raises(ValueError):
        module.apply_ai_reviews(rows, [review])
