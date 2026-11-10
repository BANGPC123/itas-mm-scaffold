from importlib import import_module
import csv
import hashlib
import json


def test_select_frames_deduplicates_and_sorts_by_clip_and_frame_number():
    select_frames = import_module("scripts.build_ai_review_packs_20261028").select_frames
    rows = [
        {"split": "TEST", "representative_source_path": "x/IMG_4022-MOV_out0010.jpg", "image_sha256": "bb"},
        {"split": "TEST", "representative_source_path": "x/IMG_4022-MOV_out0010_other.jpg", "image_sha256": "aa"},
        {"split": "TEST", "representative_source_path": "x/IMG_4022-MOV_out0002.jpg", "image_sha256": "cc"},
        {"split": "TRAIN", "representative_source_path": "x/IMG_4022-MOV_out0001.jpg", "image_sha256": "dd"},
        {"split": "TEST", "representative_source_path": "x/IMG_9999-MOV_out0001.jpg", "image_sha256": "ee"},
    ]
    chosen = select_frames(rows, {"4022"})
    assert [(item["clip"], item["frame_number"], item["row"]["image_sha256"]) for item in chosen] == [
        ("IMG_4022", 2, "cc"), ("IMG_4022", 10, "aa")
    ]
    assert chosen[1]["candidate_sha256"] == ["aa", "bb"]


def test_build_pack_copies_verified_frame_and_canonical_boxes(tmp_path):
    module = import_module("scripts.build_ai_review_packs_20261028")
    root = tmp_path / "dataset"
    loc = root / "processed/canonical/itas_mm_v1/localization"
    image = loc / "images/test/example.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"canonical image bytes")
    sha = hashlib.sha256(image.read_bytes()).hexdigest()
    row = {"split": "TEST", "sample_id": "sample:1", "image_sha256": sha,
           "representative_source_path": "train/IMG_4022-MOV_out0002.jpg",
           "image_path": "images/test/example.jpg", "image_width_px": "100",
           "image_height_px": "80"}
    with (loc / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    box = {"sample_id": "sample:1", "canonical_box_id": "box:1",
           "canonical_xyxy_normalized_json": json.dumps([0.1, 0.25, 0.5, 0.75])}
    with (loc / "boxes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(box)); writer.writeheader(); writer.writerow(box)
    out = tmp_path / "pack"
    module.build_pack(root, out, {"4022"})
    manifest = json.loads((out / "review_manifest.json").read_text(encoding="utf-8"))
    frame = manifest["frames"][0]
    assert manifest["schema_version"] == "itasmm-field-review-v1"
    assert frame["source_group"] == "IMG_4022" and frame["frame_number"] == 2
    assert frame["human_confirmed"] is False and frame["timestamp_seconds"] is None
    assert frame["signs"][0]["bbox_xyxy"] == [10.0, 20.0, 50.0, 60.0]
    assert frame["signs"][0]["id"] == "box:1"
    assert (out / frame["image"]).read_bytes() == image.read_bytes()


def test_ai_review_keeps_unmatched_boxes_uncertain_and_preserves_owner_origin():
    review_frame = import_module("scripts.run_ai_review_20261028").review_frame
    frame = {"id": "frame_1", "reviewer": "owner", "human_confirmed": True,
             "reviewed_at": "2026-10-26", "all_visible_signs_annotated": True,
             "signs": [{"id": "a", "bbox_xyxy": [0, 0, 10, 10], "judgment": "sign"},
                       {"id": "b", "bbox_xyxy": [20, 20, 30, 30], "judgment": "sign"}]}
    detections = [{"bbox_xyxy": [0, 0, 10, 10], "confidence": 0.9, "label": "P.130"},
                  {"bbox_xyxy": [40, 40, 50, 50], "confidence": 0.7, "label": "unknown"}]
    codes = {"a": {"label": "P.130", "confidence": 0.8},
             "b": {"label": "unknown", "confidence": 0.2}}
    result = review_frame(frame, detections, codes, [[[1, 2], [3, 4]]], "2026-10-28")
    assert result["reviewer"] == "ai-review (Codex)"
    assert result["human_confirmed"] is False
    assert result["review_method"]["type"] == "ai_reviewed"
    assert result["source_review_provenance"]["human_confirmed"] is True
    assert [s["judgment"] for s in result["signs"]] == ["sign", "uncertain", "uncertain"]
    assert result["signs"][0]["label"] == "P.130"
    assert all(s["parent_id"] is None and s["panel_role"] == "unknown" for s in result["signs"])
    assert result["lane_status"] == "annotated" and result["lanes"][0]["role"] == "other"
    assert frame["human_confirmed"] is True


def test_pilot_copy_keeps_original_review_and_verifies_image_bytes(tmp_path):
    import pytest
    copy_pilot = import_module("scripts.run_ai_review_20261028").copy_pilot
    source = tmp_path / "original"
    (source / "frames").mkdir(parents=True)
    content = b"pilot frame"
    (source / "frames/0000.jpg").write_bytes(content)
    frame = {"id": "frame_0000", "image": "frames/0000.jpg",
             "image_sha256": hashlib.sha256(content).hexdigest()}
    manifest = {"frames": [frame]}
    review = {"frames": [dict(frame, human_confirmed=True)],
              "review_method": {"type": "owner_bulk_approval_of_ai_boxes"}}
    (source / "review_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (source / "human_review.json").write_text(json.dumps(review), encoding="utf-8")
    dest = tmp_path / "copy"
    copy_pilot(source, dest)
    assert (dest / "frames/0000.jpg").read_bytes() == content
    assert json.loads((dest / "owner_review_source.json").read_text(encoding="utf-8")) == review
    assert json.loads((source / "human_review.json").read_text(encoding="utf-8")) == review
    with pytest.raises(FileExistsError):
        copy_pilot(source, dest)


def test_detector_match_does_not_override_owner_not_sign_judgment():
    review_frame = import_module("scripts.run_ai_review_20261028").review_frame
    frame = {"id": "f", "human_confirmed": True,
             "signs": [{"id": "a", "bbox_xyxy": [0, 0, 10, 10], "judgment": "not_sign"}]}
    pred = [{"bbox_xyxy": [0, 0, 10, 10], "confidence": 0.9}]
    result = review_frame(frame, pred, {}, [], "2026-10-28")
    assert result["signs"][0]["judgment"] == "uncertain"
    assert result["signs"][0]["source_annotation"]["judgment"] == "not_sign"
    assert result["lane_status"] == "uncertain"
