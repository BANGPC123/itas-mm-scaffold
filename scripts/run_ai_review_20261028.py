"""Create traceable AI review proposals; these are not human ground truth."""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from src.evaluation.field_metrics import associate

ROOT = Path(__file__).resolve().parents[1]
WP2_ROOT = ROOT / "output/kiet_wp_20261028/wp2"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_pilot(source, output):
    """Copy only validated pilot inputs and retain the owner export unchanged."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output.exists():
        raise FileExistsError(output)
    manifest_path, review_path = source / "review_manifest.json", source / "human_review.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    owner = json.loads(review_path.read_text(encoding="utf-8"))
    frames = manifest["frames"]
    reviewed = {f["id"]: f for f in owner["frames"]}
    if len(reviewed) != len(frames) or len({f["id"] for f in frames}) != len(frames):
        raise ValueError("Pilot frame IDs are not unique or do not match")
    files = []
    for frame in frames:
        path = (source / frame["image"]).resolve()
        if source not in path.parents or not path.is_file():
            raise ValueError("Invalid pilot frame path")
        if frame["id"] not in reviewed or reviewed[frame["id"]]["image_sha256"] != frame["image_sha256"]:
            raise ValueError("Pilot review provenance mismatch")
        if file_sha256(path) != frame["image_sha256"].lower():
            raise ValueError("Pilot image hash mismatch")
        files.append((path, frame["image"], frame["image_sha256"].lower()))
    output.mkdir(parents=True)
    for path, relative, digest in files:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if file_sha256(destination) != digest:
            raise ValueError("Pilot copied image hash mismatch")
    shutil.copyfile(manifest_path, output / "review_manifest.json")
    shutil.copyfile(review_path, output / "owner_review_source.json")


def review_frame(frame, detections, codes, polylines, reviewed_at):
    """Combine source boxes and detector proposals without inventing negatives."""
    result = deepcopy(frame)
    original_signs = frame.get("signs", [])
    matches = associate(original_signs, detections, threshold=0.5)
    by_source = {i: (j, score) for i, j, score in matches}
    matched_detections = {j for _, j, _ in matches}
    result["source_review_provenance"] = {
        key: frame.get(key) for key in
        ("reviewer", "reviewed_at", "human_confirmed", "all_visible_signs_annotated")
    }
    reviewed = []
    for index, source in enumerate(original_signs):
        item = deepcopy(source)
        item["source_annotation"] = {key: source.get(key) for key in
                                     ("judgment", "label", "panel_role", "parent_id")}
        matched = by_source.get(index)
        conflict = frame.get("human_confirmed") is True and source.get("judgment") in {"not_sign", "uncertain"}
        item.update(judgment="sign" if matched and not conflict else "uncertain", panel_role="unknown", parent_id=None)
        code = codes.get(source["id"], {})
        item["label"] = code.get("label", "unknown")
        item["recognition_confidence"] = code.get("confidence")
        item["review_reason"] = ("detector conflicts with owner judgment; unresolved" if conflict else
                                 "source box agrees with detector at IoU >= 0.5" if matched else
                                 "source box has no detector match; unresolved")
        if matched:
            item["detector_match"] = {"index": matched[0], "iou": matched[1]}
        reviewed.append(item)
    for index, detection in enumerate(detections):
        if index in matched_detections:
            continue
        reviewed.append({"id": f"ai_unmatched_{index}", "bbox_xyxy": list(detection["bbox_xyxy"]),
                         "label": detection.get("label", "unknown"), "judgment": "uncertain",
                         "panel_role": "unknown", "parent_id": None,
                         "detector_confidence": detection.get("confidence"),
                         "review_reason": "detector proposal has no source-box match; unresolved"})
    result.update(reviewer="ai-review (Codex)", reviewed_at=reviewed_at, human_confirmed=False,
                  all_visible_signs_annotated=False, signs=reviewed,
                  lane_status="annotated" if polylines else "uncertain",
                  lanes=[{"role": "other", "points": [list(p) for p in points]} for points in polylines],
                  review_method={"type": "ai_reviewed",
                                 "reason": "model proposals compared with source boxes; no human confirmation"},
                  detector_proposals=deepcopy(detections))
    return result


def run_review(pack, source_review, result_dir, limit=None):
    """Run frozen models on copied images and write per-frame AI proposals."""
    import cv2
    import torch
    import ultralytics
    from src.evaluation.label_review_schema import validate_review
    from src.perception.lane_detector import LaneDetector
    from src.perception.sign_detector import SignDetector
    from src.perception.sign_recognizer import SignRecognizer
    from src.utils.config import load_config
    from src.utils.types import SignDetection

    pack, source_review, result_dir = Path(pack).resolve(), Path(source_review).resolve(), Path(result_dir).resolve()
    allowed_root = WP2_ROOT.resolve()
    if allowed_root not in pack.parents or allowed_root not in result_dir.parents:
        raise ValueError("Pack and result directory must be inside WP2 output")
    if result_dir.exists():
        raise FileExistsError(result_dir)
    manifest_path = pack / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = json.loads(source_review.read_text(encoding="utf-8"))
    validate_review(source, manifest)
    frames = source["frames"]
    if limit is not None and not 1 <= limit <= len(frames):
        raise ValueError("Invalid limit")
    selected = frames[:limit] if limit else frames
    cfg = load_config("perception")
    cfg["sign_detector"]["weights_path"] = str(ROOT / "models/sign_detector/finetuned_20261026.pt")
    cfg["sign_recognizer"]["weights_path"] = str(ROOT / "models/sign_recognizer/retrained_20261009.pt")
    for section, key in (("lane_detector", "weights_path"), ("sign_recognizer", "reference_gate_path")):
        if cfg[section].get(key):
            cfg[section][key] = str((ROOT / cfg[section][key]).resolve())
    if cfg["lane_detector"].get("backend") != "ufld_v2":
        raise ValueError("Configured lane backend is not UFLD-v2")
    paths = {"detector": Path(cfg["sign_detector"]["weights_path"]),
             "recognizer": Path(cfg["sign_recognizer"]["weights_path"]),
             "lane": Path(cfg["lane_detector"]["weights_path"])}
    if cfg["sign_recognizer"].get("reference_gate_path"):
        paths["recognizer_reference_gate"] = Path(cfg["sign_recognizer"]["reference_gate_path"])
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("Required frozen checkpoint or reference gate missing")
    provenance = {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "input_manifest_sha256": file_sha256(manifest_path),
                  "source_review_sha256": file_sha256(source_review),
                  "perception_config_sha256": file_sha256(ROOT / "configs/perception.yaml"),
                  "runner_sha256": file_sha256(Path(__file__)),
                  "model_sha256": {name: file_sha256(path) for name, path in paths.items()},
                  "matching": "greedy one-to-one IoU >= 0.5",
                  "detector_confidence": cfg["sign_detector"]["confidence_threshold"],
                  "detector_nms_iou": cfg["sign_detector"]["iou_threshold"],
                  "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                  "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                  "command": sys.argv, "expected_frames": len(frames), "requested_frames": len(selected)}
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    detector, recognizer, lane = SignDetector(cfg), SignRecognizer(cfg), LaneDetector(cfg)
    if detector._model is None or recognizer.model is None or lane.learned is None:
        raise RuntimeError("A required model failed to load")
    result_dir.mkdir(parents=True)
    (result_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    reviewed = []
    started = time.perf_counter()
    with (result_dir / "review_frames.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for index, frame in enumerate(selected, 1):
            image_path = (pack / frame["image"]).resolve()
            if pack not in image_path.parents or file_sha256(image_path) != frame["image_sha256"].lower():
                raise ValueError("Copied image path or SHA256 mismatch")
            image = cv2.imread(str(image_path))
            if image is None or image.shape[:2] != (frame["height"], frame["width"]):
                raise ValueError("Image decode or dimensions mismatch")
            predictions = detector.detect(image)
            recognizer.recognize(image, predictions)
            detections = [{"bbox_xyxy": list(p.bbox_xyxy), "confidence": p.confidence,
                           "label": p.label, "recognition_confidence": p.recognition_confidence,
                           "recognition_source": p.recognition_source} for p in predictions]
            source_signs = [SignDetection(label="unknown", confidence=1.0,
                                         bbox_xyxy=tuple(s["bbox_xyxy"])) for s in frame["signs"]]
            recognizer.recognize(image, source_signs)
            codes = {s["id"]: {"label": p.label, "confidence": p.recognition_confidence}
                     for s, p in zip(frame["signs"], source_signs)}
            lane_result = lane.detect(image)
            if lane_result.backend != "ufld_v2":
                raise RuntimeError("Lane inference backend unavailable")
            reviewed_frame = review_frame(frame, detections, codes, lane_result.polylines,
                                          datetime.now().astimezone().isoformat())
            reviewed_frame["lane_inference_reason"] = lane_result.rejection_reason
            reviewed.append(reviewed_frame)
            stream.write(json.dumps(reviewed_frame, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 10 == 0 or index == len(selected):
                print(json.dumps({"processed": index, "total": len(selected),
                                  "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
    result = {"schema_version": "itasmm-field-review-v1", "frames": reviewed,
              "review_method": {"type": "ai_reviewed", "reviewer": "ai-review (Codex)",
                                "reason": "automated model/source-box comparison, without independent visual or human review"},
              "source_review_method": source.get("review_method"), "provenance": provenance,
              "status": "COMPLETED" if len(reviewed) == len(frames) else "PARTIAL"}
    validate_review(result, manifest)
    (result_dir / "ai_review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--source-review", type=Path)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    pack = args.pack.resolve()
    source = args.source_review.resolve() if args.source_review else pack / "review_manifest.json"
    output = args.result_dir.resolve()
    os.chdir(ROOT)
    result = run_review(pack, source, output, args.limit)
    print(json.dumps({"status": result["status"], "frames": len(result["frames"])}), flush=True)


if __name__ == "__main__":
    main()
