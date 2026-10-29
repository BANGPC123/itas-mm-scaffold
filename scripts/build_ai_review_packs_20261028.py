"""Build traceable AI review inputs from canonical TEST frames."""

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

LOC = Path("processed/canonical/itas_mm_v1/localization")
CLIPS = {"4017", "4022", "4024", "4032", "4047", "4071", "4094", "4112", "4135", "4186", "4189"}


def select_frames(rows, clips):
    """Choose one TEST image for each clip/frame index, preserving duplicates."""
    candidates = defaultdict(list)
    for row in rows:
        match = re.search(r"IMG_(\d{4})-MOV_out(\d+)", row["representative_source_path"], re.I)
        if row["split"] != "TEST" or not match or match.group(1) not in clips:
            continue
        candidates[(match.group(1), int(match.group(2)))].append(row)
    result = []
    for (clip, number), group in sorted(candidates.items()):
        group = sorted(group, key=lambda r: (r["image_sha256"].lower(), r["representative_source_path"]))
        result.append({"clip": "IMG_" + clip, "frame_number": number, "row": group[0],
                       "candidate_sha256": sorted({r["image_sha256"].lower() for r in group}),
                       "candidate_source_paths": sorted({r["representative_source_path"] for r in group})})
    return result


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_pack(dataset_root, output, clips):
    dataset_root, output = Path(dataset_root).resolve(), Path(output).resolve()
    if output == dataset_root or dataset_root in output.parents:
        raise ValueError("Output cannot be inside the source dataset")
    if output.exists():
        raise FileExistsError(output)
    loc = (dataset_root / LOC).resolve()
    selected = select_frames(read_csv(loc / "samples.csv"), clips)
    if {item["clip"][4:] for item in selected} != set(clips):
        raise ValueError("Missing selected clip")
    selected_ids = {item["row"]["sample_id"] for item in selected}
    boxes_by_sample = defaultdict(list)
    for box in read_csv(loc / "boxes.csv"):
        if box["sample_id"] in selected_ids:
            boxes_by_sample[box["sample_id"]].append(box)

    frames = []
    copies = []
    for item in selected:
        row = item["row"]
        source = (loc / row["image_path"]).resolve()
        if loc not in source.parents or not source.is_file():
            raise ValueError("Invalid canonical image path")
        digest = file_sha256(source)
        if digest != row["image_sha256"].lower():
            raise ValueError("Canonical image SHA256 mismatch")
        width, height = int(row["image_width_px"]), int(row["image_height_px"])
        boxes = boxes_by_sample[row["sample_id"]]
        if row.get("canonical_box_count") and len(boxes) != int(row["canonical_box_count"]):
            raise ValueError("Canonical box count mismatch")
        signs = []
        for box in sorted(boxes, key=lambda b: b["canonical_box_id"]):
            xyxy = [float(value) for value in json.loads(box["canonical_xyxy_normalized_json"])]
            if len(xyxy) != 4 or not (0 <= xyxy[0] < xyxy[2] <= 1 and 0 <= xyxy[1] < xyxy[3] <= 1):
                raise ValueError("Invalid canonical box")
            signs.append({"id": box["canonical_box_id"],
                          "bbox_xyxy": [xyxy[0]*width, xyxy[1]*height, xyxy[2]*width, xyxy[3]*height],
                          "label": "unknown", "judgment": "unreviewed", "panel_role": "unknown",
                          "parent_id": None, "source": "canonical_box"})
        relpath = f"frames/{item['clip']}_out{item['frame_number']:04d}.jpg"
        frames.append({"id": f"{item['clip']}_out{item['frame_number']:04d}",
                       "source_group": item["clip"], "frame_number": item["frame_number"],
                       "timestamp_seconds": None, "image": relpath, "image_sha256": digest,
                       "width": width, "height": height, "reviewer": "", "reviewed_at": None,
                       "human_confirmed": False, "all_visible_signs_annotated": False,
                       "lane_status": "unreviewed", "signs": signs, "lanes": [], "notes": "",
                       "canonical_sample_id": row["sample_id"],
                       "representative_source_path": row["representative_source_path"],
                       "candidate_sha256": item["candidate_sha256"],
                       "candidate_source_paths": item["candidate_source_paths"]})
        copies.append((source, relpath, digest))
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    for source, relpath, digest in copies:
        destination = output / relpath
        shutil.copyfile(source, destination)
        if file_sha256(destination) != digest:
            raise ValueError("Copied image SHA256 mismatch")
    manifest = {"schema_version": "itasmm-field-review-v1",
                "provenance": "Canonical TEST images and source boxes; AI review not yet performed.",
                "selection_rule": "One image per clip/out index, lowest SHA256; index is not a timestamp.",
                "frames": frames}
    (output / "review_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_pack(args.dataset_root, args.output, CLIPS)
    print(json.dumps({"frames": len(manifest["frames"]), "clips": len(CLIPS)}))


if __name__ == "__main__":
    main()
