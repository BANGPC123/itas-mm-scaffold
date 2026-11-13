"""Extract forward-warning review pairs from the recorded demo renders."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import file_digest
import json
from pathlib import Path

import cv2


REPO = Path(r"D:\DSP391m\project\itas-mm-scaffold")
DEMO = REPO / "output/demo_20261114"
OUTPUT = Path(__file__).resolve().parents[1] / "output/demo_20261114/M11"
CLIPS = {
    "v1": ("M5", ("night_90s", "demo_video")),
    "v2": ("M10", ("night_90s", "demo_video", "demo_short_30s")),
}
FIELDS = (
    "id", "timestamp", "output_timestamp", "prior_output_timestamp", "track_id", "class",
    "level", "bbox_xyxy", "output_bbox_xyxy", "ttc_proxy_s", "event_frame", "prior_frame",
    "ai_verdict", "ai_reason", "owner_verdict",
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def render_dir(stage: str, clip: str) -> Path:
    folder = DEMO / stage / "clips" / clip
    return folder if (folder / "summary.json").is_file() else folder / "render"


def source_size(summary: dict) -> tuple[int, int]:
    path = REPO / summary["source_file"]
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE])
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not read source video geometry: {path}")
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()


def event_key(alert: dict) -> str:
    return json.dumps(alert, sort_keys=True, ensure_ascii=False)


def matched_events(path: Path, alerts: list[dict]) -> list[tuple[dict, int]]:
    events = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            for alert in sample.get("forward_alerts", []):
                if abs(alert["t"] - sample["timestamp"]) > 0.001:
                    raise ValueError(f"Alert and prediction timestamps disagree in {path}")
                events.append((alert, sample["source_frame"]))
    if Counter(event_key(a) for a in alerts) != Counter(event_key(a) for a, _ in events):
        raise ValueError(f"Summary and predictions contain different alerts: {path}")
    return events


def read_frames(path: Path, targets: set[int], size: tuple[int, int], fps: float) -> dict:
    frames = {}
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE])
    try:
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - fps) > 0.01:
            raise RuntimeError(f"Could not open render or render FPS differs: {path}")
        for number in sorted(targets):
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, number):
                raise RuntimeError(f"Could not seek to frame {number}: {path}")
            ok, frame = cap.read()
            if not ok or frame.shape[1::-1] != size or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != number + 1:
                raise RuntimeError(f"Could not decode frame {number} at expected size: {path}")
            frames[number] = frame
    finally:
        cap.release()
    return frames


def write_image(path: Path, frame) -> None:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    with path.open("xb") as stream:
        stream.write(encoded.tobytes())


def build(stage: str, clip: str, out: Path) -> None:
    render = render_dir(stage, clip)
    summary_path = render / "summary.json"
    predictions = render / "predictions.jsonl"
    video = render / "test_video_offline_max.mp4"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    video_sha = digest(video)
    if video_sha != summary["video_sha256"] or not summary["verified_full_decode"]:
        raise ValueError(f"Render is not the verified video in {summary_path}")
    events = matched_events(predictions, summary["forward_warning"]["alerts"])
    fps = summary["fps"]
    out_fps = summary["output_fps"]
    step = round(fps / out_fps)
    if step < 1 or abs(fps / out_fps - step) > 1e-6:
        raise ValueError(f"Unexpected frame mapping in {summary_path}")
    source_start = int(summary["start_sec"] * fps)
    size = tuple(summary["output_size"])
    sw, sh = source_size(summary)
    if min(sw, sh) < 1:
        raise ValueError(f"Invalid source video dimensions: {summary_path}")
    frame_pairs = []
    for _, source_frame in events:
        event, remainder = divmod(source_frame - source_start, step)
        if remainder or not 0 <= event < summary["output_frames"]:
            raise ValueError(f"Alert frame {source_frame} is missing from {video}")
        prior = max(0, round(event - 0.5 * out_fps))
        frame_pairs.append((event, prior))
    frames = read_frames(video, {frame for pair in frame_pairs for frame in pair}, size, out_fps)
    out.mkdir(parents=True)
    with (out / "review_sheet.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for index, ((alert, _), (event, prior)) in enumerate(zip(events, frame_pairs, strict=True), 1):
            identifier = f"{index:03d}"
            event_name, prior_name = f"{identifier}_event.jpg", f"{identifier}_prior.jpg"
            bbox = alert["bbox_xyxy"]
            w, h = size
            scales = (w / sw, h / sh, w / sw, h / sh)
            bounds = (w, h, w, h)
            mapped = [max(0, min(limit - 1, round(value * scale)))
                      for value, scale, limit in zip(bbox, scales, bounds, strict=True)]
            x1, y1, x2, y2 = mapped
            frame = frames[event].copy()
            # The cyan rectangle marks the source alert box in render coordinates.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            write_image(out / event_name, frame)
            write_image(out / prior_name, frames[prior])
            writer.writerow({
                "id": identifier, "timestamp": alert["t"], "output_timestamp": f"{event/out_fps:.6f}",
                "prior_output_timestamp": f"{prior/out_fps:.6f}", "track_id": alert["track_id"],
                "class": alert["cls"], "level": alert["level"], "bbox_xyxy": json.dumps(bbox),
                "output_bbox_xyxy": json.dumps(mapped), "ttc_proxy_s": alert.get("ttc_proxy_s"),
                "event_frame": event_name, "prior_frame": prior_name,
            })
    provenance = {
        "summary": str(summary_path), "summary_sha256": digest(summary_path),
        "predictions": str(predictions), "predictions_sha256": digest(predictions),
        "video": str(video), "video_sha256": video_sha, "source_size": [sw, sh],
        "output_size": list(size), "output_fps": out_fps,
        "alerts": len(events), "frame_pairs": frame_pairs,
    }
    with (out / "sources.json").open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)
        stream.write("\n")
    print(f"{clip}: {len(events)} alert pairs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=CLIPS)
    args = parser.parse_args()
    stage, clips = CLIPS[args.version]
    for clip in clips:
        build(stage, clip, OUTPUT / args.version / clip)


if __name__ == "__main__":
    main()
