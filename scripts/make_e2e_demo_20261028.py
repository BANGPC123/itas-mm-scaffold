"""Add offline speech and research-only reference captions to a measured V0 segment."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE = "eb109f8c852df580611c236e2b313b01cf4a62662586ce71fa35525bdda05ac4"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_events(rows, start, duration, repeat_seconds=20.0):
    last, events = {}, []
    for row in rows:
        timestamp, text = row["timestamp"], row["alert"]
        if not text or not start <= timestamp < start + duration:
            continue
        if timestamp - last.get(text, float("-inf")) < repeat_seconds:
            continue
        events.append({"timestamp": timestamp - start, "text": text, "kind": row["alert_kind"]})
        last[text] = timestamp
    return events


def ass_time(seconds):
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def reference_cues(rows, start, duration):
    cues, last_end = [], -1.0
    for row in rows:
        timestamp = row["timestamp"] - start
        if not 0 <= timestamp < duration or timestamp < last_end or not row["references"]:
            continue
        ref = row["references"][0]
        doc = re.sub(r"[^A-Za-z0-9 .:/_-]", "", str(ref.get("document", "")))[:64]
        chunk = re.sub(r"[^A-Za-z0-9 .:/_-]", "", str(ref.get("chunk_id", "")))[:90]
        if not doc or not chunk:
            continue
        end = min(duration, timestamp + 4)
        cues.append({"start": timestamp, "end": end, "document": doc, "chunk_id": chunk,
                     "locator": ref.get("locator"), "source_url": ref.get("source_url"),
                     "legal_currency_verified": ref.get("legal_currency_verified")})
        last_end = end
    return cues


def write_captions(path, cues):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 752
WrapStyle: 2
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00102030,&H88000000,0,0,0,0,100,100,0,0,3,2,0,8,15,15,5,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for cue in cues:
        text = f"THAM KHAO TU DONG - CHUA KIEM CHUNG: {cue['document']} | {cue['chunk_id']}"
        lines.append(f"Dialogue: 0,{ass_time(cue['start'])},{ass_time(cue['end'])},Default,,0,0,0,,{text}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main():
    import imageio_ffmpeg
    from scripts.render_product_video import mix_speech
    from src.interaction.tts_engine import TtsEngine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tts-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT / "output/kiet_wp_20261028/wp4" not in output.parents or output.exists():
        raise ValueError("Output must be a new WP4 directory")
    if sha256(args.source) != EXPECTED_SOURCE or not args.tts_python.is_file():
        raise ValueError("Source SHA or TTS interpreter is invalid")
    temporal = ROOT / "output/kiet_wp_20261028/wp3/temporal_benchmark"
    metadata = json.loads((temporal / "temporal_benchmark.json").read_text(encoding="utf-8"))
    segment = metadata["segments"][0]
    event_path = temporal / "events_0.jsonl"
    video_path = temporal / "segment_0/test_video_offline_max.mp4"
    if (metadata["status"] != "COMPLETED" or segment["start_seconds"] != 0
            or segment["duration_seconds"] != 60 or sha256(event_path) != segment["events_sha256"]
            or sha256(video_path) != segment["video_sha256"]):
        raise ValueError("Temporal input provenance mismatch")
    rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    events, cues = select_events(rows, 0, 60), reference_cues(rows, 0, 60)
    if not events or not cues:
        raise ValueError("No logged speech or reference to include")
    output.mkdir(parents=True)
    write_captions(output / "legal_refs.ass", cues)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(temporal / "segment_0/silent.mp4"),
                    "-vf", "pad=iw:ih+32:0:32:color=0x0f1722,ass=legal_refs.ass", "-an", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "21", "-threads", "2", "-pix_fmt", "yuv420p", "silent.mp4"],
                   cwd=output, check=True, timeout=300)
    engine = TtsEngine({"tts": {"provider": "vieneu-onnx", "output_dir": str(output / "tts_cache"),
                                "python_path": str(args.tts_python.resolve()), "timeout_seconds": 120}})
    try:
        spoken, failures = mix_speech(events, engine, output, 60, max_failures=None)
    finally:
        engine.close()
    if not spoken or failures:
        raise RuntimeError(f"TTS synthesis incomplete: {len(spoken)} spoken, {len(failures)} failures")
    video = output / "test_video_annotated.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-i", str(video), "-map", "0:v:0", "-map", "0:a:0",
                    "-f", "null", "-"], check=True, timeout=180)
    report = {"status": "COMPLETED", "source_sha256": EXPECTED_SOURCE, "output_frames": 1800,
              "duration_seconds": 60, "video_sha256": sha256(video),
              "speech_sha256": sha256(output / "speech.wav"), "speech_events": spoken,
              "tts_failures": failures, "reference_cues": cues,
              "events_sha256": sha256(event_path), "input_video_sha256": segment["video_sha256"],
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "command": sys.argv, "tts_provider": "vieneu 3.8.3 ONNX int8 CPU",
              "limitations": ["Audio is sampled and deduplicated from logged pipeline observations, not all alerts.",
                              "References are automatically retrieved and research-only; locator and legal currency are unverified.",
                              "At least one returned P.130 chunk is associated with an inconsistent Article 84 locator.",
                              "No human event or lane ground truth is established."]}
    (output / "demo_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "speech_events": len(spoken), "reference_cues": len(cues)}))


if __name__ == "__main__":
    main()
