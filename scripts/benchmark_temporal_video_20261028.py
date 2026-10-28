"""Measure sampled offline-video behavior using explicit event definitions."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VIDEO_SHA256 = "eb109f8c852df580611c236e2b313b01cf4a62662586ce71fa35525bdda05ac4"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def apply_threshold(recognizer, threshold):
    """Override a checkpoint threshold in memory only, when explicitly requested."""
    if threshold is not None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between zero and one")
        recognizer.threshold = float(threshold)
    return recognizer.threshold


def recognition_snapshot(signs, image_shape):
    """Copy the merged crop decisions before temporal confirmation mutates them."""
    from src.perception.sign_multiscale import crop_side
    return [{"bbox_xyxy": list(sign.bbox_xyxy), "short_side_px": crop_side(sign.bbox_xyxy, image_shape),
             "label": sign.label, "detector_confidence": float(sign.confidence),
             "recognition_confidence": (float(sign.recognition_confidence)
                                        if sign.recognition_confidence is not None else None),
             "recognition_source": sign.recognition_source} for sign in signs]


def summarize_events(events, max_crop_age_ms=750):
    """Count consecutive repeats and samples with an over-age recognition crop."""
    previous_warnings, previous_alert = set(), ""
    warning_count = repeat_warnings = alert_samples = alert_episodes = repeat_alerts = stale = 0
    elapsed_ms = 0.0
    last_timestamp = None
    for event in events:
        timestamp = event["timestamp"]
        if last_timestamp is not None and timestamp <= last_timestamp:
            raise ValueError("Event timestamps must increase")
        last_timestamp = timestamp
        warnings = event["warnings"]
        warning_count += len(warnings)
        repeat_warnings += sum(message in previous_warnings for message in warnings)
        previous_warnings = set(warnings)
        alert = event["alert"]
        if alert:
            alert_samples += 1
            if alert == previous_alert:
                repeat_alerts += 1
            else:
                alert_episodes += 1
            if any(age is not None and age > max_crop_age_ms for age in event["crop_ages_ms"]):
                stale += 1
        previous_alert = alert
        elapsed_ms += event["call_ms"]
    return {"samples": len(events), "warning_occurrences": warning_count,
            "consecutive_repeated_warning_occurrences": repeat_warnings,
            "alert_samples": alert_samples, "alert_episodes": alert_episodes,
            "consecutive_repeated_alert_samples": repeat_alerts, "stale_alert_samples": stale,
            "inference_fps": len(events) * 1000 / elapsed_ms if elapsed_ms else None}


class AuditPipeline:
    def __init__(self, pipeline, event_path, trace_raw_recognition=False):
        self.pipeline = pipeline
        self.events = []
        self.event_path = event_path
        self.trace_raw_recognition = trace_raw_recognition

    def run(self, *args, **kwargs):
        started = time.perf_counter()
        if self.trace_raw_recognition:
            from src.perception import sign_multiscale
            raw_direct, raw_merged = [], []
            original_recognize = self.pipeline.sign_recognizer.recognize
            original_views = sign_multiscale.recognize_views

            def capture_recognize(image, *inner_args, **inner_kwargs):
                detected = original_recognize(image, *inner_args, **inner_kwargs)
                raw_direct.append(recognition_snapshot(detected, image.shape))
                return detected

            def capture_views(recognizer, image, *inner_args, **inner_kwargs):
                detected = original_views(recognizer, image, *inner_args, **inner_kwargs)
                raw_merged.append(recognition_snapshot(detected, image.shape))
                return detected

            with patch.object(self.pipeline.sign_recognizer, "recognize", side_effect=capture_recognize), \
                 patch.object(sign_multiscale, "recognize_views", side_effect=capture_views):
                result = self.pipeline.run(*args, **kwargs)
            raw = raw_merged[-1] if raw_merged else raw_direct[-1] if raw_direct else None
        else:
            result = self.pipeline.run(*args, **kwargs)
            raw = None
        elapsed_ms = (time.perf_counter() - started) * 1000
        speech = result.reasoning.speech_text
        observation = result.reasoning.observation_speech_text
        signs = [{"track_id": s.track_id, "label": s.label, "crop_age_ms": s.crop_age_ms,
                  "recognition_source": s.recognition_source} for s in result.perception.signs]
        event = {"timestamp": kwargs["frame_timestamp"], "warnings": result.warnings,
                 "alert": speech or observation, "alert_kind": "verified_speech" if speech else "ai_observation" if observation else "none",
                 "crop_ages_ms": [s["crop_age_ms"] for s in signs], "signs": signs,
                 "references": result.reasoning.references, "call_ms": elapsed_ms,
                 "pipeline_timings_ms": result.timings_ms}
        if self.trace_raw_recognition:
            event["pre_temporal_signs"] = raw
        self.events.append(event)
        with self.event_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        return result


def run(source, output, starts, duration, analysis_hz, recognizer=None, reference_gate=None, threshold=None,
        trace_raw_recognition=False):
    import torch
    from src.pipeline import orchestrator
    from scripts.render_offline_video_max_20261014 import render

    source, output = Path(source).resolve(), Path(output).resolve()
    if ROOT / "output/kiet_wp_20261028/wp3" not in output.parents or output.exists():
        raise ValueError("Output must be a new directory inside WP3 output")
    if sha256(source) != EXPECTED_VIDEO_SHA256:
        raise ValueError("V0 SHA256 does not match registered source")
    output.mkdir(parents=True)
    original_load = orchestrator.load_config
    perception = original_load("perception")
    # --recognizer evaluates a candidate without editing configs/perception.yaml; the gate is dropped unless one is given.
    checkpoints = {"detector": ROOT / "models/sign_detector/finetuned_20261026.pt",
                   "recognizer": Path(recognizer).resolve() if recognizer else ROOT / "models/sign_recognizer/retrained_20261009.pt",
                   "lane": ROOT / "models/lane/selected_20261009.pth"}
    if recognizer:
        perception["sign_recognizer"]["weights_path"] = str(checkpoints["recognizer"])
        perception["sign_recognizer"].pop("reference_gate_path", None)
        if reference_gate:
            perception["sign_recognizer"]["reference_gate_path"] = str(Path(reference_gate).resolve())
    for section, expected in (("sign_detector", checkpoints["detector"]),
                              ("sign_recognizer", checkpoints["recognizer"]),
                              ("lane_detector", checkpoints["lane"])):
        if (ROOT / perception[section]["weights_path"]).resolve() != expected.resolve():
            raise ValueError(f"Unexpected configured checkpoint: {section}")
    if perception["lane_detector"].get("backend") != "ufld_v2" or original_load("reasoning").get("validated_corpus"):
        raise ValueError("Expected UFLD-v2 and offline legal retrieval configuration")

    def load_scoped(name):
        cfg = original_load(name)
        if name == "perception" and recognizer:
            cfg["sign_recognizer"] = perception["sign_recognizer"]
        if name == "interaction":
            cfg["tts"]["output_dir"] = str(output / "unused_tts_cache")
        return cfg

    with patch.object(orchestrator, "load_config", side_effect=load_scoped):
        pipeline = orchestrator.Pipeline()
    effective_threshold = apply_threshold(pipeline.sign_recognizer, threshold)
    if pipeline.sign_detector._model is None or pipeline.sign_recognizer.model is None or pipeline.lane_detector.learned is None:
        raise RuntimeError("Pipeline models unavailable")
    results = []
    try:
        for start in starts:
            event_path = output / f"events_{start}.jsonl"
            audit = AuditPipeline(pipeline, event_path, trace_raw_recognition)
            begin = time.perf_counter()
            summary = render(source, output / f"segment_{start}", analysis_hz=analysis_hz,
                             start_second=start, max_seconds=duration, pipeline=audit)
            wall = time.perf_counter() - begin
            stats = summarize_events(audit.events)
            stats.update(start_seconds=start, duration_seconds=duration,
                         rendered_frames=summary["output_frames"], render_wall_seconds=wall,
                         render_fps=summary["output_frames"] / wall,
                         events_sha256=sha256(event_path), video_sha256=summary["video_sha256"],
                         verified_full_decode=summary["verified_full_decode"],
                         samples_with_references=sum(bool(e["references"]) for e in audit.events))
            results.append(stats)
    finally:
        pipeline.tts_engine.close()
        if pipeline.supplementary_reader is not None:
            pipeline.supplementary_reader.close()
    report = {"status": "COMPLETED" if duration == 60 and starts == [0, 390, 780] else "PARTIAL",
              "segments": results,
              "definitions": {"warnings": "all pipeline warning entries, including repeated status and safety notices",
                              "duplicate_warnings": "same warning text in consecutive sampled frames; not verified false alerts",
                              "alerts": "nonempty verified speech text, otherwise nonempty AI observation speech text",
                              "duplicate_alerts": "identical alert text in consecutive sampled frames",
                              "stale_alerts": "alert samples with any sign best-crop age > 750 ms; a conservative proxy, not proof that a particular alert is stale",
                              "inference_fps": "sample count divided by wall time inside Pipeline.run",
                              "render_fps": "output frames divided by end-to-end render wall time, including inference and verification"},
              "limitations": ["No human event ground truth; repeated status messages are not necessarily erroneous.",
                              "No per-render-frame overlay age is logged; stale alert count is a crop-age proxy.",
                              "TTS was disabled during temporal measurement; output preserves source audio only."],
              "provenance": {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                             "source_sha256": EXPECTED_VIDEO_SHA256,
                             "config_sha256": {name: sha256(ROOT / f"configs/{name}.yaml")
                                               for name in ("perception", "context", "reasoning", "interaction")},
                             "model_sha256": {name: sha256(path) for name, path in checkpoints.items()},
                             "runner_sha256": sha256(Path(__file__)),
                             "recognizer_override": str(checkpoints["recognizer"]) if recognizer else None,
                             "recognizer_threshold": effective_threshold,
                             "trace_raw_recognition": trace_raw_recognition,
                             "reference_gate": (str(reference_gate) if reference_gate else None) if recognizer else perception["sign_recognizer"].get("reference_gate_path"),
                             "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                             "torch": torch.__version__, "analysis_hz": analysis_hz, "warmup_frames": 0,
                             "command": sys.argv, "tts_output_override": "WP3 output; synthesis disabled"}}
    (output / "temporal_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", nargs="+", type=int, default=[0, 390, 780])
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--analysis-hz", type=float, default=6)
    parser.add_argument("--recognizer", type=Path, help="candidate recognizer checkpoint (config left unchanged)")
    parser.add_argument("--reference-gate", type=Path, help="reference gate for the candidate; omitted means no gate")
    parser.add_argument("--threshold", type=float, help="explicit in-memory recognition threshold")
    parser.add_argument("--trace-raw-recognition", action="store_true", help="log merged crop decisions before temporal confirmation")
    args = parser.parse_args()
    result = run(args.source, args.output, args.starts, args.duration, args.analysis_hz, args.recognizer, args.reference_gate,
                 args.threshold, args.trace_raw_recognition)
    print(json.dumps({"status": result["status"], "segments": len(result["segments"])}), flush=True)


if __name__ == "__main__":
    main()
