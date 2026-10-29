import json

import pytest

from scripts.field_video_report_20261029 import aggregate_metrics, config_table, fmt, precision_cell, run_metrics, summarize_graded


def write_run(root, name, start, frames, report):
    folder = root / name
    (folder / f"segment_{start}").mkdir(parents=True)
    (folder / "temporal_benchmark.json").write_text(json.dumps({"segments": [report]}), encoding="utf-8")
    (folder / f"segment_{start}" / "predictions.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames), encoding="utf-8")
    return folder


def test_run_metrics_counts_tracks_confirmed_pairs_and_unknown_samples(tmp_path):
    frames = [
        {"timestamp": 0.0, "signs": [{"track_id": 1, "label": "unknown", "recognition_source": "temporal_pending"},
                                     {"track_id": 2, "label": "unknown", "recognition_source": "low_confidence"}]},
        {"timestamp": 0.2, "signs": [{"track_id": 1, "label": "P.131a", "recognition_source": "temporal_best_crop"},
                                     {"track_id": 3, "label": "P.130", "recognition_source": "single_frame"}]},
        {"timestamp": 0.4, "signs": [{"track_id": 1, "label": "P.131a", "recognition_source": "temporal_best_crop"}]},
    ]
    folder = write_run(tmp_path, "B_abcdefghijk_30", 30, frames, {"samples": 3, "alert_samples": 2, "alert_episodes": 1,
                                                                    "inference_fps": 3.0, "verified_full_decode": True})
    metrics = run_metrics(folder)
    assert metrics["tracks"] == 3 and metrics["confirmed_pairs"] == 1 and metrics["unknown_samples"] == 2
    assert metrics["alert_samples"] == 2 and metrics["call_seconds"] == pytest.approx(1.0)


def test_aggregate_metrics_weights_inference_speed_by_samples():
    rows = [{"samples": 300, "tracks": 10, "confirmed_pairs": 4, "unknown_samples": 5, "alert_samples": 1, "alert_episodes": 1, "call_seconds": 100.0},
            {"samples": 100, "tracks": 2, "confirmed_pairs": 1, "unknown_samples": 0, "alert_samples": 0, "alert_episodes": 0, "call_seconds": 50.0}]
    total = aggregate_metrics(rows)
    assert total["runs"] == 2 and total["tracks"] == 12 and total["inference_fps"] == pytest.approx(400 / 150)


def test_summarize_graded_and_table_text():
    rows = [{"verdict": "correct", "cluster": "a"}] * 6 + [{"verdict": "wrong_code", "cluster": "b"}] * 2 + [{"verdict": "uncertain", "cluster": "b"}]
    summary = summarize_graded(rows, lambda r: r["cluster"], n_boot=100)
    assert summary["precision"] == pytest.approx(0.75) and summary["uncertain"] == 1 and summary["n"] == 9
    table = config_table("Title", {"B": summary})
    assert "| B | new 0.87 | 9 | 6 | 2 | 0 | 1 | 0.750 [" in table
    assert fmt(None) == "-" and precision_cell({"precision": None, "ci_low": None, "ci_high": None}) == "-"
