from importlib import import_module


def test_temporal_counts_repeat_and_stale_by_explicit_definitions():
    summarize = import_module("scripts.benchmark_temporal_video_20261028").summarize_events
    events = [
        {"timestamp": 0.0, "warnings": ["persistent"], "alert": "A", "crop_ages_ms": [0], "call_ms": 50},
        {"timestamp": 0.2, "warnings": ["persistent"], "alert": "A", "crop_ages_ms": [800], "call_ms": 50},
        {"timestamp": 0.4, "warnings": ["persistent", "new"], "alert": "", "crop_ages_ms": [], "call_ms": 50},
    ]
    result = summarize(events, max_crop_age_ms=750)
    assert result["warning_occurrences"] == 4
    assert result["consecutive_repeated_warning_occurrences"] == 2
    assert result["alert_samples"] == 2 and result["alert_episodes"] == 1
    assert result["consecutive_repeated_alert_samples"] == 1
    assert result["stale_alert_samples"] == 1
    assert result["inference_fps"] == 20


def test_threshold_override_is_explicit_and_preserves_default():
    from types import SimpleNamespace
    import pytest
    from scripts.benchmark_temporal_video_20261028 import apply_threshold

    recognizer = SimpleNamespace(threshold=0.87)
    assert apply_threshold(recognizer, None) == 0.87
    assert apply_threshold(recognizer, 0.641) == 0.641
    with pytest.raises(ValueError):
        apply_threshold(recognizer, 1.01)


def test_recognition_snapshot_preserves_pre_temporal_labels_and_clipped_size():
    from types import SimpleNamespace
    from scripts.benchmark_temporal_video_20261028 import recognition_snapshot

    sign = SimpleNamespace(bbox_xyxy=(-2.0, 1.0, 12.0, 15.0), label="P.127",
                           confidence=0.8, recognition_confidence=0.9,
                           recognition_source="single_frame")
    snapshot = recognition_snapshot([sign], (100, 100, 3))
    sign.label = "unknown"
    assert snapshot[0]["label"] == "P.127" and snapshot[0]["short_side_px"] == 12
