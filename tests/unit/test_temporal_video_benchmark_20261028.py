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
