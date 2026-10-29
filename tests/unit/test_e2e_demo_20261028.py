from scripts.make_e2e_demo_20261028 import reference_cues, select_events


def test_select_events_uses_only_logged_alerts_and_applies_repeat_interval():
    rows = [
        {"timestamp": 10.0, "alert": "A", "alert_kind": "ai_observation"},
        {"timestamp": 11.0, "alert": "A", "alert_kind": "ai_observation"},
        {"timestamp": 12.0, "alert": "B", "alert_kind": "verified_speech"},
        {"timestamp": 31.0, "alert": "A", "alert_kind": "ai_observation"},
        {"timestamp": 32.0, "alert": "", "alert_kind": "none"},
    ]
    events = select_events(rows, start=10.0, duration=60.0, repeat_seconds=20.0)
    assert [(e["timestamp"], e["text"], e["kind"]) for e in events] == [
        (0.0, "A", "ai_observation"), (2.0, "B", "verified_speech"),
        (21.0, "A", "ai_observation"),
    ]


def test_reference_cues_keep_locator_unverified_and_bound_overlaps():
    ref = {"document": "QCVN 41:2024/BGTVT", "chunk_id": "doc:194:abc",
           "locator": "Article 84", "legal_currency_verified": False}
    rows = [{"timestamp": 1.0, "references": [ref]}, {"timestamp": 2.0, "references": [ref]},
            {"timestamp": 5.0, "references": [ref]}]
    cues = reference_cues(rows, 0, 60)
    assert [(cue["start"], cue["end"]) for cue in cues] == [(1.0, 5.0), (5.0, 9.0)]
    assert all(not cue["legal_currency_verified"] for cue in cues)
    assert cues[0]["locator"] == "Article 84"
