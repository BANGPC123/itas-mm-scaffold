import csv
import random

import pytest

from scripts.field_video_eval_20261029 import (
    SOURCE_FIELDS, bootstrap_precision, decide_recommendations, iou, precision, proxy_recall,
    read_sources, select_segments, summarize_review, tally, verdict)


def write_sources(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{field: "" for field in SOURCE_FIELDS}, **row})


def test_read_sources_types_and_roles(tmp_path):
    path = tmp_path / "sources.csv"
    write_sources(path, [
        {"video_id": "aaa", "role": "eval", "duration_s": "3321", "fps": "60.0", "near_duplicate_rate": "0.0",
         "segments_used": "30;880;1730;2580;3231"},
        {"video_id": "bbb", "role": "rejected", "duration_s": "100", "fps": "30", "near_duplicate_rate": "0.02",
         "segments_used": ""}])
    rows = read_sources(path)
    assert [row["video_id"] for row in rows] == ["aaa", "bbb"]
    assert rows[0]["duration_s"] == 3321.0 and rows[0]["fps"] == 60.0
    assert rows[0]["segments_used"] == [30, 880, 1730, 2580, 3231]
    assert rows[1]["segments_used"] == []
    assert rows[1]["near_duplicate_rate"] == 0.02


def test_build_source_row_and_roundtrip(tmp_path):
    from scripts.field_video_eval_20261029 import build_source_row, write_sources
    meta = {"id": "abc", "webpage_url": "https://www.youtube.com/watch?v=abc", "channel": "Some Channel ",
            "upload_date": "20250113"}
    criteria = {key: f"pass: {key}" for key in ("a_view", "b_speed", "c_resolution", "d_overlay", "e_signs", "f_duration")}
    screening = {"role": "eval", "city": "Hanoi", "day_night": "day", "criteria": criteria, "notes": "n"}
    measured = {"width": 1920, "height": 1080, "fps": 60.0, "frames": 199237, "duration_s": 3320.6}
    row = build_source_row(meta, screening, "ff" * 32, {"rate_any": 0.0}, measured)
    assert row["channel"] == "Some Channel" and row["resolution"] == "1920x1080"
    assert row["segments_used"] == "30;830;1630;2430;3230"
    harvest = build_source_row({**meta, "id": "hh"}, {**screening, "role": "harvest"}, "aa" * 32, {"rate_any": 0.0}, measured)
    assert harvest["segments_used"] == ""
    path = tmp_path / "sources.csv"
    write_sources(path, [row, harvest])
    loaded = read_sources(path)
    assert loaded[0]["segments_used"] == [30, 830, 1630, 2430, 3230] and loaded[1]["role"] == "harvest"
    assert loaded[0]["verdict_a_view"] == "pass: a_view"


def test_read_sources_rejects_unknown_role(tmp_path):
    path = tmp_path / "sources.csv"
    write_sources(path, [{"video_id": "aaa", "role": "maybe", "duration_s": "600"}])
    with pytest.raises(ValueError, match="role"):
        read_sources(path)


def test_select_segments_is_even_and_trims_both_ends():
    starts = select_segments(3321)
    assert len(starts) == 5
    assert starts[0] == 30
    assert starts[-1] + 60 <= 3321 - 30
    assert all(b > a for a, b in zip(starts, starts[1:]))
    gaps = {b - a for a, b in zip(starts, starts[1:])}
    assert max(gaps) - min(gaps) <= 1


def test_select_segments_exact_fit_and_too_short():
    assert select_segments(480) == [30, 120, 210, 300, 390]
    with pytest.raises(ValueError):
        select_segments(100)


@pytest.mark.parametrize("prediction,blind,expected", [
    ("P.131a", "P.131a", "correct"),
    ("P.130", "P.131a", "wrong_code"),
    ("P.130", "not_a_sign", "not_a_sign"),
    ("P.130", "uncertain", "uncertain"),
])
def test_verdict(prediction, blind, expected):
    assert verdict(prediction, blind) == expected


def test_tally_and_precision_exclude_uncertain():
    counts = tally(["correct"] * 6 + ["wrong_code"] * 2 + ["not_a_sign"] + ["uncertain"] * 5)
    assert counts == {"correct": 6, "wrong_code": 2, "not_a_sign": 1, "uncertain": 5, "n": 14}
    assert precision(counts) == pytest.approx(6 / 9)


def test_precision_is_none_without_decidable_labels():
    assert precision(tally(["uncertain", "uncertain"])) is None
    assert precision(tally([])) is None


def test_tally_rejects_unknown_verdict():
    with pytest.raises(ValueError):
        tally(["correct", "maybe"])


def test_bootstrap_precision_is_deterministic_and_brackets_point():
    rows = [("s1", "correct")] * 8 + [("s1", "wrong_code")] * 2 + [("s2", "correct")] * 5 + [("s2", "not_a_sign")] * 5
    first = bootstrap_precision(rows, n_boot=500, seed=7)
    second = bootstrap_precision(rows, n_boot=500, seed=7)
    assert first == second
    low, high = first
    assert 0 <= low <= 13 / 20 <= high <= 1


def test_bootstrap_precision_none_when_empty():
    assert bootstrap_precision([("s1", "uncertain")], n_boot=50, seed=1) == (None, None)


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def frame(timestamp, *signs):
    return {"timestamp": timestamp, "signs": [{"track_id": t, "bbox_xyxy": b, "label": label} for t, b, label in signs]}


def test_proxy_recall_matches_same_track_within_two_seconds():
    frames = [
        frame(10.0, (1, [100, 100, 140, 140], "unknown")),
        frame(11.0, (1, [102, 100, 142, 140], "unknown")),
        frame(11.8, (1, [104, 100, 144, 140], "P.131a")),
        frame(30.0, (2, [300, 200, 340, 240], "P.130")),
    ]
    annotations = [
        {"t": 10.0, "code": "P.131a", "bbox": [98, 98, 142, 142]},   # confirmed 1.8 s later: caught
        {"t": 30.0, "code": "P.131a", "bbox": [298, 198, 342, 242]},   # wrong code on track: missed
        {"t": 50.0, "code": "W.224", "bbox": [500, 100, 540, 140]},    # no frame: missed
    ]
    result = proxy_recall(annotations, frames, tolerance_s=2.0)
    assert result["annotated"] == 3 and result["caught"] == 1
    assert result["recall"] == pytest.approx(1 / 3)
    assert [a["caught"] for a in result["details"]] == [True, False, False]


def test_proxy_recall_ignores_confirmation_outside_window():
    frames = [frame(10.0, (1, [0, 0, 40, 40], "unknown")), frame(13.5, (1, [0, 0, 40, 40], "P.131a"))]
    result = proxy_recall([{"t": 10.0, "code": "P.131a", "bbox": [0, 0, 40, 40]}], frames, tolerance_s=2.0)
    assert result["caught"] == 0


def test_proxy_recall_empty_annotations():
    assert proxy_recall([], [], tolerance_s=2.0)["recall"] is None


def summary(correct, wrong, recall):
    return {"correct": correct, "wrong_code": wrong, "not_a_sign": 0, "uncertain": 0, "n": correct + wrong,
            "proxy_recall": recall}


def test_decide_t2_recommended_when_all_three_conditions_hold():
    table = {"A": summary(30, 10, 0.5), "B": summary(32, 4, 0.50), "C": summary(37, 5, 0.60)}
    result = decide_recommendations(table, min_labels=30)
    assert result["t2"]["decision"] == "recommend"


def test_decide_t2_rejected_when_precision_drops_more_than_two_points():
    table = {"A": summary(30, 10, 0.5), "B": summary(35, 1, 0.5), "C": summary(31, 6, 0.6)}
    result = decide_recommendations(table, min_labels=30)
    assert result["t2"]["decision"] == "not_recommended"
    assert not result["t2"]["checks"]["precision_within_0.02"]


def test_decide_insufficient_evidence_below_minimum_labels():
    table = {"A": summary(10, 2, 0.5), "B": summary(10, 1, 0.5), "C": summary(10, 1, 0.6)}
    assert decide_recommendations(table, min_labels=30)["t2"]["decision"] == "insufficient_evidence"


def test_decide_gate_needs_error_drop_and_small_correct_loss_and_val_f1():
    table = {"C": summary(100, 20, 0.6), "D": summary(96, 13, 0.6)}
    ok = decide_recommendations(table, min_labels=30, gate_pairs={"D": "C"}, val_macro_f1_drop={"D": 0.003})
    assert ok["gate"]["D"]["decision"] == "recommend"
    bad_f1 = decide_recommendations(table, min_labels=30, gate_pairs={"D": "C"}, val_macro_f1_drop={"D": 0.01})
    assert bad_f1["gate"]["D"]["decision"] == "not_recommended"
    too_many_lost = {"C": summary(100, 20, 0.6), "D": summary(90, 5, 0.6)}
    lost = decide_recommendations(too_many_lost, min_labels=30, gate_pairs={"D": "C"}, val_macro_f1_drop={"D": 0.0})
    assert lost["gate"]["D"]["decision"] == "not_recommended"


def test_summarize_review_grades_per_config_and_total():
    items = [{"number": 1, "config": "A", "prediction": "P.130", "segment": 0},
             {"number": 2, "config": "A", "prediction": "P.131a", "segment": 0},
             {"number": 3, "config": "B", "prediction": "P.131a", "segment": 240},
             {"number": 4, "config": "B", "prediction": "W.224", "segment": 240}]
    labels = {"1": "P.131a", "2": "P.131a", "3": "P.131a", "4": "uncertain"}
    summary, graded = summarize_review(items, labels, lambda g: g["segment"], n_boot=50, seed=1)
    assert summary["A"]["correct"] == 1 and summary["A"]["wrong_code"] == 1 and summary["A"]["precision"] == 0.5
    assert summary["B"]["uncertain"] == 1 and summary["B"]["precision"] == 1.0
    assert summary["ALL"]["n"] == 4 and summary["ALL"]["decidable"] == 3
    assert [g["verdict"] for g in graded] == ["wrong_code", "correct", "correct", "uncertain"]


def test_summarize_review_requires_every_label():
    with pytest.raises(ValueError, match="No blind label"):
        summarize_review([{"number": 9, "config": "A", "prediction": "P.130", "segment": 0}], {}, lambda g: 0)


def test_bootstrap_uses_python_random_only_through_seed():
    rows = [("s1", "correct"), ("s2", "wrong_code")] * 5
    state = random.getstate()
    bootstrap_precision(rows, n_boot=20, seed=3)
    assert random.getstate() == state
