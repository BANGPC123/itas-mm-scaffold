import pytest

from scripts.run_field_eval_20261029 import build_jobs, configs

SOURCES = [
    {"video_id": "h1", "role": "harvest", "segments_used": [], "sha256": "00"},
    {"video_id": "e1", "role": "eval", "segments_used": [30, 830], "sha256": "11"},
    {"video_id": "e2", "role": "eval", "segments_used": [30], "sha256": "22"},
    {"video_id": "x1", "role": "rejected", "segments_used": [30], "sha256": "33"},
]


def test_configs_without_gate_v2_have_only_a_b_c():
    table = configs(0.64)
    assert sorted(table) == ["A", "B", "C"]
    assert table["A"]["gate"] and table["A"]["threshold"] == 0.63
    assert table["B"]["gate"] is None and table["B"]["threshold"] == 0.87
    assert table["C"]["threshold"] == 0.64 and table["C"]["gate"] is None


def test_configs_with_gate_v2_add_d_and_e_with_matching_thresholds():
    table = configs(0.64, "models/sign_recognizer/reference_gate_20261029_v2.npz", 0.24)
    assert table["D"]["margin"] == table["E"]["margin"] == 0.24
    assert table["D"]["threshold"] == 0.64 and table["E"]["threshold"] == 0.87
    assert table["D"]["gate"] == table["E"]["gate"] == "models/sign_recognizer/reference_gate_20261029_v2.npz"
    assert table["D"]["model"] == table["B"]["model"]


def test_jobs_are_segment_major_and_only_for_eval_videos():
    jobs = build_jobs(SOURCES, ["A", "B", "C"], configs(0.64))
    assert [j["name"] for j in jobs] == [
        "A_e1_30", "B_e1_30", "C_e1_30", "A_e1_830", "B_e1_830", "C_e1_830", "A_e2_30", "B_e2_30", "C_e2_30"]
    assert jobs[0]["sha256"] == "11" and jobs[-1]["sha256"] == "22"


def test_unavailable_configuration_is_an_error():
    with pytest.raises(ValueError, match="D"):
        build_jobs(SOURCES, ["A", "D"], configs(0.64))
