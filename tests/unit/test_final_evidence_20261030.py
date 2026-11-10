import json
from collections import Counter

import pytest

from scripts.final_evidence_20261030 import (
    agreement, blind_order, normalize, precision_summary, stratified_sample,
    public_blind_items, uniform_frame_times, verify_seal,
)


def test_blind_order_is_reproducible_without_exposing_source_fields():
    rows = [{"source": "a", "original_key": str(i), "prediction": "P.127", "short_side_px": 20} for i in range(30)]
    ordered = blind_order(rows)
    assert ordered == blind_order(rows)
    assert [r["original_key"] for r in ordered] != [r["original_key"] for r in rows]
    assert len({r["item_id"] for r in ordered}) == 30
    assert all(r["item_id"] == f"{i:04d}" for i, r in enumerate(ordered, 1))
    public = public_blind_items(ordered)
    assert all(set(r) == {"item_id", "short_side_px"} for r in public)
    assert "P.127" not in json.dumps(public)


def test_stratified_sample_respects_cap_and_does_not_use_test():
    rows = [{"canonical_box_id": f"{c}_{s}_{i}", "code": c, "split": s}
            for c in ["P.127", "P.130", "P.131a"]
            for s in ["TRAIN", "VAL", "TEST"] for i in range(12)]
    picked = stratified_sample(rows, total=24, per_code=10)
    assert len(picked) == 24
    assert all(r["split"] != "TEST" for r in picked)
    assert max(Counter(r["code"] for r in picked).values()) <= 10
    assert set(r["split"] for r in picked) == {"TRAIN", "VAL"}
    assert picked == stratified_sample(rows, total=24, per_code=10)
    with pytest.raises(ValueError):
        stratified_sample(rows, total=31, per_code=10)


def test_kappa_accounts_for_marginal_frequencies_and_undefined_case():
    result = agreement(["P.127", "P.127", "P.130", "P.130"],
                       ["P.127", "P.130", "P.130", "P.130"])
    assert result["agreement"] == .75
    assert result["kappa"] == .5
    assert agreement(["P.127"], ["P.127"])["kappa"] is None
    assert agreement([], [])["agreement"] is None
    with pytest.raises(ValueError):
        agreement(["P.127"], [])


def test_precision_excludes_uncertain_and_bootstraps_whole_segments():
    rows = [{"cluster": "v@0", "prediction": "P.127", "label": "P.127"},
            {"cluster": "v@0", "prediction": "P.127", "label": "not_a_sign"},
            {"cluster": "v@60", "prediction": "P.127", "label": "uncertain"}]
    result = precision_summary(rows, draws=100)
    assert result["n"] == 3 and result["decidable"] == 2
    assert result["precision"] == .5
    assert result["ci95"] == [.5, .5]
    assert result["clusters"] == 2
    assert normalize("not_qcvn_sign") == "not_a_sign"
    assert normalize("unreadable") == "uncertain"


def test_seal_prevents_unlock_on_missing_duplicate_or_changed_labels(tmp_path):
    import hashlib
    path = tmp_path / "labels.jsonl"
    rows = [{"item_id": "0001", "label_sol": "P.127", "confidence": .95,
             "description": "Red speed ring with black numeral."}]
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_seal(path, digest, ["0001"]) == rows
    with pytest.raises(ValueError):
        verify_seal(path, digest, ["0001", "0002"])
    path.write_text(path.read_text() + path.read_text(), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_seal(path, digest, ["0001"])


def test_uniform_frames_cover_video_without_endpoint_decode_risk():
    times = uniform_frame_times(60, 15)
    assert len(times) == 15 and times[0] == 2 and times[-1] == 58
    assert all(b - a == 4 for a, b in zip(times, times[1:]))
    with pytest.raises(ValueError):
        uniform_frame_times(0, 15)
