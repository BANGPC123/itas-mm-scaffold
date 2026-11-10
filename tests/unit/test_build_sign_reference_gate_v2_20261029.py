import pandas as pd

from scripts.build_sign_reference_gate import fit_negative_records, select_positive_rows


def boxes():
    rows = []
    for index in range(5):
        rows.append(("box:a%d" % index, "TRAIN", "P.130", 40.0))
    rows += [("box:b0", "TRAIN", "P.131a", 15.0), ("box:b1", "TRAIN", "P.131a", 30.0), ("box:v0", "VAL", "P.130", 40.0),
             ("box:t0", "TEST", "P.130", 40.0), ("box:u0", "TRAIN", "unknown", 40.0), ("box:w0", "TRAIN", "W.999", 40.0)]
    return pd.DataFrame(rows, columns=["canonical_box_id", "split", "code", "short_side_px"])


def test_positives_are_train_only_trained_codes_at_least_16_px_and_capped():
    selected = select_positive_rows(boxes(), ["P.130", "P.131a", "unknown"], per_class=3, min_side=16)
    assert set(selected.split) == {"TRAIN"}
    assert selected.code.value_counts().to_dict() == {"P.130": 3, "P.131a": 1}      # cap, 15 px box dropped
    assert "unknown" not in set(selected.code) and "W.999" not in set(selected.code)
    assert list(selected[selected.code == "P.130"].canonical_box_id) == ["box:a0", "box:a1", "box:a2"]


def test_only_fit_split_not_a_sign_negatives_enter_the_bank():
    records = [{"id": 1, "label": "not_a_sign", "split": "fit"}, {"id": 2, "label": "not_a_sign", "split": "heldout"},
               {"id": 3, "label": "qcvn_sign:P.130", "split": "fit"}, {"id": 4, "label": "uncertain", "split": "fit"}]
    assert [r["id"] for r in fit_negative_records(records)] == [1]
