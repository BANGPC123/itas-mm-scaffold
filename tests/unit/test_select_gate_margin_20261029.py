import numpy as np
import pytest
import torch

from scripts.select_gate_margin_20261029 import choose_margin, gate_scores, operating_point, rejected, sweep
from src.perception.sign_open_set import FrozenReferenceGate


def unit(rng, n, d=16):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def make_bank(tmp_path, rng):
    positive = unit(rng, 12)
    labels = np.array(["A"] * 6 + ["B"] * 6)
    negative = unit(rng, 5)
    path = tmp_path / "bank.npz"
    np.savez(path, positive=positive, labels=labels, negative=negative,
             negative_targets=np.array(["*"] * 4 + ["A"]), negative_kinds=np.array(["irrelevant"] * 5))
    return path, dict(np.load(path))


def test_vectorised_scores_match_the_runtime_gate(tmp_path):
    rng = np.random.default_rng(3)
    path, bank = make_bank(tmp_path, rng)
    queries = unit(rng, 40)
    labels = np.array(rng.choice(["A", "B", "unknown", "C"], size=40))
    neg, pos = gate_scores(queries, labels, bank)
    gate = FrozenReferenceGate(path, "cpu", margin=0.05, min_similarity=0.0)
    runtime = gate.verdict(torch.from_numpy(queries), list(labels))
    ours = rejected(neg, pos, 0.05, min_similarity=0.0)
    assert [r is not None for r in runtime] == list(ours)
    assert np.isnan(neg[labels == "unknown"]).all() and np.isnan(neg[labels == "C"]).all()


def test_gate_abstains_below_min_similarity():
    neg, pos = np.array([0.79, 0.95]), np.array([0.1, 0.1])
    assert list(rejected(neg, pos, 0.015)) == [False, True]


def val_fixture():
    # 3 classes: 0 and 1 main, 2 background(unknown). Sample 3 is a class-1 crop the gate would wrongly veto.
    logits = np.array([[5, 0, 0], [0, 5, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]], dtype=np.float64)
    y = np.array([0, 1, 0, 1, 2])
    neg = np.array([0.1, 0.1, 0.1, 0.95, np.nan])
    pos = np.array([0.5, 0.5, 0.5, 0.2, np.nan])
    return logits, y, neg, pos


def test_operating_point_counts_vetoed_crops_as_missed():
    logits, y, neg, pos = val_fixture()
    free = operating_point(logits, y, neg, pos, None, 0.5, 2, [0, 1])
    gated = operating_point(logits, y, neg, pos, 0.015, 0.5, 2, [0, 1])
    assert free["macro_f1_main"] == pytest.approx(1.0) and free["coverage_known"] == 1.0
    assert gated["coverage_known"] == 0.75 and gated["macro_f1_main"] < free["macro_f1_main"]


def test_sweep_and_choose_margin_prefers_highest_block_rate_within_f1_budget():
    logits, y, neg, pos = val_fixture()
    held = {"neg": np.array([0.9, 0.9, 0.6]), "pos": np.array([0.2, 0.89, 0.2]),
            "pred": np.array([0, 1, 0]), "conf": np.array([0.9, 0.9, 0.9])}
    base, rows = sweep({"logits": logits, "y": y, "neg": neg, "pos": pos}, held, {"T": 0.5}, 2, [0, 1], [0.0, 0.1, 0.8])
    by_margin = {r["margin"]: r for r in rows}
    assert by_margin[0.0]["block_rate_heldout"] == pytest.approx(2 / 3)   # 0.9 > 0.2 and 0.9 > 0.89 + 0 ; third is under 0.80
    assert by_margin[0.1]["block_rate_heldout"] == pytest.approx(1 / 3)
    assert by_margin[0.8]["block_rate_heldout"] == 0.0
    # the wrongly vetoed class-1 crop costs far more than 0.005 F1 until the margin exceeds 0.75
    assert choose_margin(rows)["margin"] == 0.8


def test_choose_margin_returns_none_when_no_margin_fits():
    rows = [{"margin": 0.0, "points": {"T": {"f1_drop": 0.02}}, "block_rate_heldout": 0.9}]
    assert choose_margin(rows) is None
