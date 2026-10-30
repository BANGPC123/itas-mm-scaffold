import numpy as np
from src.perception.temporal_signs import TemporalSigns, leading_label
from src.utils.types import SignDetection

IMAGE = np.random.default_rng(4).integers(0, 256, (200, 200, 3), dtype=np.uint8)
BOX = (50, 50, 110, 110)


def sign(label):
    return SignDetection(label, .9, BOX, recognition_confidence=.95 if label != 'unknown' else None)


def run_sequence(labels, min_votes):
    tracker = TemporalSigns(min_votes=min_votes)
    results = []
    for index, label in enumerate(labels):
        results.append(tracker.update(IMAGE, [sign(label)], .1 * index)[0].label)
    return results


def test_one_misread_does_not_cancel_majority():
    assert run_sequence(['P.131a', 'P.131a', 'W.245a', 'P.131a'], min_votes=3)[-1] == 'P.131a'


def test_alternating_labels_never_confirm():
    assert run_sequence(['A', 'B', 'A', 'B'], min_votes=3) == ['unknown'] * 4


def test_unknown_samples_cast_no_ballot():
    assert run_sequence(['P.131a', 'unknown', 'P.131a', 'unknown', 'P.131a'], min_votes=3)[-1] == 'P.131a'


def test_leader_needs_one_more_ballot_than_runner_up():
    assert leading_label([(0, 'A'), (0, 'A'), (0, 'B'), (0, 'B')], 2) is None
    assert leading_label([(0, 'A'), (0, 'A'), (0, 'A'), (0, 'B'), (0, 'B')], 3) == 'A'
    assert leading_label([(0, 'A')], 2) is None
