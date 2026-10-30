import numpy as np
from src.perception.temporal_signs import TemporalSigns
from src.utils.types import SignDetection

IMAGE = np.random.default_rng(3).integers(0, 256, (200, 400, 3), dtype=np.uint8)


def sign(box, label='unknown'):
    return SignDetection(label, .9, box, recognition_confidence=.95 if label != 'unknown' else None)


def test_sign_sliding_60px_keeps_track_when_iou_is_low():
    tracker = TemporalSigns()
    first = tracker.update(IMAGE, [sign((10, 10, 90, 90))], 0)[0]
    second = tracker.update(IMAGE, [sign((70, 10, 150, 90))], .2)[0]
    assert second.track_id == first.track_id


def test_misread_label_still_keeps_track():
    tracker = TemporalSigns()
    first = tracker.update(IMAGE, [sign((10, 10, 90, 90), 'P.130')], 0)[0]
    second = tracker.update(IMAGE, [sign((70, 10, 150, 90), 'P.131a')], .2)[0]
    assert second.track_id == first.track_id


def test_three_close_signs_moving_together_get_three_tracks():
    tracker = TemporalSigns()
    boxes = [(0, 0, 60, 60), (70, 0, 130, 60), (140, 0, 200, 60)]
    first = tracker.update(IMAGE, [sign(b) for b in boxes], 0)
    moved = [(b[0] + 50, b[1], b[2] + 50, b[3]) for b in boxes]
    second = tracker.update(IMAGE, [sign(b) for b in moved], .2)
    assert len({s.track_id for s in first}) == 3
    assert len({s.track_id for s in second}) == 3


def test_one_track_never_takes_two_boxes():
    tracker = TemporalSigns()
    first = tracker.update(IMAGE, [sign((100, 50, 160, 110))], 0)[0]
    second = tracker.update(IMAGE, [sign((120, 50, 180, 110)), sign((150, 50, 210, 110))], .2)
    ids = [s.track_id for s in second]
    assert len(set(ids)) == 2
    assert first.track_id in ids


def test_backward_seek_resets_tracks():
    tracker = TemporalSigns()
    first = tracker.update(IMAGE, [sign((10, 10, 90, 90))], 1)[0]
    after_seek = tracker.update(IMAGE, [sign((10, 10, 90, 90))], .5)[0]
    assert after_seek.track_id != first.track_id
