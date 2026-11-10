import numpy as np
import torch
from src.perception.sign_recognizer import SignRecognizer
from src.perception.temporal_signs import TemporalSigns
from src.utils.types import SignDetection

IMAGE = np.random.default_rng(2).integers(0, 256, (100, 100, 3), dtype=np.uint8)


def recognizer_with_logits(logits, classes):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([logits] * len(x))
    recognizer = SignRecognizer.__new__(SignRecognizer)
    recognizer.model = Model()
    recognizer.config = {'min_short_side': 16}
    recognizer.device = 'cpu'
    recognizer.size = 128
    recognizer.threshold = .7
    recognizer.classes = classes
    return recognizer


def test_below_threshold_keeps_top1_as_raw_fields():
    recognizer = recognizer_with_logits([0., .4], ['P.130', 'P.131a'])
    sign, = recognizer.recognize(np.zeros((60, 60, 3), np.uint8), [SignDetection('traffic_sign', .9, (0, 0, 40, 40))])
    assert sign.label == 'unknown'
    assert sign.raw_label == 'P.131a'
    assert 0.55 < sign.raw_score < 0.65


def test_crop_too_small_has_no_raw_fields():
    recognizer = recognizer_with_logits([0., 5.], ['P.130', 'P.131a'])
    sign, = recognizer.recognize(np.zeros((60, 60, 3), np.uint8), [SignDetection('traffic_sign', .9, (0, 0, 8, 8))])
    assert sign.recognition_source == 'crop_too_small'
    assert sign.raw_label is None and sign.raw_score is None


def test_temporal_confirmation_does_not_clear_raw_fields():
    def sign(label, raw_label, raw_score, box=(10, 10, 50, 50)):
        return SignDetection(label, .9, box,
                             recognition_confidence=.95 if label != 'unknown' else None,
                             raw_label=raw_label, raw_score=raw_score)
    tracker = TemporalSigns()
    tracker.update(IMAGE, [sign('P.130', 'P.130', .95)], 0)
    confirmed = tracker.update(IMAGE, [sign('P.130', 'P.130', .95)], .1)[0]
    assert confirmed.label == 'P.130'
    assert confirmed.raw_label == 'P.130' and confirmed.raw_score == .95
    blurred = tracker.update(IMAGE, [sign('unknown', 'P.131a', .6)], .2)[0]
    assert blurred.label == 'P.130'
    assert blurred.raw_label == 'P.131a' and blurred.raw_score == .6
