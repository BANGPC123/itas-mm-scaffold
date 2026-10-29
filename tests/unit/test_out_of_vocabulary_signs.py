import numpy as np
import torch
from src.perception.sign_recognizer import SignRecognizer
from src.perception.temporal_signs import TemporalSigns
from src.utils.types import SignDetection

IMAGE = np.random.default_rng(3).integers(0, 256, (120, 120, 3), dtype=np.uint8)


class Background(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([[0., 10.]]).repeat(len(x), 1)


def recognizer(config):
    r = SignRecognizer.__new__(SignRecognizer)
    r.model = Background(); r.config = config
    r.device = 'cpu'; r.size = 128; r.threshold = .7
    r.classes = ['P.130', 'unknown']
    return r


def test_background_rejection_stays_default_when_not_configured():
    result = recognizer({'min_short_side': 16}).recognize(IMAGE, [SignDetection('traffic_sign', .9, (0, 0, 40, 40))])
    assert result[0].recognition_source == 'background_rejected'


def test_confident_detection_outside_vocabulary_is_kept_as_unknown():
    config = {'min_short_side': 16, 'keep_background_detector_confidence': .5, 'keep_background_min_side': 16}
    signs = [SignDetection('traffic_sign', .78, (0, 0, 40, 40)),
             SignDetection('traffic_sign', .3, (0, 0, 40, 40)),
             SignDetection('traffic_sign', .9, (0, 0, 40, 12))]
    result = recognizer(config).recognize(IMAGE, signs, min_short_side=12)
    assert [s.label for s in result] == ['unknown', 'unknown', 'unknown']
    assert [s.recognition_source for s in result] == [
        'out_of_vocabulary_candidate', 'background_rejected', 'background_rejected']


def test_kept_candidate_survives_temporal_filter_without_a_code():
    tracker = TemporalSigns()
    kept = SignDetection('unknown', .78, (10, 10, 50, 50), recognition_confidence=.99,
                         recognition_source='out_of_vocabulary_candidate')
    rejected = SignDetection('unknown', .78, (60, 60, 100, 100), recognition_confidence=.99,
                             recognition_source='background_rejected')
    result = tracker.update(IMAGE, [kept, rejected], 0)
    assert len(result) == 1 and result[0].label == 'unknown'
    assert result[0].bbox_xyxy == (10, 10, 50, 50)
