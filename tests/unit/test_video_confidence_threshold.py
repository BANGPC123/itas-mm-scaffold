import math

import numpy as np
import torch
from src.perception.sign_recognizer import SignRecognizer
from src.utils.types import SignDetection

# Softmax top-1 probability of exactly 0.70 for the second class.
LOGITS = [0., math.log(7 / 3)]


def recognizer(checkpoint_threshold=.87):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([LOGITS] * len(x))
    rec = SignRecognizer.__new__(SignRecognizer)
    rec.model = Model()
    rec.config = {'min_short_side': 16}
    rec.device = 'cpu'
    rec.size = 128
    rec.threshold = checkpoint_threshold
    rec.classes = ['P.130', 'P.131a']
    return rec


def crop_sign():
    return [SignDetection('traffic_sign', .9, (0, 0, 40, 40))]


def test_video_threshold_below_score_accepts_label():
    rec = recognizer()
    sign, = rec.recognize(np.zeros((60, 60, 3), np.uint8), crop_sign(), threshold=.64)
    assert sign.label == 'P.131a'
    assert round(sign.raw_score, 2) == .70


def test_still_image_path_keeps_checkpoint_threshold():
    rec = recognizer()
    sign, = rec.recognize(np.zeros((60, 60, 3), np.uint8), crop_sign())
    assert sign.label == 'unknown'
    assert sign.raw_label == 'P.131a'


def test_checkpoint_threshold_is_not_changed_by_video_override():
    rec = recognizer()
    rec.recognize(np.zeros((60, 60, 3), np.uint8), crop_sign(), threshold=.64)
    assert rec.threshold == .87
