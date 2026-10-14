"""Offline video max-quality mode only adds quality-gated hypotheses.

Run on stored MP4s; no live-device timing promise. These tests never assert
unverified sign-to-car legal applicability.
"""
from copy import deepcopy
import numpy as np
from src.perception.offline_sign_quality import safe_refinement,detect_offline_high_accuracy
from src.utils.types import SignDetection
from src.perception.temporal_signs import TemporalSigns


def sign(label,box=(15,15,65,65),confidence=.88,rec=.98):
    return SignDetection(label,confidence,box,
        recognition_confidence=rec,recognition_source='single_frame')


def test_weak_extra_cannot_degrade_primary():
    primary=[sign('P.130')]
    candidates=[sign('P.102',(100,100,140,140),.40,1.)]
    before=deepcopy(primary)
    output=safe_refinement(primary,candidates)
    assert output==before
    assert primary==before


def test_rescue_unknown_only_and_not_overwrite_known():
    known=sign('P.130')
    unknown=sign('unknown',(105,105,165,165),.4,0)
    extras=[sign('P.102',(12,12,63,65),.94,.99),
            sign('P.131a',(108,108,165,165),.93,.98)]
    out=safe_refinement([known,unknown],extras)
    assert [p.label for p in out]==['P.130','P.131a']
    assert out[1].recognition_source=='offline_highres_rescue'


def test_new_extra_tracks_are_not_automatically_legal_truth():
    low=[sign('unknown',(20,20,90,90),.7,0)]
    x=sign('P.130',(130,120,190,190),.92,.99)
    proposed=safe_refinement(low,[x])
    assert proposed[-1].recognition_source=='offline_highres_new'
    assert proposed[-1].label=='P.130'
    assert not proposed[-1].label_verified
    assert not proposed[-1].context_verified
    assert proposed[-1].applicability=='uncertain'


def test_temporal_video_requires_two_samples_for_rescued_code():
    trace=TemporalSigns()
    image=np.random.default_rng(12).integers(0,255,(300,400,3),dtype=np.uint8)
    hi=sign('P.130',(140,90,200,150),.94,.99)
    first=trace.update(image,[deepcopy(hi)],timestamp=.0)
    assert first[0].label=='unknown'
    second=trace.update(image,[deepcopy(hi)],timestamp=.2)
    assert second[0].label=='P.130'
    assert second[0].recognition_source=='temporal_best_crop'
    none=trace.update(image,[],timestamp=.4)
    assert none==[]


class FakeDetector:
    def __init__(self):self.image_size=640;self.calls=[]
    def detect(self,img):
        self.calls.append(self.image_size)
        return [sign('traffic_sign',confidence=.95,rec=.0)]


class FakeRecognizer:
    def recognize(self,img,signs,*,min_short_side):
        for s in signs:
            s.label='P.130'
            s.recognition_confidence=.98
            s.recognition_source='single_frame'
        return signs


def test_three_passes_restore_original_detection_size():
    d=FakeDetector()
    out=detect_offline_high_accuracy(d,FakeRecognizer(),
        np.zeros((400,800,3),dtype=np.uint8))
    assert d.calls==[960,640,1280]
    assert d.image_size==640
    assert len(out)==1


def test_invalid_unconfirmed_code_stays_unknown():
    uncertain=sign('unknown',confidence=.4,rec=.0)
    high=sign('P.130',(130,100,200,170),confidence=.9,rec=.89)
    out=safe_refinement([uncertain],[high])
    assert [s.label for s in out]==['unknown']
