import numpy as np
from src.perception.temporal_signs import TemporalSigns,StreamEvidence
from src.utils.types import SignDetection

IMAGE=np.random.default_rng(1).integers(0,256,(100,100,3),dtype=np.uint8)
def sign(label='P.130',box=(10,10,50,50)):
    return SignDetection(label,.9,box,recognition_confidence=.95 if label!='unknown' else None)

def test_two_independent_observations_then_short_blur_gap():
    tracker=TemporalSigns()
    assert tracker.update(IMAGE,[sign()],0)[0].label=='unknown'
    result=tracker.update(IMAGE,[sign()],.1)[0]
    assert result.label=='P.130' and result.confirmation_delay_ms==100
    result=tracker.update(IMAGE,[sign('unknown')],.2)[0]
    assert result.label=='P.130' and round(result.crop_age_ms)==100
    assert tracker.update(IMAGE,[sign('unknown')],1)[0].label=='unknown'

def test_conflict_missing_seek_and_blank_reset_evidence():
    for action in ('conflict','missing','seek','blank'):
        t=TemporalSigns();t.update(IMAGE,[sign()],1);t.update(IMAGE,[sign()],1.1)
        if action=='conflict':assert t.update(IMAGE,[sign('P.131a')],1.2)[0].label=='unknown'
        if action=='missing':
            assert t.update(IMAGE,[],1.2)==[]
            assert t.update(IMAGE,[sign()],1.3)[0].label=='unknown'
        if action=='seek':assert t.update(IMAGE,[sign()],.5)[0].label=='unknown'
        if action=='blank':assert t.update(np.full_like(IMAGE,255),[sign()],1.2)[0].label=='unknown'

def test_ambiguous_matching_does_not_copy_one_label_to_two_signs():
    t=TemporalSigns();t.update(IMAGE,[sign()],0);t.update(IMAGE,[sign()],.1)
    assert all(s.label=='unknown' for s in t.update(IMAGE,[sign('unknown'),sign('unknown')],.2))


def test_background_rejection_cannot_inherit_confirmed_label():
    tracker=TemporalSigns();tracker.update(IMAGE,[sign()],0);tracker.update(IMAGE,[sign()],.1)
    rejected=sign('unknown');rejected.recognition_source='background_rejected'
    assert tracker.update(IMAGE,[rejected],.2)==[]
    assert tracker.update(IMAGE,[sign()],.3)[0].label=='unknown'


def test_physical_symbol_contradiction_clears_previous_confirmed_label():
    tracker=TemporalSigns();tracker.update(IMAGE,[sign('I.409')],0);tracker.update(IMAGE,[sign('I.409')],.1)
    rejected=sign('unknown');rejected.recognition_source='symbol_shape_unsupported'
    assert tracker.update(IMAGE,[rejected],.2)[0].label=='unknown'
    assert tracker.update(IMAGE,[sign('I.409')],.3)[0].label=='unknown'

def test_streams_and_roi_are_isolated_and_capacity_bounded():
    pool=StreamEvidence();a=pool.get('a',IMAGE.shape,.65)
    a.update(IMAGE,[sign()],0);a.update(IMAGE,[sign()],.1)
    assert pool.get('b',IMAGE.shape,.65).tracks==[]
    assert pool.get('a',IMAGE.shape,1).tracks==[]
    for i in range(20):pool.get(str(i),IMAGE.shape,1)
    assert len(pool.sessions)==8


def test_continuing_correct_observations_do_not_expire_with_best_crop(monkeypatch):
    qualities=iter([100,90,80,70,60,50,40,30,20,10])
    monkeypatch.setattr('src.perception.temporal_signs.crop_quality',lambda crop:next(qualities))
    tracker=TemporalSigns()
    for i in range(10):
        result=tracker.update(IMAGE,[sign()],i*.1)[0]
        if i:assert result.label=='P.130'


def test_small_moving_sign_can_confirm_with_same_semantics():
    tracker=TemporalSigns()
    tracker.update(IMAGE,[sign(box=(10,10,30,30))],0)
    result=tracker.update(IMAGE,[sign(box=(22,14,42,34))],.2)[0]
    assert result.label=='P.130'


def test_motion_match_requires_same_known_label_and_unique_neighbor():
    for label in ['unknown','P.131a']:
        tracker=TemporalSigns();tracker.update(IMAGE,[sign(box=(10,10,30,30))],0)
        result=tracker.update(IMAGE,[sign(label,box=(22,14,42,34))],.2)[0]
        assert result.label=='unknown'
