"""Regression: offline video consumes multiple independent frames, not a camera.

Secondary YOLO cues must not assert legal applicability. UFLD visual proposals
must be current-frame only and never create legal ego-lane geometry.
"""
import numpy as np
from src.perception.temporal_signs import TemporalSigns,StreamEvidence
from src.perception.lane_temporal import OfflineLaneEvidence
from src.utils.types import SignDetection,LaneInfo


def _frame(seed=123):
    return np.random.default_rng(seed).integers(30,200,(360,640,3),dtype=np.uint8)


def _sign(code='P.130'):
    return SignDetection(code,.95,(120,100,185,175),
        recognition_confidence=.98,recognition_source='single_frame')


def test_offline_requires_three_detections_no_early_sign():
    t=TemporalSigns(min_votes=3,allow_one_miss=True)
    f=_frame()
    assert t.update(f,[_sign()],0)[0].label=='unknown'
    assert t.update(f,[_sign()],.16)[0].label=='unknown'
    assert t.update(f,[_sign()],.32)[0].label=='P.130'


def test_gap_retains_evidence_without_ghost_output():
    t=TemporalSigns(min_votes=3,allow_one_miss=True)
    f=_frame()
    assert t.update(f,[_sign()],0)[0].label=='unknown'
    assert t.update(f,[],.16)==[]
    assert t.update(f,[_sign()],.32)[0].label=='unknown'
    assert t.update(f,[_sign()],.48)[0].label=='P.130'


def test_offline_scene_cut_invalidates_old_evidence():
    t=TemporalSigns(min_votes=3,allow_one_miss=True)
    first=_frame(4);cut=np.clip(first.astype(np.int16)+130,0,255).astype(np.uint8)
    t.update(first,[_sign()],0)
    t.update(first,[_sign()],.16)
    assert t.update(cut,[_sign()],.32)[0].label=='unknown'
    assert t.update(cut,[_sign()],.48)[0].label=='unknown'
    assert t.update(cut,[_sign()],.64)[0].label=='P.130'


def test_offline_tracks_isolated_from_live():
    online=StreamEvidence().get('road',_frame().shape,.7)
    offline=StreamEvidence(min_votes=3,allow_one_miss=True).get('road',_frame().shape,.7)
    assert online.min_votes==2 and offline.min_votes==3
    assert not online.allow_one_miss and offline.allow_one_miss


def test_lane_offline_proposals_need_temporal_evidence_and_no_ghost():
    f=_frame()
    t=OfflineLaneEvidence(min_hits=2)
    p=[[(250+x,50+y) for x,y in ((0,0),(5,40),(10,80),(18,120),(22,180))]]
    assert t.update(p,0,f.shape,image=f)==[]
    assert t.update(p,.16,f.shape,image=f)==p
    assert t.update([],.32,f.shape,image=f)==[]
    assert t.update(p,.48,f.shape,image=f)==[]


def test_lane_temporal_cut_and_seek_clear_tracks():
    t=OfflineLaneEvidence()
    line=[[(100,100),(120,150),(130,200)]]
    f=_frame(5)
    assert t.update(line,0,f.shape,f)==[]
    assert t.update(line,.15,f.shape,f)==line
    abrupt=np.clip(f.astype(np.int16)+130,0,255).astype(np.uint8)
    assert t.update(line,.30,f.shape,abrupt)==[]
    assert t.update(line,.10,f.shape,abrupt)==[]


def test_lane_different_curve_does_not_confirm():
    t=OfflineLaneEvidence()
    f=_frame()
    a=[[(100,100),(110,150),(120,200)]]
    b=[[(450,100),(470,150),(490,200)]]
    assert t.update(a,0,f.shape,f)==[]
    assert t.update(b,.16,f.shape,f)==[]


def test_candidate_fields_do_not_modify_verified_ego_geometry():
    lane=LaneInfo([],polylines=[],ego_boundaries=[],backend='camera_road_paint_visual')
    lane.candidate_polylines=[[(20,30),(25,60)]]
    assert lane.ego_boundaries==[]
    assert lane.polylines==[]


def test_background_rejection_clears_offline_sign_memory():
    t=TemporalSigns(min_votes=3,allow_one_miss=True)
    frame=_frame()
    t.update(frame,[_sign()],0)
    t.update(frame,[_sign()],.16)
    rejected=_sign('unknown')
    rejected.recognition_source='background_rejected'
    assert t.update(frame,[rejected],.32)==[]
    assert t.update(frame,[_sign()],.48)[0].label=='unknown'
    assert t.update(frame,[_sign()],.64)[0].label=='unknown'
    assert t.update(frame,[_sign()],.80)[0].label=='P.130'


def test_road_paint_veto_cannot_be_overridden_by_temporal_ufld_proposals():
    from src.perception.lane_temporal import offline_visual_fallback_allowed
    rejected=LaneInfo([],polylines=[],backend='camera_road_paint_visual',
                      camera_view='forward',rejection_reason='no_supported_road_paint')
    assert offline_visual_fallback_allowed(rejected) is False
    clean_but_unknown=LaneInfo([],polylines=[],backend='ufld_v2',camera_view='unknown')
    assert offline_visual_fallback_allowed(clean_but_unknown) is False
    forward=LaneInfo([],polylines=[],backend='ufld_v2',camera_view='forward')
    assert offline_visual_fallback_allowed(forward) is True
