from src.context.sign_applicability import SignApplicability
from src.utils.types import LaneInfo, SignDetection
from src.perception.lane_consistency import reject_obvious_artifacts, credible_ego


def sign(**kwargs):
    return SignDetection('P.130', .98, (250, 70, 290, 120), panel_role='primary', **kwargs)


def test_verified_roadside_scope_requires_review():
    model=SignApplicability()
    lane=LaneInfo([])
    raw=model.apply([sign(placement='roadside',lane_scope='all')],lane,(600,800,3))[0]
    assert raw.applicability=='uncertain'
    checked=model.apply([sign(placement='roadside',lane_scope='all',context_verified=True,label_verified=True)],lane,(600,800,3))[0]
    assert checked.applicability=='applies'


def test_verified_other_lane_ignored():
    result=SignApplicability().apply([sign(placement='overhead',lane_scope='other',context_verified=True,label_verified=True)],LaneInfo([]),(600,800,3))[0]
    assert result.applicability=='ignored'


def test_unknown_roadside_does_not_issue_instruction():
    result=SignApplicability().apply([sign(placement='roadside',context_verified=True)],LaneInfo([]),(600,800,3))[0]
    assert result.applicability=='uncertain'


def test_vehicle_restriction_precedes_lane_scope():
    result=SignApplicability().apply([sign(placement='roadside',lane_scope='all',context_verified=True,label_verified=True,allowed_vehicles=['truck'])],LaneInfo([]),(600,800,3),'car')[0]
    assert result.applicability=='ignored'


def test_lane_geometry_crossing_pair_rejected():
    h,w=600,800
    a=[(200,400),(310,470),(450,599)]
    b=[(500,400),(390,470),(250,599)]
    lane=LaneInfo([],polylines=[a,b],ego_boundaries=[a,b],lane_center_offset_px=20,backend='ufld_v2')
    assert not credible_ego([a,b],(h,w,3))
    filtered=reject_obvious_artifacts(lane,(h,w,3))
    assert filtered.ego_boundaries==[] and filtered.lane_center_offset_px is None


def test_lane_geometry_keeps_smooth_curves():
    a=[(290,380),(260,440),(210,520),(160,599)]
    b=[(510,380),(540,440),(590,520),(640,599)]
    lane=LaneInfo([],polylines=[a,b],ego_boundaries=[a,b],backend='ufld_v2')
    filtered=reject_obvious_artifacts(lane,(600,800,3))
    assert len(filtered.polylines)==2 and len(filtered.ego_boundaries)==2


def test_lane_geometry_rejects_short_dashboard_fragment():
    fragment=[(100,540),(120,550),(140,560)]
    lane=LaneInfo([],polylines=[fragment],backend='ufld_v2')
    assert reject_obvious_artifacts(lane,(600,800,3)).polylines==[]



def test_unverified_overhead_sign_cannot_become_roadside_applicable():
    # Image geometry never establishes roadside sign jurisdiction.
    sign_obj=sign(placement='roadside',lane_scope='all',context_verified=True,label_verified=False)
    output=SignApplicability().apply([sign_obj],LaneInfo([]),(600,800,3))
    assert output[0].applicability=='uncertain'


def test_partial_supplement_review_cannot_invent_vehicle_applicability():
    main=sign()
    supplement=SignDetection('unknown',.8,(255,122,285,140),panel_role='supplementary')
    grouped=SignApplicability().apply([main,supplement],LaneInfo([]),(600,800,3))
    assert grouped[0].is_compound
    assert grouped[0].applicability=='uncertain'
