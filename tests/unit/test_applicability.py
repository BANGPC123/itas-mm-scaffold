from src.context.sign_applicability import SignApplicability
from src.utils.types import SignDetection,LaneInfo

def sign(label,box,**kwargs):return SignDetection(label,.9,box,**kwargs)

def test_group_before_filtering_truck_supplement_limits_parking_sign():
    signs=[sign('P.130',(100,100,200,200),panel_role='primary'),
           sign('truck supplement',(115,205,185,240),panel_role='supplementary',allowed_vehicles=['truck'])]
    result=SignApplicability().apply(signs,LaneInfo([]),(600,800,3),'car')
    assert len(result)==1 and result[0].is_compound
    assert result[0].applicability=='uncertain'  # unverified supplementary text
    assert not signs[0].is_compound

def test_horizontal_neighbors_are_never_grouped():
    result=SignApplicability().group([sign('P.130',(10,10,50,50),panel_role='primary'),sign('S',(55,10,95,50),panel_role='supplementary')])
    assert len(result)==2

def test_unknown_role_panels_are_not_grouped():
    result=SignApplicability().group([sign('a',(10,10,50,50)),sign('b',(12,52,48,70))])
    assert len(result)==2

def lane():return LaneInfo([],ego_boundaries=[[(200,300),(200,599)],[(400,300),(400,599)]])

def test_overhead_neighbor_lane_without_review_is_uncertain():
    result=SignApplicability().apply([sign('P.127',(450,50,500,100),overhead=True)],lane(),(600,800,3))
    assert result[0].applicability=='uncertain'

def test_overhead_current_lane_without_review_is_uncertain():
    result=SignApplicability().apply([sign('P.127',(275,50,325,100),overhead=True)],lane(),(600,800,3))
    assert result[0].applicability=='uncertain'

def test_borderline_and_missing_geometry_stay_uncertain():
    for geometry in [lane(),LaneInfo([])]:
        result=SignApplicability().apply([sign('P.127',(180,50,220,100),overhead=True)],geometry,(600,800,3))
        assert result[0].applicability=='uncertain'

def test_unknown_semantic_cannot_apply_despite_geometry():
    result=SignApplicability().apply([sign('unknown',(275,50,325,100),overhead=True)],lane(),(600,800,3))
    assert result[0].applicability=='uncertain'
