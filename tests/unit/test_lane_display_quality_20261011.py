"""Test visual-only lane filtering; raw model geometry must remain unchanged."""
from src.perception.lane_display_quality import display_quality,visual_polylines
from src.utils.types import LaneInfo
def test_straight_road_paint_remains_visible():
    line=[(250+i*8,100+i*25) for i in range(13)]
    assert display_quality(line,(720,1280,3))
def test_horizontal_car_bumper_is_hidden():
    line=[(300+i*25,400+i*3) for i in range(15)]
    assert not display_quality(line,(720,1280,3))
def test_zigzag_car_outline_is_hidden():
    line=[(300+(60 if i%2 else -60),110+i*27) for i in range(15)]
    assert not display_quality(line,(720,1280,3))
def test_unfiltered_original_model_geometry_preserved():
    valid=[(250+i*8,100+i*25) for i in range(13)]
    wrong=[(300+i*25,400+i*3) for i in range(15)]
    lane=LaneInfo([],polylines=[valid,wrong],backend='ufld_v2')
    result=visual_polylines(lane,(720,1280,3))
    assert result==[valid]
    assert lane.polylines==[valid,wrong]
