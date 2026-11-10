"""Quality-first tests protecting actual user-video lane visualization."""
from pathlib import Path
import numpy as np
from src.utils.types import LaneInfo
from src.perception.ufld_display_screen import screen_ufld_for_demo
from src.perception.lane_multimodel import SecondaryLaneVisual

def lane(points):
    return LaneInfo([],polylines=points,ego_boundaries=points[:2],
                    backend='ufld_v2',lane_center_offset_px=35)

def test_plausible_left_road_mark_kept_above_dashboard_not_extended():
    image=np.full((720,1280,3),175,np.uint8)
    line=[(700-int((y-290)*.72),y) for y in range(300,720,12)]
    out,_=screen_ufld_for_demo(image,lane([line]))
    assert len(out.polylines)==1
    assert max(p[1] for line in out.polylines for p in line)<=.72*720
    assert out.ego_boundaries==[] and out.lane_center_offset_px is None

def test_zigzag_cross_vehicle_rejected():
    image=np.full((720,1280,3),200,np.uint8)
    line=[(280 if i%2 else 1000,290+i*12) for i in range(22)]
    out,rejected=screen_ufld_for_demo(image,lane([line]))
    assert rejected and out.polylines==[]
    assert out.rejection_reason=='no_credible_ufld_road_polyline'

def test_short_near_dashboard_fragment_does_not_become_lane():
    image=np.full((720,1280,3),200,np.uint8)
    line=[(620-i*8,500+i*7) for i in range(22)]
    out,rejected=screen_ufld_for_demo(image,lane([line]))
    assert rejected and out.polylines==[]

def test_horizontal_vehicle_edge_rejected_even_if_many_points():
    image=np.full((720,1280,3),200,np.uint8)
    line=[(160+i*30,300+i*6) for i in range(26)]
    out,rejected=screen_ufld_for_demo(image,lane([line]))
    assert rejected and out.polylines==[]

def test_reject_false_right_yolopv2_when_left_road_mark_valid():
    model=SecondaryLaneVisual('unused.pt',device='cpu')
    model.path=Path(__file__)
    class Candidate:
        def detect(self,image):
            left=[(670-int(i*5),240+i*12) for i in range(23)]
            right=[(720+i*18,240+i*12) for i in range(23)]
            return LaneInfo([],polylines=[left,right],backend='yolopv2')
    model.candidate=Candidate()
    arr=np.full((720,1280,3),200,np.uint8)
    result,used=model.detect_if_needed(arr,LaneInfo([],backend='ufld_v2'))
    assert used and len(result.polylines)==1
    assert result.polylines[0][0][0]==670
    assert result.ego_boundaries==[]

def test_road_like_but_wiggly_vehicle_edge_is_not_a_lane():
    image=np.full((720,1280,3),200,np.uint8)
    line=[(int(700+1.1*6*i+(6 if i%2 else -6)),310+6*i) for i in range(30)]
    out,rejected=screen_ufld_for_demo(image,lane([line]))
    assert rejected and not out.polylines


def test_parallel_edges_of_one_stripe_count_as_one_line():
    model=SecondaryLaneVisual('unused.pt',device='cpu')
    model.path=Path(__file__)
    class Candidate:
        def detect(self,image):
            left=[(700-5*i,230+12*i) for i in range(24)]
            same_stripe=[(740-5*i,230+12*i) for i in range(24)]
            return LaneInfo([],polylines=[left,same_stripe],backend='yolopv2')
    model.candidate=Candidate()
    image=np.full((720,1280,3),205,np.uint8)
    out,used=model.detect_if_needed(image,LaneInfo([],backend='ufld_v2'))
    assert used and len(out.polylines)==1 and not out.ego_boundaries


def test_model_without_lines_never_hallucinates_line():
    original=LaneInfo([],backend='ufld_v2')
    result,screened=screen_ufld_for_demo(
        np.full((720,1280,3),180,np.uint8),original)
    assert result is original and not screened
