import cv2
import numpy as np
from src.perception.crosswalk_veto import crossing_mask,reject_crosswalk_segments
from src.perception.camera_lane import camera_view_from_ground


def crossing():
    im=np.full((240,400,3),65,np.uint8)
    for x in (50,150,250):cv2.rectangle(im,(x,155),(x+65,175),(225,225,225),-1)
    return im


def test_broad_white_crossing_rejected_while_longitudinal_yellow_remains():
    im=crossing();cv2.line(im,(200,95),(200,135),(20,180,200),5)
    stripes=[[(80,y) for y in range(155,176,4)],[(200,y) for y in range(95,136,4)]]
    kept=reject_crosswalk_segments(im,stripes,ground_probability=np.ones((240,400),np.float32))
    assert kept==[stripes[1]]


def test_white_vehicles_outside_semantic_road_do_not_remove_paint():
    im=crossing();ground=np.zeros((240,400),np.float32)
    assert not crossing_mask(im,ground).any()
    lines=[[(80,y) for y in range(155,176,4)]]
    assert reject_crosswalk_segments(im,lines,ground_probability=ground)==lines


def test_rejection_preserves_original_coordinates_at_native_resolution():
    im=crossing();line=[(160,2*y) for y in range(130,191,4)]
    kept=reject_crosswalk_segments(im,[line],coordinate_scale=.5)
    original=set(line)
    assert all(p in original for piece in kept for p in piece)
    assert all(not 155<=p[1]/2<=175 for piece in kept for p in piece)


def test_bright_side_window_with_only_low_offcenter_road_is_uncertain():
    ground=np.zeros((100,200),np.float32);ground[63:74,165:198]=.95
    assert camera_view_from_ground(ground,np.full((100,200,3),190,np.uint8))=='unknown'
    ground=np.zeros((100,200),np.float32);ground[52:74,35:130]=.95
    assert camera_view_from_ground(ground,np.full((100,200,3),190,np.uint8))=='forward'
