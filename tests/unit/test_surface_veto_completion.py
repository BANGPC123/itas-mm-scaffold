import numpy as np
from src.perception.road_surface_veto import filter_sidewalk_overlap
from src.interaction.ai_observation import ai_observation
from src.utils.types import SignDetection

def test_sidewalk_curve_rejected_without_inventing_a_road_line():
    lines=[[(20,20),(20,30),(20,40),(20,50)],[(70,20),(70,30),(70,40),(70,50)]]
    mask=np.zeros((80,100),np.float32);mask[:,60:]=.9
    kept,removed=filter_sidewalk_overlap(lines,mask)
    assert kept==[lines[0]] and removed==1
    assert filter_sidewalk_overlap([],mask)==([],0)

def test_confident_tiny_video_sign_is_not_spoken():
    sign=SignDetection('P.127',.99,(10,10,26,26),recognition_confidence=.99,recognition_source='temporal_best_crop')
    assert ai_observation([sign],video=True)==''
    sign.bbox_xyxy=(10,10,60,60)
    assert ai_observation([sign],video=True)
