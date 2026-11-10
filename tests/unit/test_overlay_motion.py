import cv2,numpy as np
from src.perception.overlay_motion import OverlayMotion
from src.perception.ocr_line_order import printed_lines


def test_exported_lane_moves_with_pixels_and_vanishes_when_occluded():
    image=np.full((180,300,3),60,np.uint8)
    cv2.line(image,(110,90),(145,150),(245,245,245),7)
    line=[(110+round((y-90)*35/60),y) for y in range(94,149,3)]
    tracker=OverlayMotion();tracker.reset(image,[line],[])
    moved=cv2.warpAffine(image,np.float32([[1,0,7],[0,1,4]]),(300,180),borderValue=(60,60,60))
    lines,_=tracker.advance(moved)
    assert lines and abs(lines[0][0][0]-line[0][0]-7)<=2
    assert abs(lines[0][0][1]-line[0][1]-4)<=2
    assert tracker.advance(np.full_like(image,60))[0]==[]


def test_split_ocr_time_line_uses_left_to_right_geometry_without_filling_dash():
    box=lambda x,y:[[x,y],[x+45,y],[x+45,y+12],[x,y+12]]
    texts,scores=printed_lines(['9:00-16:00','6:00','20:00-'],[.95,.98,.97],[box(5,5),box(75,30),box(5,30)])
    assert texts==['9:00-16:00','20:00- 6:00'] and scores==[.95,.97]
    texts,_=printed_lines(['6:00','20:00'],[.98,.97],[box(75,30),box(5,30)])
    assert texts==['20:00 6:00']


def test_ocr_separator_is_read_from_pixels_and_is_never_filled_in_blank_gap():
    image=np.full((60,180,3),240,np.uint8)
    boxes=[[[10,20],[60,20],[60,40],[10,40]],[[110,20],[155,20],[155,40],[110,40]]]
    assert printed_lines(['20:00','6:00'],[.97,.97],boxes,image)[0]==['20:00 6:00']
    cv2.rectangle(image,(75,29),(90,31),(25,25,25),-1)
    assert printed_lines(['20:00','6:00'],[.97,.97],boxes,image)[0]==['20:00 - 6:00']


def test_line_rerecognition_must_preserve_observed_time_values():
    image=np.full((60,180,3),240,np.uint8)
    boxes=[[[10,20],[60,20],[60,40],[10,40]],[[110,20],[155,20],[155,40],[110,40]]]
    for text in ['20:00-7:00','19:00-6:00']:
        result=printed_lines(['20:00','6:00'],[.97,.97],boxes,image,lambda crop:(text,.96))
        assert result[0]==['20:00 6:00']
    result=printed_lines(['20:00','6:00'],[.97,.97],boxes,image,lambda crop:('20:00-6:00',.91))
    assert result==(['20:00-6:00'],[.91])
