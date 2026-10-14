import threading
import time
import cv2
import numpy as np
from src.perception.camera_lane import paint_evidence,refine_candidates,visible_paint_segments
from src.perception.supplementary_reader import SupplementaryReader
from src.utils.types import SignDetection


def test_paint_refinement_requires_visible_pixels_and_excludes_occluder():
    image=np.full((240,400,3),65,np.uint8)
    cv2.line(image,(200,95),(200,215),(240,240,240),5)
    road=np.ones((240,400),np.float32)
    seed=[(200,y) for y in range(95,216,4)]
    mask=paint_evidence(image,road)
    assert refine_candidates([seed],mask,image.shape)
    covered=paint_evidence(image,road,[(180,80,220,230)])
    assert not refine_candidates([seed],covered,image.shape)
    assert not refine_candidates([seed],np.zeros_like(mask),image.shape)


def test_transverse_zebra_is_not_a_longitudinal_seed():
    image=np.full((240,400,3),65,np.uint8)
    for y in (100,130,160,190):cv2.rectangle(image,(100,y),(300,y+8),(240,240,240),-1)
    mask=paint_evidence(image,np.ones((240,400),np.float32))
    seed=[(200,y) for y in range(90,220,3)]
    assert not refine_candidates([seed],mask,image.shape)


def test_paint_is_retained_when_drivable_head_excludes_white_marking():
    image=np.full((240,400,3),65,np.uint8)
    cv2.line(image,(200,95),(200,215),(240,240,240),5)
    road=np.ones((240,400),np.float32);paint=np.zeros_like(road)
    road[:,197:204]=0;paint[95:216,197:204]=1
    seed=[(200,y) for y in range(95,216,4)]
    mask=paint_evidence(image,road,paint_probability=paint)
    assert refine_candidates([seed],mask,image.shape)


def test_video_ocr_does_not_block_or_reuse_a_result_after_seek():
    reader=SupplementaryReader();entered=threading.Event();release=threading.Event()
    second_entered=threading.Event();second_release=threading.Event();calls=[]
    sign=SignDetection('P.130',.9,(10,10,40,40),track_id=1)
    image=np.zeros((80,100,3),np.uint8)
    def slow_read(image,signs,**kwargs):
        calls.append(kwargs['timestamp'])
        if len(calls)==1:entered.set();assert release.wait(3)
        else:second_entered.set();assert second_release.wait(3)
        return [{'primary_label':'P.130','primary_track_id':1,'bbox_xyxy':[10,42,40,60],'raw_texts':[str(kwargs['timestamp'])]}]
    reader.read=slow_read
    begin=time.perf_counter();assert reader.read_video(image,[sign],stream_id='a',timestamp=10)==[]
    assert time.perf_counter()-begin<.2 and entered.wait(1)
    # Tua rồi tạo lại cùng track ID: kết quả đang đọc thuộc phiên cũ.
    assert reader.read_video(image,[],stream_id='a',timestamp=1)==[]
    assert reader.read_video(image,[sign],stream_id='a',timestamp=1.1)==[]
    release.set();assert second_entered.wait(1)
    # The old result has finished, while the new result is still blocked.
    assert reader.read_video(image,[sign],stream_id='a',timestamp=1.2)==[]
    second_release.set()
    deadline=time.monotonic()+1
    while 'a' not in reader.completed and time.monotonic()<deadline:time.sleep(.01)
    result=reader.read_video(image,[sign],stream_id='a',timestamp=1.3)
    assert result and result[0]['raw_texts']==['1.1']
    reader.close()
    assert reader.read_video(image,[sign],stream_id='a',timestamp=1.4)==[]


def test_one_sided_asphalt_shadow_is_not_a_paint_stripe():
    image=np.full((240,400,3),200,np.uint8)
    image[:,:200]=70
    seed=[(200,y) for y in range(95,216,4)]
    mask=paint_evidence(image,np.ones((240,400),np.float32))
    assert not refine_candidates([seed],mask,image.shape)


def test_semantic_road_mask_rejects_bright_dashboard_line():
    image=np.full((240,400,3),65,np.uint8)
    cv2.line(image,(200,95),(200,215),(240,240,240),5)
    seed=[(200,y) for y in range(95,216,4)]
    road=np.ones((240,400),np.float32)
    mask=paint_evidence(image,road,ground_probability=np.zeros_like(road))
    assert not refine_candidates([seed],mask,image.shape)


def test_display_never_connects_observed_paint_across_an_occluder():
    mask=np.zeros((100,200),np.uint8);mask[20:40,100]=1;mask[60:90,100]=1
    line=[(100,y) for y in range(20,90,2)]
    pieces=visible_paint_segments([line],mask)
    assert len(pieces)==2
    assert pieces[0][-1][1]<40 and pieces[1][0][1]>=60


def test_small_observed_dash_is_not_discarded_by_long_line_display_limit():
    mask=np.zeros((100,200),np.uint8);mask[60:66,100]=1
    assert visible_paint_segments([[(100,y) for y in range(50,90,2)]],mask)


def test_perspective_zebra_row_is_rejected_but_center_lane_dashes_remain():
    image=np.full((240,400,3),65,np.uint8)
    for x in (60,110,280,330):
        cv2.fillConvexPoly(image,np.array([[x,170],[x+10,150],[x+30,150],[x+20,170]],np.int32),(240,240,240))
    for y in (90,110,130):cv2.rectangle(image,(198,y),(204,y+7),(240,240,240),-1)
    mask=paint_evidence(image,np.ones((240,400),np.float32))
    assert mask[90:138,196:206].any()
    assert not mask[150:171,55:150].any()


def test_faded_white_needs_independent_paint_head_and_pixel_contrast():
    image=np.full((240,400,3),65,np.uint8)
    cv2.line(image,(200,95),(200,215),(135,135,135),5)
    road=np.ones((240,400),np.float32);paint=np.zeros_like(road)
    assert not paint_evidence(image,road,paint_probability=paint).any()
    paint[95:216,197:204]=.9
    assert paint_evidence(image,road,paint_probability=paint).any()
    assert not paint_evidence(image,road,[(180,80,220,230)],paint,road).any()


def test_visible_yellow_is_not_erased_by_negative_bdd_paint_prediction():
    image=np.full((240,400,3),65,np.uint8)
    cv2.line(image,(200,95),(200,215),(20,180,200),5)
    road=np.ones((240,400),np.float32);paint=np.zeros_like(road)
    assert paint_evidence(image,road,paint_probability=paint,ground_probability=road).any()
    assert not paint_evidence(image,road,paint_probability=paint,ground_probability=np.zeros_like(road)).any()
