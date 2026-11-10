from datetime import datetime,timezone
import cv2,numpy as np
import pytest
from src.context.car_context import assess_car
from src.perception.supplementary_semantics import parse_supplementary
from src.perception.traffic_symbols import templates,blue_symbol_verdict,read_vehicle_icon
from src.perception.camera_lane import camera_view_from_ground,observed_paint_seeds,paint_evidence,refine_candidates
from src.utils.types import SignDetection

IMAGE=np.zeros((180,260,3),np.uint8)
BOX=[100,65,150,90]


def primary(code='P.130'):
    return SignDetection(code,.9,(100,20,150,65),track_id=3,
        recognition_confidence=.95,recognition_source='temporal_best_crop')


def panel(**kwargs):
    return {'primary_label':'P.130','primary_track_id':3,'bbox_xyxy':BOX,**kwargs}


@pytest.fixture
def has_plate(monkeypatch):
    monkeypatch.setattr('src.context.car_context.propose_plates',lambda *a,**k:[{'bbox_xyxy':BOX}])


def test_car_base_scope_does_not_turn_into_verified_legal_effect(monkeypatch):
    monkeypatch.setattr('src.context.car_context.propose_plates',lambda *a,**k:[])
    s=primary();r=assess_car(s,IMAGE,video=True)
    assert r['status']=='applies_candidate' and r['vehicle_scope']=='included'
    assert not r['human_verified'] and not r['legal_effect_established']
    assert not s.context_verified and not s.label_verified and s.applicability=='uncertain'


def test_unreadable_plate_blocks_unconditional_car_message(has_plate):
    r=assess_car(primary(),IMAGE,video=True)
    assert r['status']=='conditional' and 'unreadable_supplementary' in r['unresolved']


def test_recent_plate_does_not_become_unconditional_when_proposal_is_missing(monkeypatch):
    monkeypatch.setattr('src.context.car_context.propose_plates',lambda *a,**k:[])
    row=panel(variant_agreement=True,time_windows=[['09:00','16:00']])
    r=assess_car(primary(),IMAGE,[row],video=True)
    assert r['status']=='conditional' and 'supplementary_not_visible' in r['unresolved']


@pytest.mark.parametrize('polarity,status',[('unspecified','excluded_candidate'),('exclusion','applies_candidate')])
def test_truck_pictogram_scopes_car_in_the_correct_direction(has_plate,polarity,status):
    row=panel(vehicle_variant_agreement=True,vehicle_candidates=['truck'],condition_polarity=polarity)
    r=assess_car(primary(),IMAGE,[row],video=True)
    assert r['status']==status
    assert r['vehicle_scope']==('excluded' if polarity=='unspecified' else 'included')


def test_time_panel_needs_recording_time_then_uses_vietnam_zone(has_plate):
    row=panel(variant_agreement=True,time_windows=[['07:00','09:00']])
    s=primary();r=assess_car(s,IMAGE,[row],video=True)
    assert r['status']=='conditional' and r['unresolved']==['recording_date_time']
    r=assess_car(s,IMAGE,[row],video=True,observed_at=datetime(2026,10,12,5,tzinfo=timezone.utc))
    assert r['status']=='excluded_candidate'
    r=assess_car(s,IMAGE,[row],video=True,observed_at=datetime(2026,10,12,1,tzinfo=timezone.utc))
    assert r['status']=='applies_candidate'
    assert not s.context_verified


def test_stale_other_track_plate_cannot_scope_car(has_plate):
    row=panel(vehicle_variant_agreement=True,vehicle_candidates=['truck'],condition_polarity='unspecified')
    row['primary_track_id']=999
    r=assess_car(primary(),IMAGE,[row],video=True)
    assert r['status']=='conditional' and r['vehicle_scope']=='included'


def test_mirror_and_unconfirmed_sign_do_not_get_applicable_car_context():
    s=primary()
    assert assess_car(s,IMAGE,video=True,camera_view='side_or_cabin')['status']=='uncertain'
    s.recognition_source='single_frame'
    assert assess_car(s,IMAGE,video=True)['status']=='uncertain'


@pytest.mark.parametrize('text,expected',[('Ô tô tải',['truck']),('Ô tô con',['car']),('Ô tô khách',['bus']),('Ô tô',['automobile']),('Trừ xe tải',['truck'])])
def test_vehicle_text_keeps_truck_distinct_from_passenger_car(text,expected):
    r=parse_supplementary([text],[.97])
    assert r['vehicle_candidates']==expected
    if text.startswith('Trừ'):assert r['condition_polarity']=='exclusion'


def test_official_uturn_shape_is_required_for_a_blue_logo():
    image=np.full((100,100,3),(180,100,20),np.uint8)
    mask=cv2.resize(templates()['I.409'].astype(np.uint8),(58,60),interpolation=cv2.INTER_NEAREST)
    image[20:80,21:79][mask>0]=240
    assert blue_symbol_verdict(image,'I.409') is None
    logo=np.full_like(image,(180,100,20));cv2.putText(logo,'k',(22,77),cv2.FONT_HERSHEY_SIMPLEX,2.5,(240,240,240),8)
    assert blue_symbol_verdict(logo,'I.409')=='symbol_shape_unsupported'


def test_vehicle_glyph_reader_accepts_truck_but_not_direction_arrow():
    image=np.full((90,150,3),240,np.uint8)
    m=cv2.resize(templates()['truck'].astype(np.uint8),(90,45),interpolation=cv2.INTER_NEAREST)
    image[25:70,30:120][m>0]=25
    result=read_vehicle_icon(image)
    assert result and result['vehicle']=='truck' and result['variant_agreement']
    arrow=np.full_like(image,240);cv2.arrowedLine(arrow,(30,40),(120,40),(20,20,20),9,tipLength=.25)
    assert read_vehicle_icon(arrow) is None


def test_ground_distribution_identifies_side_camera_and_missing_road():
    ground=np.zeros((100,200),np.float32)
    assert camera_view_from_ground(ground)=='unknown'
    ground[32:70,:45]=1
    assert camera_view_from_ground(ground)=='unknown'
    assert camera_view_from_ground(ground,np.full((100,200,3),25,np.uint8))=='side_or_cabin'
    assert camera_view_from_ground(ground,np.full((100,200,3),170,np.uint8))=='forward'
    right_ground=np.zeros_like(ground);right_ground[32:70,155:]=1
    left_cabin=np.full((100,200,3),170,np.uint8);left_cabin[:,:80]=25
    assert camera_view_from_ground(right_ground,left_cabin)=='side_or_cabin'
    assert camera_view_from_ground(right_ground,np.full((100,200,3),170,np.uint8))=='forward'
    ground[32:70,70:140]=1
    assert camera_view_from_ground(ground)=='forward'


def test_separated_short_dashes_can_seed_a_longitudinal_path():
    image=np.full((240,400,3),60,np.uint8)
    for y in (90,112,140,175):cv2.rectangle(image,(198,y),(203,y+8),(240,240,240),-1)
    evidence=paint_evidence(image,np.ones((240,400),np.float32))
    lines=refine_candidates(observed_paint_seeds(evidence),evidence,image.shape)
    assert lines and np.ptp(np.asarray(lines[0])[:,1])>65


@pytest.mark.parametrize('shape',[(2,1,4),(2,4)])
def test_observed_paint_seeds_accepts_hough_segment_shapes(monkeypatch,shape):
    segments=np.asarray([[200,90,201,190],[150,95,151,185]],dtype=np.int32).reshape(shape)
    monkeypatch.setattr(cv2,'HoughLinesP',lambda *args,**kwargs:segments)
    seeds=observed_paint_seeds(np.zeros((240,400),np.uint8))
    assert len(seeds)==2
    assert all(np.ptp(np.asarray(line)[:,1])>60 for line in seeds)
