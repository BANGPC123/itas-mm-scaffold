import cv2
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app

client=TestClient(app)

def test_invalid_image_is_rejected_without_model_loading():
    response=client.post('/pipeline/run',files={'image':('bad.jpg',b'broken','image/jpeg')},data={'latitude':10,'longitude':106})
    assert response.status_code==400

def test_invalid_gps_is_rejected():
    response=client.post('/pipeline/run',files={'image':('bad.jpg',b'broken','image/jpeg')},data={'latitude':91,'longitude':106})
    assert response.status_code==422

def test_audio_path_is_restricted():
    assert client.get('/pipeline/audio/not-a-generated-name.mp3').status_code==404

def test_status_is_independent_of_model_initialization():
    assert client.get('/pipeline/status').status_code==200


def test_video_speech_does_not_wait_for_inference_lock(monkeypatch):
    from backend.routers import pipeline
    from unittest.mock import Mock
    engine=Mock();engine.synthesize.return_value='output/audio/'+('a'*64)+'.mp3'
    monkeypatch.setattr(pipeline,'get_speech_engine',lambda:engine)
    with pipeline.lock:
        response=client.post('/pipeline/speech',json={'text':'Cấm đi ngược chiều'})
    assert response.status_code==200
    assert response.json()['audio_url'].endswith('a'*64+'.mp3')
    engine.synthesize.assert_called_once_with('Cấm đi ngược chiều')


def test_speech_failure_releases_separate_lock(monkeypatch):
    from backend.routers import pipeline
    from unittest.mock import Mock
    engine=Mock();engine.synthesize.side_effect=RuntimeError('offline')
    monkeypatch.setattr(pipeline,'get_speech_engine',lambda:engine)
    assert client.post('/pipeline/speech',json={'text':'Cấm đi ngược chiều'}).status_code==503
    assert not pipeline.speech_lock.locked()

def test_reviewed_truck_supplement_changes_car_applicability():
    response=client.post('/pipeline/interpret',json={'width':800,'height':600,'vehicle':'car','signs':[
        {'label':'P.130','confidence':.9,'recognition_confidence':.95,'bbox_xyxy':[100,100,200,200],'panel_role':'primary'},
        {'label':'truck panel','confidence':.8,'bbox_xyxy':[110,205,190,240],'panel_role':'supplementary','allowed_vehicles':['truck']} ]})
    assert response.status_code==200
    assert response.json()['signs'][0]['applicability']=='uncertain'  # unverified plate must not decide vehicle applicability
    assert response.json()['context_source']=='human_review'
    assert response.json()['signs'][0]['recognition_confidence']==.95

def test_review_rejects_reversed_bbox():
    response=client.post('/pipeline/interpret',json={'width':800,'height':600,'signs':[
        {'label':'P.130','confidence':.9,'bbox_xyxy':[200,100,100,200]}]})
    assert response.status_code==422

def test_busy_pipeline_rejects_overlap_without_queuing():
    from backend.routers.pipeline import lock
    _,encoded=cv2.imencode('.jpg',np.zeros((20,20,3),np.uint8))
    with lock:
        response=client.post('/pipeline/run',files={'image':('frame.jpg',encoded.tobytes(),'image/jpeg')},data={'latitude':10,'longitude':106})
    assert response.status_code==429

def test_valid_image_returns_serialized_overlay_and_timing(monkeypatch):
    from backend.routers import pipeline
    from src.utils.types import PipelineResult,PerceptionResult,LaneInfo,ZoneResult,ReasoningResult
    from unittest.mock import Mock
    result=PipelineResult(PerceptionResult([],LaneInfo([])),ZoneResult('Urban',10,106),ReasoningResult('No advice',[],'Urban'),timings_ms={'total':1},annotated_image='data:image/jpeg;base64,abc')
    mock=Mock();mock.run.return_value=result
    monkeypatch.setattr(pipeline,'get_pipeline',lambda:mock)
    _,encoded=cv2.imencode('.jpg',np.zeros((20,20,3),np.uint8))
    response=client.post('/pipeline/run',files={'image':('image.jpg',encoded.tobytes(),'image/jpeg')},data={'latitude':10,'longitude':106,'synthesize_audio':'false'})
    assert response.status_code==200
    assert response.json()['timings_ms']['total']==1
    assert response.json()['audio_url'] is None


def test_stream_options_reach_pipeline_without_server_overlay(monkeypatch):
    from backend.routers import pipeline
    from src.utils.types import PipelineResult,PerceptionResult,LaneInfo,ZoneResult,ReasoningResult
    from unittest.mock import Mock
    mock=Mock()
    mock.run.return_value=PipelineResult(PerceptionResult([],LaneInfo([])),ZoneResult('Urban',10,106),ReasoningResult('',[],'Urban'))
    monkeypatch.setattr(pipeline,'get_pipeline',lambda:mock)
    _,encoded=cv2.imencode('.jpg',np.zeros((20,20,3),np.uint8))
    response=client.post('/pipeline/run',files={'image':('frame.jpg',encoded.tobytes(),'image/jpeg')},data={'latitude':10,'longitude':106,'annotate':'false','road_bottom':.65,'synthesize_audio':'false'})
    assert response.status_code==200
    assert response.json()['annotated_image'] is None
    assert mock.run.call_args.kwargs=={'road_bottom':.65,'annotate':False}


def test_temporal_stream_requires_timestamp_pair():
    response=client.post('/pipeline/run',files={'image':('frame.jpg',b'broken','image/jpeg')},data={'latitude':10,'longitude':106,'stream_id':'session-1'})
    assert response.status_code==422


def test_temporal_fields_reach_pipeline(monkeypatch):
    from backend.routers import pipeline
    from src.utils.types import PipelineResult,PerceptionResult,LaneInfo,ZoneResult,ReasoningResult
    from unittest.mock import Mock
    mock=Mock();mock.run.return_value=PipelineResult(PerceptionResult([],LaneInfo([])),ZoneResult('Urban',10,106),ReasoningResult('',[],'Urban'))
    monkeypatch.setattr(pipeline,'get_pipeline',lambda:mock)
    _,encoded=cv2.imencode('.jpg',np.zeros((20,20,3),np.uint8))
    response=client.post('/pipeline/run',files={'image':('frame.jpg',encoded.tobytes(),'image/jpeg')},data={'latitude':10,'longitude':106,'stream_id':'session-1','frame_timestamp':1.25})
    assert response.status_code==200
    assert mock.run.call_args.kwargs['stream_id']=='session-1'
    assert mock.run.call_args.kwargs['frame_timestamp']==1.25


def test_recording_time_requires_timezone_and_is_passed_to_inference(monkeypatch):
    from backend.routers import pipeline
    from src.utils.types import PipelineResult,PerceptionResult,LaneInfo,ZoneResult,ReasoningResult
    from unittest.mock import Mock
    mock=Mock();mock.run.return_value=PipelineResult(PerceptionResult([],LaneInfo([])),ZoneResult('Urban',10,106),ReasoningResult('',[],'Urban'))
    monkeypatch.setattr(pipeline,'get_pipeline',lambda:mock)
    _,image=cv2.imencode('.jpg',np.zeros((20,20,3),np.uint8))
    data={'latitude':10,'longitude':106,'observed_at':'2026-10-12T08:00:00'}
    assert client.post('/pipeline/run',files={'image':('f.jpg',image.tobytes(),'image/jpeg')},data=data).status_code==422
    data['observed_at']='2026-10-12T08:00:00+07:00'
    r=client.post('/pipeline/run',files={'image':('f.jpg',image.tobytes(),'image/jpeg')},data=data)
    assert r.status_code==200 and mock.run.call_args.kwargs['observed_at'].hour==8
    assert r.json()['reasoning']['car_summary']==''



def test_reviewed_roadside_scope_respects_vehicle_and_lane():
    base={'label':'P.130','confidence':.95,'bbox_xyxy':[200,80,250,130],
          'panel_role':'primary','placement':'roadside','lane_scope':'all','label_verified':True}
    accepted=client.post('/pipeline/interpret',json={'width':800,'height':600,
        'vehicle':'car','signs':[base]})
    assert accepted.status_code==200
    assert accepted.json()['signs'][0]['applicability']=='applies'
    assert accepted.json()['signs'][0]['context_verified'] is True
    uncertain=client.post('/pipeline/interpret',json={'width':800,'height':600,
        'vehicle':'car','signs':[{**base,'lane_scope':'unknown'}]})
    assert uncertain.json()['signs'][0]['applicability']=='uncertain'
    other=client.post('/pipeline/interpret',json={'width':800,'height':600,
        'vehicle':'car','signs':[{**base,'lane_scope':'other'}]})
    assert other.json()['signs'][0]['applicability']=='ignored'


def test_verified_label_rejects_unknown_catalog_code():
    r=client.post('/pipeline/interpret',json={'width':800,'height':600,'vehicle':'car','signs':[
        {'label':'billboard','confidence':.97,'bbox_xyxy':[200,80,250,130],
         'panel_role':'primary','placement':'roadside','lane_scope':'all','label_verified':True}]})
    assert r.status_code==422


def test_roadside_scope_cannot_override_unknown_sign_or_supplement():
    for patch in [{'label':'unknown'},{'panel_role':'supplementary'}]:
        payload={'label':'P.130','confidence':.95,'bbox_xyxy':[200,80,250,130],
            'panel_role':'primary','placement':'roadside','lane_scope':'all',**patch}
        response=client.post('/pipeline/interpret',json={
            'width':800,'height':600,'vehicle':'car','signs':[payload]})
        assert response.status_code==200
        assert response.json()['signs'][0]['applicability']=='uncertain'



def test_status_is_explicitly_non_field_validated():
    r=client.get('/pipeline/status')
    assert r.status_code==200
    assert r.json()['field_validated'] is False
    assert r.json()['mode']=='research_preview'


def test_verified_supplementary_vehicle_scope_in_reviewed_api():
    base={'label':'P.130','confidence':.98,'bbox_xyxy':[100,100,200,200],
          'panel_role':'primary','placement':'roadside','lane_scope':'all','label_verified':True}
    sub={'label':'truck panel','confidence':.94,'bbox_xyxy':[115,205,185,240],
         'panel_role':'supplementary','allowed_vehicles':['truck'],'label_verified':True}
    for vehicle, expected in [('car','ignored'),('truck','applies')]:
        resp=client.post('/pipeline/interpret',json={
            'width':800,'height':600,'vehicle':vehicle,'signs':[base,sub]})
        assert resp.status_code==200
        assert resp.json()['signs'][0]['applicability']==expected
    assert 'xe tải' in resp.json()['speech_text']


def test_unstructured_supplement_cannot_be_marked_verified():
    sub={'label':'unreadable plate','confidence':.9,'bbox_xyxy':[115,205,185,240],
         'panel_role':'supplementary','label_verified':True}
    r=client.post('/pipeline/interpret',json={
        'width':800,'height':600,'signs':[sub]})
    assert r.status_code==422



def test_camera_exclusion_zone_schema_rejects_invalid_values():
    for zones in ['{"x":1}', '[[.1,.1,1.4,.4]]', '[[0,0,0,1]]', '[[0,NaN,.5,.5]]']:
        response=client.post('/pipeline/run',
            files={'image':('bad.jpg',b'bad','image/jpeg')},
            data={'latitude':10,'longitude':106,'exclusion_zones':zones})
        assert response.status_code==422


def test_camera_exclusion_zones_pass_to_pipeline(monkeypatch):
    import json
    from backend.routers import pipeline
    from src.utils.types import PipelineResult,PerceptionResult,LaneInfo,ZoneResult,ReasoningResult
    from unittest.mock import Mock
    mock=Mock()
    mock.run.return_value=PipelineResult(PerceptionResult([],LaneInfo([])),ZoneResult('Urban',10,106),
                                         ReasoningResult('',[],'Urban'))
    monkeypatch.setattr(pipeline,'get_pipeline',lambda:mock)
    _,enc=cv2.imencode('.jpg',np.zeros((100,100,3),np.uint8))
    r=client.post('/pipeline/run',files={'image':('frame.jpg',enc.tobytes(),'image/jpeg')},
        data={'latitude':10,'longitude':106,'synthesize_audio':'false',
              'exclusion_zones':json.dumps([[.1,.1,.3,.3]])})
    assert r.status_code==200
    assert mock.run.call_args.kwargs['exclude_rects']==[[.1,.1,.3,.3]]
