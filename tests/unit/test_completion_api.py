from fastapi.testclient import TestClient
from backend.main import app

client=TestClient(app)


def payload(code='P.131a'):
    return {'width':800,'height':600,'vehicle':'motorbike','signs':[
        {'label':code,'confidence':.95,'bbox_xyxy':[100,100,200,200],
         'panel_role':'primary','placement':'roadside','lane_scope':'all','label_verified':True}]}


def test_api_keeps_verified_scope_metadata():
    response=client.post('/pipeline/interpret',json=payload())
    assert response.status_code==200
    assert response.json()['signs'][0]['label_verified'] is True
    assert response.json()['signs'][0]['context_verified'] is True
    assert response.json()['signs'][0]['applicability']=='applies'


def test_api_never_substitutes_server_clock_for_video_time():
    unknown=client.post('/pipeline/interpret',json=payload('P.131b'))
    assert unknown.status_code==200
    assert unknown.json()['signs'][0]['applicability']=='uncertain'
    naive={**payload('P.131b'),'observed_at':'2026-10-11T08:30:00'}
    assert client.post('/pipeline/interpret',json=naive).status_code==422
    dated={**payload('P.131b'),'observed_at':'2026-10-11T08:30:00+07:00'}
    assert client.post('/pipeline/interpret',json=dated).json()['signs'][0]['applicability']=='applies'


def test_api_validates_and_applies_reviewed_time_panel():
    body=payload()
    body['vehicle']='car'
    body['observed_at']='2026-10-11T08:30:00+07:00'
    sub={'label':'S.508a','confidence':.9,'bbox_xyxy':[110,210,190,240],
         'panel_role':'supplementary','label_verified':True,'time_windows':[['07:00','09:00']]}
    body['signs'].append(sub)
    result=client.post('/pipeline/interpret',json=body)
    assert result.status_code==200
    assert result.json()['signs'][0]['applicability']=='applies'
    assert '07:00 đến 09:00' in result.json()['speech_text']
    sub['time_windows']=[['25:00','09:00']]
    assert client.post('/pipeline/interpret',json=body).status_code==422
