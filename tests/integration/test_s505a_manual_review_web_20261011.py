"""Web demo verified S.505a plate affects only specified vehicle class."""
from fastapi.testclient import TestClient
from backend.main import app

MAIN={'label':'P.131a','confidence':.96,'bbox_xyxy':[100,100,200,200],
      'panel_role':'primary','placement':'roadside','lane_scope':'all',
      'label_verified':True,'allowed_vehicles':[]}
CAR_PLATE={'label':'S.505a','confidence':.93,'bbox_xyxy':[117,204,187,235],
      'panel_role':'supplementary','allowed_vehicles':['car'],
      'label_verified':True,'placement':'roadside','lane_scope':'unknown'}

def interpret(signs,vehicle):
    c=TestClient(app)
    return c.post('/pipeline/interpret',json={
        'signs':signs,'width':1280,'height':720,'vehicle':vehicle,
        'ego_boundaries':[]})

def test_reviewer_added_s505a_car_limitation_narrows_P131a():
    car=interpret([MAIN,CAR_PLATE],'car')
    motorcycle=interpret([MAIN,CAR_PLATE],'motorcycle')
    assert car.status_code==motorcycle.status_code==200
    result=car.json()
    assert len(result['signs'])==1
    assert result['signs'][0]['is_compound']
    assert result['signs'][0]['panel_labels']==['P.131a','S.505a']
    assert result['signs'][0]['allowed_vehicles']==['car']
    assert result['signs'][0]['applicability']=='applies'
    assert motorcycle.json()['signs'][0]['applicability']=='ignored'
    assert motorcycle.json()['speech_text']==''

def test_no_secondary_sign_inherits_parking_motor_scope():
    for v in ('car','motorcycle','truck'):
        r=interpret([MAIN],v)
        assert r.status_code==200
        assert r.json()['signs'][0]['applicability']=='applies'

def test_unreadable_vehicle_plate_no_auto_scope():
    s={**CAR_PLATE,'allowed_vehicles':[],'label_verified':False}
    r=interpret([MAIN,s],'car')
    assert r.status_code==200
    assert r.json()['signs'][0]['applicability']=='uncertain'
    assert not r.json()['speech_text']
