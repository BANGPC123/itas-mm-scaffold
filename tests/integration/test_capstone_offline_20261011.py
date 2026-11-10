"""End-to-end CAPSTONE acceptance does not depend on real camera/GPS or live Ollama."""
import pytest
from pathlib import Path
import json,cv2
from fastapi.testclient import TestClient
from backend.main import app
ROOT=Path(__file__).resolve().parents[2]
@pytest.mark.legacy_artifact
def test_desktop_browser_bundle_and_offline_video_artifacts():
    doc=json.loads((ROOT/'output/capstone_demo_20261011/OFFLINE_CAPSTONE_ACCEPTANCE.json').read_text(encoding='utf-8'))
    assert doc['offline_mp4_visual_demo_ready'] and not doc['camera_hardware_needed'] and not doc['real_gps_needed']
    assert not doc['penalties_auto_readable_from_decree_168']
    for row in doc['video_artifacts'].values():
        p=ROOT/row['file']
        assert p.is_file() and p.stat().st_size>500_000
        capture=cv2.VideoCapture(str(p))
        assert capture.isOpened() and capture.get(cv2.CAP_PROP_FRAME_COUNT)>100
        capture.release()
    assert (ROOT/'frontend/dist/index.html').is_file()

def test_manual_review_can_complete_perception_retrieval_and_tts_text_without_gps():
    client=TestClient(app)
    s={'label':'P.130','confidence':.99,'bbox_xyxy':[110,90,210,190],
       'panel_role':'primary','label_verified':True,'placement':'roadside','lane_scope':'all'}
    response=client.post('/pipeline/interpret',json={
       'signs':[s],'ego_boundaries':[],'width':1280,'height':720,'vehicle':'car'})
    assert response.status_code==200
    doc=response.json()
    assert doc['context_source']=='human_review'
    assert doc['speech_text'] and 'Cấm' in doc['speech_text']
    assert doc['references'] and doc['retrieved_chunks']
    assert doc['penalty_verified'] is False and not doc['legal_advice_generated']

def test_unverified_detection_remains_silent_even_if_known_code():
    client=TestClient(app)
    s={'label':'P.130','confidence':.99,'bbox_xyxy':[110,90,210,190],
        'panel_role':'primary','label_verified':False,'placement':'roadside','lane_scope':'all'}
    r=client.post('/pipeline/interpret',json={'signs':[s],
         'ego_boundaries':[],'width':1280,'height':720,'vehicle':'car'})
    assert r.status_code==200
    assert r.json()['speech_text']==''
