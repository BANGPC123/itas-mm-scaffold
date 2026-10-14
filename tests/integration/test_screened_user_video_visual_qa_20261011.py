"""QA acceptance: latest REAL MP4, human-GT abstention, no fake accuracy."""
from pathlib import Path
import json,cv2
from fastapi.testclient import TestClient
from backend.main import app
ROOT=Path(__file__).resolve().parents[2]

def test_real_user_video_final_file_playable_and_screened():
    root=ROOT/'output/video_tests/user_test_video_lane_dedup_checked_20261011'
    info=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    assert info['inference_events']==225 and info['none_count']>0
    assert info['model_weights_overwritten'] is False
    file=root/'hybrid_lane_75s.mp4'
    assert file.stat().st_size>20_000_000
    capture=cv2.VideoCapture(str(file))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT))==1125
    capture.release()

def test_visual_qa_clearly_abstains_from_absolute_accuracy_claims():
    x=json.loads((ROOT/'output/evaluation/ITAS_MM_USER_VIDEO_VISUAL_QA_20261011.json').read_text(encoding='utf-8'))
    assert x['visually_inspected_unique_frames']==51
    assert not x['all_video_frames_visually_checked']
    assert not x['human_confirmed_lane_ground_truth_available']
    assert x['accuracy_f1_precision_recall_for_vietnam_video'] is None
    assert not x['claimed_absolute_correctness']
    assert x['no_prediction_events']>0

def test_api_video_route_points_to_screened_latest_not_old_bad_export():
    response=TestClient(app).get('/pipeline/capstone-video/user-hybrid',headers={'Range':'bytes=0-1023'})
    assert response.status_code==206
    assert b'ftyp' in response.content[:32]
    # response Content-Range byte size matches new output exactly.
    expected=(ROOT/'output/video_tests/user_test_video_lane_dedup_checked_20261011/hybrid_lane_75s.mp4').stat().st_size
    assert int(response.headers['Content-Range'].split('/')[-1])==expected
