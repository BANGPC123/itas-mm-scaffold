"""Offline Video MP4 YOLOPv2 lane fallback release checks; not field accreditation."""
from pathlib import Path
import json,torch
from fastapi.testclient import TestClient
from backend.main import app
from backend.schemas import LaneResponse
from src.utils.config import load_config

ROOT=Path(__file__).resolve().parents[2]
CAND='models/lane/candidates/yolopv2_official_20261011/yolopv2.pt'
VIDEO='output/video_tests/user_test_video_yolopv2_hybrid_20261011/hybrid_lane_75s.mp4'

def test_primary_lane_model_weight_not_swapped():
    c=load_config('perception')
    assert c['lane_detector']['backend']=='ufld_v2'
    assert c['lane_detector']['weights_path']=='models/lane/selected_20261009.pth'
    assert c['lane_visual_fallback']['model']=='yolopv2'
    assert c['lane_visual_fallback']['weights_path']==CAND
    assert (ROOT/CAND).stat().st_size>100_000_000

def test_inference_contract_exposes_temporal_display_for_web_overlay():
    m=LaneResponse(lane_lines=[],lane_center_offset_px=None,polylines=[],
        display_polylines=[[(100,120),(120,170)]],ego_boundaries=[],
        backend='yolopv2_visual_candidate')
    assert m.model_dump(mode='json')['display_polylines']==[[[100,120],[120,170]]]
    assert m.ego_boundaries==[]

def test_real_user_video_is_sampled_and_missing_marks_are_not_faked():
    rep=json.loads((ROOT/'output/video_tests/user_test_video_yolopv2_hybrid_20261011/summary.json').read_text(encoding='utf-8'))
    assert rep['inference_events']>=200
    assert rep['none_count']>0
    assert rep['backend_positive_counts']['ufld_v2']>0
    assert rep['backend_positive_counts']['yolopv2_visual_candidate']>0
    assert not rep['fallback_used_for_legal_scope']
    assert not rep['model_weights_overwritten']
    assert not rep['independent_road_gt_verified']

def test_quantitative_culane_val_rejects_secondary_as_primary():
    d=json.loads((ROOT/'output/evaluation/yolopv2_culane_val_20261011/results.json').read_text(encoding='utf-8'))
    assert d['checked_frames']==70
    assert d['results']['ufld']['f1']>.8
    assert d['results']['yolopv2']['f1']<d['results']['ufld']['f1']
    assert d['results']['fallback']['f1']>=d['results']['ufld']['f1']-.005
    assert d['not_independent_Vietnam_gt']

def test_new_vietnam_video_route_is_whitelisted_and_partial_content():
    with TestClient(app) as client:
        out=client.get('/pipeline/capstone-video/user-hybrid',
                       headers={'Range':'bytes=0-1023'})
        assert out.status_code==206
        assert out.headers['content-type'].startswith('video/mp4')
        assert b'ftyp' in out.content[:32]
        assert client.get('/pipeline/capstone-video/../../secret').status_code in (404,422)
