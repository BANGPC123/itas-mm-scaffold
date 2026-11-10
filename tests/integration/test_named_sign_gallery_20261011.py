"""Prerendered second video must contain names and speech, not generic narration."""
import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/capstone_sign_gallery_named_tts_20261011'

@pytest.mark.legacy_artifact
def test_new_gallery_has_actual_scene_voice_and_vietnamese_sign_names():
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    assert summary['scenes']==24
    assert summary['spoken_scenes']>=5
    assert summary['scene_audio_aligned']
    assert not summary['source_frame_gt_verified']
    entries=[json.loads(x) for x in (OUT/'scene_events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(entries)==24
    known=next(x for x in entries if x['scene']=='frame_005')
    assert any(label=='P.130' and name=='Cấm dừng xe và đỗ xe'
               for label,name in known['labels'])
    assert 'P.130' in known['speech'] and 'chưa xác minh' in known['speech']

@pytest.mark.legacy_artifact
def test_live_web_video_route_serves_named_gallery():
    c=TestClient(app)
    response=c.get('/pipeline/capstone-video/signs',headers={'Range':'bytes=0-1023'})
    assert response.status_code==206
    assert response.headers['content-type'].startswith('video/mp4')
    assert int(response.headers['content-range'].split('/')[-1])>10_000_000

def test_runtime_video_default_still_keeps_roi_only_opt_in():
    from src.pipeline.orchestrator import Pipeline
    from inspect import signature
    assert signature(Pipeline.run).parameters['lane_visual_roi'].default is False
