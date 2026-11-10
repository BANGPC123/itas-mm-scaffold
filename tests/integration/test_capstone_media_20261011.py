"""Capstone pre-rendered video serving is explicitly source-whitelisted."""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.mark.legacy_artifact
def test_capstone_status_matches_offline_only_requirements():
    r=TestClient(app).get('/pipeline/capstone-status')
    assert r.status_code==200
    data=r.json()
    assert data['offline_demo_ready'] is True
    assert data['camera_hardware_required'] is False
    assert data['real_gps_required'] is False
    assert not data['research_goals_fully_validated']
    assert len(data['remaining_gaps'])>=2

@pytest.mark.legacy_artifact
def test_capstone_video_stream_has_correct_mime_and_rejects_other_names():
    with TestClient(app) as client:
        for name in ('lane','signs'):
            result=client.get('/pipeline/capstone-video/'+name,
                  headers={'Range':'bytes=0-2047'})
            assert result.status_code in (200,206)
            assert result.headers['content-type'].startswith('video/mp4')
            assert b'ftyp' in result.content[:32]
        assert client.get('/pipeline/capstone-video/private').status_code==404
