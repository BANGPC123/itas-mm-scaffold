import json
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.routers import pipeline


@pytest.mark.parametrize('failure', ['incomplete', 'inference', 'tts', 'verification'])
def test_complete_video_is_not_published_before_every_operational_check(tmp_path,monkeypatch,failure):
    monkeypatch.setattr(pipeline,'VIDEO_COMPLETION_DIR',tmp_path)
    (tmp_path/'test_video_annotated.mp4').write_bytes(b'not-yet-verified')
    summary={'complete_source_scan':failure!='incomplete',
             'failures':['failed'] if failure=='inference' else [],
             'tts_failures':['failed'] if failure=='tts' else []}
    (tmp_path/'summary.json').write_text(json.dumps(summary))
    (tmp_path/'verification.json').write_text(json.dumps({'passed':failure!='verification'}))
    client=TestClient(app)
    assert not client.get('/pipeline/video-completion-status').json()['complete']
    assert client.get('/pipeline/capstone-video/user-complete').status_code==404


def test_video_status_missing_artifacts_is_explicit(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'VIDEO_COMPLETION_DIR',tmp_path)
    data=TestClient(app).get('/pipeline/video-completion-status').json()
    assert not data['complete'] and data['video_url'] is None
    assert not data['independent_video_accuracy_available']


def test_verified_video_range_and_changed_artifact_gate(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'VIDEO_COMPLETION_DIR',tmp_path)
    video=tmp_path/'test_video_annotated.mp4';video.write_bytes(b'\x00\x00\x00\x18ftypisom'+bytes(8192))
    monkeypatch.setitem(pipeline.CAPSTONE_VIDEOS,'user-complete',str(video))
    (tmp_path/'summary.json').write_text(json.dumps({'complete_source_scan':True,'failures':[],'tts_failures':[]}))
    (tmp_path/'verification.json').write_text(json.dumps({'passed':True,
        'video_size':video.stat().st_size,'video_mtime_ns':video.stat().st_mtime_ns}))
    client=TestClient(app)
    assert client.get('/pipeline/video-completion-status').json()['complete']
    response=client.get('/pipeline/capstone-video/user-complete',headers={'Range':'bytes=0-1023'})
    assert response.status_code==206 and len(response.content)==1024
    video.write_bytes(video.read_bytes()+b'changed')
    assert not client.get('/pipeline/video-completion-status').json()['complete']


def test_independent_video_needs_verified_unchanged_bytes(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'INDEPENDENT_VIDEO_DIR',tmp_path)
    video=tmp_path/'independent_car_annotated.mp4';video.write_bytes(b'\x00\x00\x00\x18ftypisom'+bytes(8192))
    monkeypatch.setitem(pipeline.CAPSTONE_VIDEOS,'independent-car',str(video))
    client=TestClient(app)
    assert client.get('/pipeline/capstone-video/independent-car').status_code==404
    (tmp_path/'report.json').write_text(json.dumps({'passed':True,'video_size':video.stat().st_size,'video_mtime_ns':video.stat().st_mtime_ns}))
    assert client.get('/pipeline/video-completion-status').json()['independent_video']['complete']
    assert client.get('/pipeline/capstone-video/independent-car',headers={'Range':'bytes=0-1023'}).status_code==206
    video.write_bytes(video.read_bytes()+b'changed')
    assert not client.get('/pipeline/video-completion-status').json()['independent_video']['complete']
    assert client.get('/pipeline/capstone-video/independent-car').status_code==404
