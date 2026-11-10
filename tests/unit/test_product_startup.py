import json
from types import SimpleNamespace
from unittest.mock import Mock
import cv2
import numpy as np

from backend.startup import warmup_product


def test_startup_warms_jpeg_camera_inputs_and_removes_bootstrap_tracking(tmp_path, monkeypatch):
    directory = tmp_path / 'output/video_completion_20261011/surface_final_current'
    directory.mkdir(parents=True)
    (directory / 'summary.json').write_text(json.dumps({'source': 'test_video.mp4'}))
    rng = np.random.default_rng(12)
    frame = rng.integers(0, 256, (80, 120, 3), dtype=np.uint8)
    capture = SimpleNamespace(set=Mock(), read=lambda: (True, frame.copy()), release=Mock())
    monkeypatch.setattr(cv2, 'VideoCapture', lambda source: capture)
    seen = []
    pipeline = SimpleNamespace(tts_engine=Mock(), auto_supplementary_enabled=True,
                               stream_evidence=SimpleNamespace(sessions={'server-warmup': 'pending'}),
                               _verified_zone_streams={'server-warmup': 'pending'},supplementary_reader=Mock())
    def run(image, *args, **kwargs):
        assert not pipeline.auto_supplementary_enabled
        assert kwargs['stream_id'] == 'server-warmup'
        seen.append(image)
    pipeline.run = run
    speech = Mock(provider='vieneu-onnx')
    result = warmup_product(pipeline, speech, root=tmp_path)
    assert len(seen) == 3 and not np.array_equal(seen[0], frame)
    assert capture.release.called and pipeline.auto_supplementary_enabled
    assert pipeline.stream_evidence.sessions == {} and pipeline._verified_zone_streams == {}
    assert pipeline.tts_engine is speech and speech.warmup.called
    assert result['camera_warmup_frames'] == 3
    assert pipeline.supplementary_reader._ocr.call_count==1
    assert result['ocr_ready_before_http']
