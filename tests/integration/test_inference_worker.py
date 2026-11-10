import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.routers import pipeline
from src.utils.types import LaneInfo, PerceptionResult, PipelineResult, ReasoningResult, ZoneResult


def test_api_frames_use_the_same_thread_as_model_warmup(monkeypatch):
    owner = pipeline.get_inference_worker()
    warmup_thread = owner.submit(threading.get_ident).result()
    observed = []
    def run(*args, **kwargs):
        observed.append(threading.get_ident())
        return PipelineResult(PerceptionResult([], LaneInfo([])), ZoneResult('Urban',10,106),
                              ReasoningResult('', [], 'Urban'))
    monkeypatch.setattr(pipeline, 'get_pipeline', lambda: SimpleNamespace(run=run))
    ok, blob = cv2.imencode('.jpg', np.zeros((80, 100, 3), dtype=np.uint8))
    assert ok
    client = TestClient(app)
    def request():
        return client.post('/pipeline/run', files={'image': ('frame.jpg', blob.tobytes(), 'image/jpeg')},
                           data={'latitude': 10, 'longitude': 106, 'synthesize_audio': 'false'}).status_code
    # Different caller threads must not create different CUDA inference threads.
    with ThreadPoolExecutor(max_workers=1) as first:
        assert first.submit(request).result() == 200
    with ThreadPoolExecutor(max_workers=1) as second:
        assert second.submit(request).result() == 200
    assert observed == [warmup_thread, warmup_thread]
