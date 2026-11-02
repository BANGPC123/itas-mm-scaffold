"""Display relaxation must leave canonical UFLD geometry and precision intact."""
from types import SimpleNamespace

import numpy as np
import torch

from src.perception.ufld_v2 import UFLDv2
from src.utils.types import LaneInfo
from tests.unit.test_lane_simple_20261102 import VALID, _pipeline, _run


def _detector(row_count):
    detector = UFLDv2.__new__(UFLDv2)
    detector.device = 'cpu'
    detector.dtype = torch.float32
    detector.mean = torch.zeros(3, 1, 1)
    detector.std = torch.ones(3, 1, 1)
    row = torch.zeros(1, 200, 72, 4)
    col = torch.zeros(1, 100, 81, 4)
    row[:, 80] = 10
    exists_row = torch.zeros(1, 2, 72, 4)
    exists_col = torch.zeros(1, 2, 81, 4)
    exists_row[:, 1, torch.linspace(0, 71, row_count).long(), 1] = 10
    calls = []
    detector.model = lambda x: (calls.append(1) or [row, col, exists_row, exists_col])
    return detector, calls


def test_none_preserves_canonical_polylines_and_has_no_relaxed_lines():
    detector, calls = _detector(40)
    image = np.random.default_rng(3).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    original = detector.detect(image)
    explicit = detector.detect(image, display_row_ratio=None)
    assert original.polylines and original.polylines == explicit.polylines
    assert original.lane_lines == explicit.lane_lines
    assert explicit.relaxed_polylines == []
    assert len(calls) == 2


def test_quarter_ratio_keeps_twenty_rows_from_one_forward_only():
    detector, calls = _detector(20)
    image = np.random.default_rng(3).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    lane = detector.detect(image, display_row_ratio=.25)
    assert lane.polylines == []
    assert lane.lane_lines == []
    assert len(lane.relaxed_polylines) == 1
    assert len(lane.relaxed_polylines[0]) == 20
    assert calls == [1]


def test_half_ratio_matches_canonical_lines():
    detector, calls = _detector(40)
    image = np.random.default_rng(3).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    lane = detector.detect(image, display_row_ratio=.5)
    assert lane.relaxed_polylines == lane.polylines
    assert calls == [1]


def test_precision_ignores_relaxed_geometry(monkeypatch):
    import src.perception.ufld_display_screen as screen
    monkeypatch.setattr(screen, 'screen_ufld_for_demo', lambda image, lane: (lane, None))
    pipe = _pipeline(lambda: LaneInfo([], backend='ufld_v2'))
    def detect(image, **kwargs):
        assert 'display_row_ratio' not in kwargs
        return LaneInfo([], polylines=[VALID], relaxed_polylines=[[(x+100, y) for x, y in VALID]], backend='ufld_v2')
    pipe.lane_detector = SimpleNamespace(detect=detect)
    pipe.camera_lane_visual.detect = lambda image, lane: lane
    lane = _run(pipe, lane_simple=False).perception.lane
    assert lane.display_polylines == [VALID]


def test_simple_uses_relaxed_geometry_without_changing_camera_input(monkeypatch):
    import src.pipeline.orchestrator as orchestrator
    original_load = orchestrator.load_config
    monkeypatch.setattr(orchestrator, 'load_config', lambda name: (
        {**original_load(name), 'offline_video_lane': {'display': 'simple', 'display_row_ratio': .25}}))
    pipe = _pipeline(lambda: LaneInfo([], backend='ufld_v2'))
    def detect(image, **kwargs):
        assert kwargs == {'display_row_ratio': .25}
        return LaneInfo([], polylines=[], relaxed_polylines=[VALID], backend='ufld_v2')
    def camera(image, lane):
        assert lane.polylines == []
        return LaneInfo([], camera_view='forward')
    pipe.lane_detector = SimpleNamespace(detect=detect)
    pipe.camera_lane_visual.detect = camera
    lane = _run(pipe, lane_simple=True).perception.lane
    assert lane.polylines == []
    assert lane.display_polylines == [VALID]
    assert lane.camera_view == 'forward'
