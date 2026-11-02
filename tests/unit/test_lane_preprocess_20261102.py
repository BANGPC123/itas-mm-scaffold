"""Lane-only preprocessing: pure transforms, and display-only use of a second UFLD forward."""
from types import SimpleNamespace

import numpy as np
import pytest

import src.perception.ufld_display_screen as screen
import src.pipeline.orchestrator as orchestrator
from src.perception.lane_preprocess import preprocess_for_lane
from src.utils.types import LaneInfo
from tests.unit.test_lane_simple_20261102 import SHAPE, VALID, _pipeline, _Stub

SHIFTED = [(x + 100, y) for x, y in VALID]


def _frame():
    return np.random.default_rng(7).integers(0, 256, SHAPE, dtype=np.uint8)


def test_none_returns_the_original_frame():
    image = _frame()
    assert preprocess_for_lane(image, 'none') is image


@pytest.mark.parametrize('mode,kwargs', [('gamma', {'gamma': .8}), ('gamma', {'gamma': .6}),
                                         ('clahe', {'clahe_clip': 2., 'clahe_tile': 8}),
                                         ('clahe', {'clahe_clip': 3., 'clahe_tile': 8})])
def test_transforms_keep_shape_dtype_and_range(mode, kwargs):
    image = _frame()
    out = preprocess_for_lane(image, mode, **kwargs)
    assert out.shape == image.shape
    assert out.dtype == image.dtype
    assert out.min() >= 0 and out.max() <= 255


def test_gamma_below_one_brightens_and_zero_stays_zero():
    dark = np.full(SHAPE, 64, np.uint8)
    out = preprocess_for_lane(dark, 'gamma', gamma=.8)
    assert out.mean() > dark.mean()
    assert preprocess_for_lane(np.zeros(SHAPE, np.uint8), 'gamma', gamma=.8).max() == 0


def test_unknown_mode_and_bad_gamma_are_rejected():
    with pytest.raises(ValueError):
        preprocess_for_lane(_frame(), 'sharpen')
    with pytest.raises(ValueError):
        preprocess_for_lane(_frame(), 'gamma', gamma=0)


def _run_lane(monkeypatch, image, preprocess, *, lane_simple=True):
    original_load = orchestrator.load_config

    def load(name):
        config = original_load(name)
        if name != 'perception':
            return config
        return {**config, 'offline_video_lane': {**config['offline_video_lane'], 'lane_preprocess': preprocess}}

    monkeypatch.setattr(orchestrator, 'load_config', load)
    pipe = _pipeline(lambda: LaneInfo([], polylines=[VALID], backend='ufld_v2'))
    calls = []

    def detect(img, **kwargs):
        calls.append((img is image, kwargs))
        if img is image:
            return LaneInfo([], polylines=[VALID], relaxed_polylines=[VALID], backend='ufld_v2')
        return LaneInfo([], polylines=[], relaxed_polylines=[SHIFTED], backend='ufld_v2')

    seen_camera = []

    def camera(img, lane):
        if not lane_simple:
            return lane
        seen_camera.append(list(lane.polylines))
        return LaneInfo([], camera_view='forward')

    pipe.lane_detector = _Stub(detect=detect)
    pipe.camera_lane_visual.detect = camera
    monkeypatch.setattr(screen, 'screen_ufld_for_demo', lambda img, lane: (lane, None))
    lane = pipe.run(image, 10.7769, 106.7009, False, annotate=False,
                    lane_visual_roi=True, lane_simple=lane_simple).perception.lane
    return lane, calls, seen_camera


def test_preprocess_display_comes_from_second_forward_and_camera_from_original(monkeypatch):
    image = np.full(SHAPE, 90, np.uint8)
    lane, calls, seen_camera = _run_lane(monkeypatch, image, 'gamma')
    assert [same for same, _ in calls] == [True, False]
    assert all(kwargs == {'display_row_ratio': .5} for _, kwargs in calls)
    assert seen_camera == [[VALID]]
    assert lane.display_polylines == [SHIFTED]
    assert lane.camera_view == 'forward'


def test_preprocess_none_uses_single_forward(monkeypatch):
    image = np.full(SHAPE, 90, np.uint8)
    lane, calls, _ = _run_lane(monkeypatch, image, 'none')
    assert [same for same, _ in calls] == [True]
    assert lane.display_polylines == [VALID]


def test_precision_never_runs_preprocess(monkeypatch):
    image = np.full(SHAPE, 90, np.uint8)
    lane, calls, _ = _run_lane(monkeypatch, image, 'clahe', lane_simple=False)
    assert [same for same, _ in calls] == [True]
    assert all('display_row_ratio' not in kwargs for _, kwargs in calls)
    assert lane.display_polylines == [VALID]
