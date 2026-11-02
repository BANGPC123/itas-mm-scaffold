"""Simplified offline lane display: UFLD polylines only, no extra overlays or ego pair."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import scripts.render_offline_video_max_20261014 as renderer
import src.perception.ufld_display_screen as screen
from src.perception.lane_display_quality import visual_polylines
from src.pipeline.orchestrator import Pipeline
from src.utils.types import LaneInfo

SHAPE = (720, 1280, 3)
VALID = [(250 + i * 8, 100 + i * 25) for i in range(13)]
EGO = [[(400, 700), (420, 400), (440, 300)]]


class _Stub:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


def _pipeline(lane_factory, camera=None):
    pipe = Pipeline.__new__(Pipeline)
    pipe.sign_detector = _Stub(detect=lambda road: [], _model=object())
    pipe.sign_recognizer = _Stub(recognize=lambda road, signs, **kw: signs, model=object())
    pipe.lane_detector = _Stub(detect=lambda image: lane_factory())
    pipe.camera_lane_visual = _Stub(
        detect=camera or (lambda image, lane: LaneInfo([], backend='camera_road_paint_visual', camera_view='forward')))
    pipe.zone_classifier = _Stub(classify=lambda lat, lon: SimpleNamespace(zone_label='Urban'))
    pipe.applicability = _Stub(apply=lambda signs, lane, shape, vehicle, observed_at=None: signs)
    pipe.current_vehicle = 'car'
    pipe.offline_legal = _Stub(for_signs=lambda signs, zone: SimpleNamespace(
        references=[], speech_text='', observation_text='', observation_speech_text='', car_summary=''))
    pipe.sign_reference = _Stub(lookup=lambda labels: [])
    pipe.reasoning_cfg = {}
    pipe.rag_chain = None
    pipe.auto_supplementary_enabled = False
    pipe.tts_engine = None
    pipe._reused_lane = {}
    pipe._verified_zone_streams = {}
    return pipe


def _run(pipe, lane_simple):
    image = np.zeros(SHAPE, np.uint8)
    return pipe.run(image, 10.7769, 106.7009, False, annotate=False,
                    lane_visual_roi=True, lane_simple=lane_simple)


def test_simple_mode_produces_no_ego_boundaries():
    pipe = _pipeline(lambda: LaneInfo([], polylines=[VALID], ego_boundaries=EGO, backend='ufld_v2'))
    lane = _run(pipe, lane_simple=True).perception.lane
    assert lane.ego_boundaries == []
    assert lane.lane_center_offset_px is None
    assert lane.candidate_polylines == []
    assert lane.display_polylines == [VALID]


def test_precision_mode_still_runs_ufld_screen(monkeypatch):
    calls = []
    monkeypatch.setattr(screen, 'screen_ufld_for_demo',
                        lambda image, lane: (calls.append(1) or (lane, None)))
    pipe = _pipeline(lambda: LaneInfo([], polylines=[VALID], ego_boundaries=EGO, backend='ufld_v2'),
                     camera=lambda image, lane: lane)
    _run(pipe, lane_simple=False)
    assert calls == [1]


def test_simple_mode_skips_ufld_screen(monkeypatch):
    calls = []
    monkeypatch.setattr(screen, 'screen_ufld_for_demo',
                        lambda image, lane: (calls.append(1) or (lane, None)))
    pipe = _pipeline(lambda: LaneInfo([], polylines=[VALID], ego_boundaries=EGO, backend='ufld_v2'))
    _run(pipe, lane_simple=True)
    assert calls == []


def test_candidate_display_drops_nan_out_of_image_and_short_lines():
    nan_line = [(100, 100), (float('nan'), 200), (120, 300), (140, 400)]
    outside = [(100, 100), (2000, 200), (120, 300), (140, 400)]
    short = [(100, 100), (120, 300)]
    lane = LaneInfo([], polylines=[VALID, nan_line, outside, short])
    assert visual_polylines(lane, SHAPE, mode='candidate') == [VALID]
    assert len(lane.polylines) == 4


class _FakePipe:
    def __init__(self):
        self.kwargs = []

    def run(self, image, latitude, longitude, synthesize_audio=True, **kwargs):
        self.kwargs.append(kwargs)
        return SimpleNamespace(
            perception=SimpleNamespace(signs=[], lane=LaneInfo([]),
                                       supplementary_candidates=[], auxiliary_sign_candidates=[]),
            reasoning=SimpleNamespace(speech_text='', car_summary=''),
            timings_ms={}, warnings=[])


def _tiny_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (320, 240))
    for i in range(20):
        writer.write(np.full((240, 320, 3), i * 5, np.uint8))
    writer.release()


needs_windows_font = pytest.mark.skipif(
    not Path('C:/Windows/Fonts/arial.ttf').is_file(), reason='renderer uses Arial')


@needs_windows_font
def test_precision_config_passes_old_lane_path_to_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, 'load_config',
                        lambda name: {'offline_video_lane': {'display': 'precision'}})
    source = tmp_path / 'clip.mp4'
    _tiny_video(source)
    fake = _FakePipe()
    summary = renderer.render(source, tmp_path / 'out', pipeline=fake)
    assert fake.kwargs and all(k['lane_simple'] is False for k in fake.kwargs)
    assert summary['offline_video_lane_display'] == 'precision'
    assert json.loads((tmp_path / 'out/summary.json').read_text(encoding='utf8'))['verified_full_decode']


@needs_windows_font
def test_repository_config_defaults_to_simple_lane_path(tmp_path):
    source = tmp_path / 'clip.mp4'
    _tiny_video(source)
    fake = _FakePipe()
    summary = renderer.render(source, tmp_path / 'out', pipeline=fake)
    assert all(k['lane_simple'] is True for k in fake.kwargs)
    assert summary['offline_video_lane_display'] == 'simple'


def test_overlay_status_text_depends_on_display_mode():
    from scripts.complete_test_video import lane_status_message
    assert lane_status_message('forward', lane_simple=True) == 'Làn: ứng viên UFLD-v2, chỉ để tham khảo, có thể sai.'
    assert lane_status_message('forward', lane_simple=False).startswith('Vạch sơn quan sát được')
    assert lane_status_message('side_or_cabin', lane_simple=True).startswith('Camera quay ngang')
    assert lane_status_message('side_or_cabin', lane_simple=False).startswith('Camera quay ngang')


def test_unknown_display_value_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(renderer, 'load_config',
                        lambda name: {'offline_video_lane': {'display': 'bogus'}})
    source = tmp_path / 'clip.mp4'
    _tiny_video(source)
    with pytest.raises(ValueError):
        renderer.render(source, tmp_path / 'out', pipeline=_FakePipe())
