"""M2 unit tests: zone geometry, TTC proxy, hysteresis, cooldown, speech priority. No model needed."""
from src.perception.forward_objects import (
    ForwardObjectWarner, ForwardWarnConfig, in_zone, pick_speech, speech_text, ttc_proxy)

W, H = 1280, 720


def _cfg(**kw):
    c = ForwardWarnConfig(enabled=True)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_zone_contains_bottom_centre_and_excludes_top_and_corners():
    c = _cfg()
    assert in_zone((W / 2, H * 0.95), W, H, c)
    assert not in_zone((W / 2, H * 0.3), W, H, c)            # above the top edge
    assert not in_zone((W * 0.05, H * 0.95), W, H, c)        # bottom-left corner, outside base
    assert not in_zone((W * 0.95, H * 0.95), W, H, c)


def test_ttc_proxy_none_when_not_growing_and_small_when_growing_fast():
    assert ttc_proxy([100, 100, 100, 100], k=3, dt_s=1 / 6, min_scale=1.03) is None
    assert ttc_proxy([100, 90, 80, 70], k=3, dt_s=1 / 6, min_scale=1.03) is None
    growing = [100, 120, 144, 172.8]                          # s = 1.728 over k=3
    v = ttc_proxy(growing, k=3, dt_s=1 / 6, min_scale=1.03)
    assert v is not None and abs(v - 0.5 / 0.728) < 1e-6


def _det(tid, y2, h, x_c=W / 2):
    return {'track_id': tid, 'cls': 2, 'conf': .9, 'bbox_xyxy': [x_c - 40, y2 - h, x_c + 40, y2]}


def test_chu_y_needs_three_consecutive_samples_in_zone_near_bottom():
    w = ForwardObjectWarner(_cfg(), analysis_hz=6)
    out = []
    for i in range(3):
        out += w.update(i / 6, W, H, [_det(1, 0.95 * H, 100)])
    assert [a['level'] for a in out] == ['chu_y']


def test_no_alert_for_single_frame_or_out_of_zone():
    w = ForwardObjectWarner(_cfg(), analysis_hz=6)
    assert w.update(0.0, W, H, [_det(1, 0.95 * H, 100)]) == []
    w2 = ForwardObjectWarner(_cfg(), analysis_hz=6)
    for i in range(5):
        assert w2.update(i / 6, W, H, [_det(2, 0.6 * H, 100)]) == []  # not near bottom


def test_nguy_co_after_two_samples_of_fast_growth():
    w = ForwardObjectWarner(_cfg(), analysis_hz=6)
    heights = [60, 75, 95, 120, 150, 190]
    levels = []
    for i, h in enumerate(heights):
        for a in w.update(i / 6, W, H, [_det(3, 0.95 * H, h)]):
            levels.append(a['level'])
    assert 'nguy_co' in levels


def test_track_cooldown_suppresses_repeat_alerts_within_6s():
    w = ForwardObjectWarner(_cfg(cooldown_global_s=0.0), analysis_hz=6)
    alerts = []
    for i in range(30):   # 5 s of a stationary close car
        alerts += w.update(i / 6, W, H, [_det(4, 0.95 * H, 100)])
    assert len(alerts) == 1


def test_missing_track_id_and_disabled_config_are_safe():
    w = ForwardObjectWarner(_cfg(), analysis_hz=6)
    assert w.update(0.0, W, H, [{'track_id': None, 'cls': 2, 'bbox_xyxy': [0, 0, 10, 10], 'conf': .9}]) == []
    off = ForwardObjectWarner(ForwardWarnConfig(enabled=False), analysis_hz=6)
    assert off.update(0.0, W, H, [_det(1, 0.95 * H, 100)]) == []


def test_speech_priority_nguy_over_sign_and_chu_y_yields_to_sign():
    nguy = {'level': 'nguy_co', 'cls': 'car'}
    chu = {'level': 'chu_y', 'cls': 'motorcycle'}
    assert pick_speech([chu, nguy], sign_cues_present=True) == nguy
    assert pick_speech([chu], sign_cues_present=True) is None
    assert pick_speech([chu], sign_cues_present=False) == chu
    assert pick_speech([], sign_cues_present=False) is None


def test_speech_wording_never_claims_collision():
    for level in ('chu_y', 'nguy_co'):
        for cls in ('car', 'person', 'motorcycle', 'truck'):
            text = speech_text(level, cls).lower()
            assert 'va chạm' not in text and 'đâm' not in text


def test_merge_pushes_overlapping_sign_cues_apart_without_shared_start():
    from src.perception.forward_objects import merge_cues
    signs = [{'t_seconds': 10.0, 'text': 'a', 'codes': ['P.1']}, {'t_seconds': 10.5, 'text': 'b', 'codes': ['P.2']}]
    fw = [{'t_seconds': 9.5, 'text': 'x', 'codes': [], 'kind': 'forward_nguy_co'}]
    out = merge_cues(signs, fw, busy_s=2.0)
    starts = [c['t_seconds'] for c in out if 'kind' not in c]
    assert len(set(starts)) == len(starts)
    assert min(starts) >= 11.5
    assert [c['text'] for c in out if 'kind' not in c] == ['a', 'b']


def test_draw_forward_rect_blend_matches_full_frame_reference():
    """M8.1: blending only inside the zone rectangle gives the same pixels as the former full-frame blend."""
    import cv2
    import numpy as np
    from src.perception.forward_objects import ForwardWarnConfig, draw_forward
    cfg = ForwardWarnConfig(enabled=True)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(144, 256, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    zone = np.array([[int(w * (0.5 - cfg.bottom_width_ratio / 2)), h - 1],
                     [int(w * (0.5 - cfg.top_width_ratio / 2)), int(cfg.top_y_ratio * h)],
                     [int(w * (0.5 + cfg.top_width_ratio / 2)), int(cfg.top_y_ratio * h)],
                     [int(w * (0.5 + cfg.bottom_width_ratio / 2)), h - 1]], np.int32)
    ref = frame.copy()
    overlay = ref.copy()
    cv2.fillPoly(overlay, [zone], (255, 255, 0))
    ref[:] = cv2.addWeighted(overlay, 0.12, ref, 0.88, 0)
    out = draw_forward(frame.copy(), [], cfg, None)
    x0, y0 = int(zone[:, 0].min()), int(zone[:, 1].min())
    # Inside the bounding rectangle: identical to the reference blend.
    assert np.array_equal(out[y0:, x0:int(zone[:, 0].max()) + 1], ref[y0:, x0:int(zone[:, 0].max()) + 1])
    # Outside the rectangle: untouched.
    assert np.array_equal(out[:y0], frame[:y0])
    assert np.array_equal(out[:, :x0], frame[:, :x0])
