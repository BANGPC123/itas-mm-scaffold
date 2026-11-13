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


def test_m9_speech_texts_are_short_and_fixed():
    from src.perception.forward_objects import prewarm_texts, speech_text
    assert speech_text('chu_y', 'motorcycle') == 'Chú ý xe máy'
    assert speech_text('chu_y', 'person') == 'Chú ý người đi bộ'
    texts = prewarm_texts()
    assert 'Chú ý xe máy' in texts
    assert all(len(t.split()) <= 9 for t in texts)


def test_m13_nguy_co_wording_follows_class_label():
    """M13.2: riders (person, bicycle, motorcycle) are never announced as a car in front."""
    from src.perception.forward_objects import prewarm_texts, speech_text
    assert speech_text('nguy_co', 'person') == 'Cẩn thận, người đi bộ phía trước'
    assert speech_text('nguy_co', 'motorcycle') == 'Cẩn thận, xe máy phía trước'
    assert speech_text('nguy_co', 'bicycle') == 'Cẩn thận, xe đạp phía trước'
    assert speech_text('nguy_co', 'car') == 'Cẩn thận, ô tô phía trước'
    texts = prewarm_texts()
    assert speech_text('nguy_co', 'motorcycle') in texts and speech_text('chu_y', 'bicycle') in texts
    assert not any('xe phía trước' in t for t in texts)
    assert not any(w in t for t in texts for w in ('va chạm', 'sắp đâm'))


def test_m9_voice_gate_cooldowns():
    from src.perception.forward_objects import VoiceGate
    cfg = ForwardWarnConfig(enabled=True, voice_cooldown_s=8.0, voice_class_cooldown_s=15.0)
    g = VoiceGate(cfg)
    assert g.decide('chu_y', 'car', 10.0) is True          # first chu_y spoken
    assert g.decide('chu_y', 'bus', 12.0) is False         # global 8 s cooldown
    assert g.decide('chu_y', 'car', 19.0) is False         # global ok, class 15 s not elapsed
    assert g.decide('chu_y', 'bus', 19.0) is True          # other class after global cooldown
    assert g.decide('chu_y', 'car', 26.5) is False         # global: only 7.5 s since the bus cue at 19 s
    assert g.decide('chu_y', 'car', 27.5) is True          # global 8.5 s, class 17.5 s: both elapsed
    assert g.decide('nguy_co', 'motorcycle', 27.0) is True  # nguy_co always spoken


def test_m9_forward_cues_one_chu_y_per_sample_and_flags():
    from src.perception.forward_objects import VoiceGate, forward_cues
    cfg = ForwardWarnConfig(enabled=True)
    gate = VoiceGate(cfg)
    alerts = [{'level': 'chu_y', 'cls': 'car'}, {'level': 'chu_y', 'cls': 'bus'}, {'level': 'nguy_co', 'cls': 'truck'}]
    cues = forward_cues(alerts, False, 5.0, gate=gate)
    kinds = [c['kind'] for c in cues]
    # M13.3: nguy_co is spoken and starts the voice cooldown, so no chu_y cue talks over it in the same sample.
    assert kinds == ['forward_nguy_co']
    assert [a['spoken'] for a in alerts] == [False, False, True]
    assert alerts[0]['speech_text'] is None
    assert forward_cues([{'level': 'chu_y', 'cls': 'car'}], False, 5.5, gate=VoiceGate(cfg))[0]['text'] == 'Chú ý ô tô'
    # Sign cue spoken in the same sample: chu_y is not spoken and the gate is not consulted.
    alerts2 = [{'level': 'chu_y', 'cls': 'motorcycle'}]
    assert forward_cues(alerts2, True, 30.0, gate=gate) == []
    assert alerts2[0]['spoken'] is False


def test_m13_canvas_box_maps_1920x1080_to_1280x720():
    from src.perception.forward_objects import canvas_box
    assert canvas_box([960, 540, 1920, 1080], (1920, 1080), (1280, 720)) == [640, 360, 1280, 720]
    got = canvas_box([0, 0, 100, 50], (1920, 1080), (1280, 720))
    assert [round(v, 3) for v in got] == [0.0, 0.0, 66.667, 33.333]


def test_m13_draw_forward_scales_source_box_to_canvas():
    """M13.1: a box measured on a 1920x1080 frame is drawn where it sits on the 1280x720 canvas."""
    import numpy as np
    from src.perception.forward_objects import ForwardWarnConfig, draw_forward
    cfg = ForwardWarnConfig(enabled=True)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    alert = {'level': 'chu_y', 'label_vi': 'xe máy', 'bbox_xyxy': [900.0, 500.0, 1100.0, 700.0]}
    out = draw_forward(canvas.copy(), [alert], cfg, None, src_wh=(1920, 1080))
    # Scaled box: x 600..733, y 333..467. BGR of chu_y colour (255,140,0) is (0,140,255).
    assert tuple(out[333, 600]) == (0, 140, 255) or tuple(out[334, 601]) == (0, 140, 255)
    assert tuple(out[400, 733]) == (0, 140, 255) or tuple(out[400, 732]) == (0, 140, 255)
    # The unscaled position (900..1100 px on the canvas) must not carry the box border.
    assert not np.any(np.all(out[500:700, 900:1100] == (0, 140, 255), axis=-1))


def test_m13_one_nguy_co_cue_per_sample_most_urgent_first():
    """M13.3: two nguy_co alerts in the same sample give one spoken cue (smallest TTC proxy)."""
    from src.perception.forward_objects import VoiceGate, forward_cues
    cfg = ForwardWarnConfig(enabled=True)
    alerts = [{'level': 'nguy_co', 'cls': 'car', 'ttc_proxy_s': 1.9},
              {'level': 'nguy_co', 'cls': 'person', 'ttc_proxy_s': 0.9},
              {'level': 'nguy_co', 'cls': 'truck', 'ttc_proxy_s': None}]
    cues = forward_cues(alerts, False, 63.333, gate=VoiceGate(cfg))
    assert [c['kind'] for c in cues] == ['forward_nguy_co']
    assert [a['spoken'] for a in alerts] == [False, True, False]
    assert cues[0]['text'] == 'Cẩn thận, người đi bộ phía trước'
    # Without a gate the same rule holds.
    alerts2 = [{'level': 'nguy_co', 'cls': 'car', 'ttc_proxy_s': 1.0},
               {'level': 'nguy_co', 'cls': 'car', 'ttc_proxy_s': 1.5}]
    assert len(forward_cues(alerts2, False, 10.0)) == 1


def test_m13_nguy_co_cues_never_overlap_in_time():
    from src.perception.forward_objects import VoiceGate
    g = VoiceGate(ForwardWarnConfig(enabled=True))
    assert g.decide('nguy_co', 'car', 10.0) is True
    assert g.decide('nguy_co', 'car', 10.33) is False   # next detection sample, within 2 s
    assert g.decide('nguy_co', 'person', 12.0) is True  # 2.0 s later


def _box(tid, cls, x1, y1, x2, y2):
    return {'track_id': tid, 'cls': cls, 'bbox_xyxy': [float(x1), float(y1), float(x2), float(y2)], 'conf': 0.9}


def test_m13_5_rider_merge_drops_person_on_motorcycle_only():
    from src.perception.forward_objects import merge_riders
    moto = _box(1, 3, 600, 500, 760, 760)
    rider = _box(2, 0, 640, 430, 720, 640)      # bottom-centre (680, 640) inside the motorcycle box
    walker = _box(3, 0, 100, 400, 180, 700)     # far from the motorcycle
    out = merge_riders([moto, rider, walker])
    assert [d['track_id'] for d in out] == [1, 3]
    assert merge_riders([rider, walker]) == [rider, walker]   # no motorcycle: nothing merged


def test_m13_5_nguy_zone_needs_narrow_zone_and_no_edge_contact():
    """Flag on: a box inside the wide zone but outside the narrow zone gives chu_y at most."""
    W, H = 1280, 720
    cfg = _cfg(nguy_zone_enabled=True, nguy_persist=1, nguy_co_persist=1, ttc_k=1, cooldown_track_s=0,
               cooldown_global_s=0, chu_y_persist=99)
    def run(x1, x2, y2, grow=True):
        w = ForwardObjectWarner(cfg, 3.0)
        h0 = 80.0
        out = []
        for k in range(3):
            hh = h0 * (1.25 ** k) if grow else h0
            out += w.update(k / 3.0, W, H, [_box(7, 2, x1, y2 - hh, x2, y2)])
        return out
    # Bottom centre x=640 (middle), lower edge at y=600: inside both zones -> nguy_co
    assert any(a['level'] == 'nguy_co' for a in run(600, 680, 600))
    # Bottom centre x=840: inside the wide zone (half width ~239 px at y=600), outside the narrow one (~173 px)
    assert not any(a['level'] == 'nguy_co' for a in run(800, 880, 600))
    # Box touching the right frame edge -> no nguy_co
    assert not any(a['level'] == 'nguy_co' for a in run(1200, 1279, 600))
    off = _cfg(nguy_zone_enabled=False, nguy_persist=1, nguy_co_persist=1, ttc_k=1, cooldown_track_s=0,
               cooldown_global_s=0, chu_y_persist=99)
    w = ForwardObjectWarner(off, 3.0)
    res = []
    for k in range(3):
        res += w.update(k / 3.0, W, H, [_box(7, 2, 800, 600 - 80 * 1.25 ** k, 880, 600)])
    assert any(a['level'] == 'nguy_co' for a in res)   # without the flag the wide zone is enough


def test_m13_5_lateral_slide_blocks_alert():
    """A growing box that also slides sideways is not an approaching object: no nguy_co."""
    W, H = 1280, 720
    cfg = _cfg(lateral_filter_enabled=True, lateral_max_per_h=0.25, ttc_k=1, nguy_co_persist=1,
               nguy_co_ttc_s=5.0, chu_y_persist=99, cooldown_track_s=0, cooldown_global_s=0)
    def run(slide):
        w = ForwardObjectWarner(cfg, 3.0)
        out = []
        for k in range(3):
            hh = 80.0 * 1.3 ** k
            x0 = 600 + slide * k
            out += w.update(k / 3.0, W, H, [_box(5, 2, x0, 600 - hh, x0 + 100, 600)])
        return [a['level'] for a in out]
    assert 'nguy_co' not in run(slide=60)     # centre moves ~0.5 box heights per sample
    assert 'nguy_co' in run(slide=0)          # same growth, no sideways motion: alert


def test_m13_5_min_age_and_class_reset_flags():
    W, H = 1280, 720
    base = dict(chu_y_persist=1, cooldown_track_s=0, cooldown_global_s=0)
    cfg_age = _cfg(track_min_age_samples=3, **base)
    w = ForwardObjectWarner(cfg_age, 3.0)
    got = []
    for k in range(4):
        got.append(len(w.update(k / 3.0, W, H, [_box(9, 2, 600, 560, 700, 640)])))
    assert got[:2] == [0, 0] and got[2] > 0      # third sample on the same track is the first alert
    cfg_cls = _cfg(track_class_reset_enabled=True, track_min_age_samples=3, **base)
    w2 = ForwardObjectWarner(cfg_cls, 3.0)
    for k in range(3):
        w2.update(k / 3.0, W, H, [_box(4, 0, 600, 560, 700, 640)])   # person, age 3
    assert w2.tracks[4].age == 3
    w2.update(1.0, W, H, [_box(4, 2, 600, 560, 700, 640)])          # class changes: history restarts
    assert w2.tracks[4].age == 1
    cfg_nocls = _cfg(track_class_reset_enabled=False, **base)
    w3 = ForwardObjectWarner(cfg_nocls, 3.0)
    for k, c in enumerate([0, 0, 0, 2]):
        w3.update(k / 3.0, W, H, [_box(4, c, 600, 560, 700, 640)])
    assert w3.tracks[4].age == 4


def test_m13_5_nguy_co_switch_turns_off_only_nguy():
    W, H = 1280, 720
    cfg = _cfg(nguy_co_enabled=False, nguy_persist=1, nguy_co_persist=1, ttc_k=1, cooldown_track_s=0,
               cooldown_global_s=0, chu_y_persist=2)
    w = ForwardObjectWarner(cfg, 3.0)
    levels = []
    for k in range(4):
        hh = 80.0 * 1.3 ** k
        levels += [a['level'] for a in w.update(k / 3.0, W, H, [_box(3, 2, 600, 600 - hh, 700, 600)])]
    assert 'nguy_co' not in levels and 'chu_y' in levels
