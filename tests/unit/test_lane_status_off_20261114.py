"""M12: with the lane stage off, no lane status line is drawn; with it on, the old line and band are kept."""
import numpy as np
from PIL import ImageFont

from scripts.complete_test_video import draw_frame, lane_status_message
from src.perception.video_text_overlay import VideoTextOverlay
from src.perception.video_text_overlay_safe import SafeVideoTextOverlay

H, W = 720, 1280
DARK_BGR = (34, 23, 15)


def _font():
    try:
        return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
    except OSError:
        return ImageFont.load_default()


def _payload(backend, view='forward'):
    return {'signs': [], 'lane': {'backend': backend, 'camera_view': view}}


def test_lane_status_message_off_returns_nothing_and_on_keeps_old_text():
    assert lane_status_message('forward', False, lane_enabled=False) is None
    assert lane_status_message('side_or_cabin', True, lane_enabled=False) is None
    assert lane_status_message('forward', False) == 'Vạch sơn quan sát được; không nối nét qua xe hoặc vùng thiếu bằng chứng.'
    assert lane_status_message('forward', True) == 'Làn: ứng viên UFLD-v2, chỉ để tham khảo, có thể sai.'
    assert lane_status_message('side_or_cabin', False).startswith('Camera quay ngang')


def test_render_frame_off_has_no_lane_line_and_no_empty_row():
    font = _font()
    off = draw_frame(np.zeros((H, W, 3), np.uint8), _payload('disabled', 'unknown'), [], [], 0.0, font, 30, 6)
    # Rows 30-56 (where the lane line used to be) stay untouched: no band, no text.
    assert not off[30:57].any()
    assert tuple(off[10, 1200]) == DARK_BGR          # the header band itself is still drawn (rows 0-29)
    on = draw_frame(np.zeros((H, W, 3), np.uint8), _payload('ufld_v2', 'forward'), [], [], 0.0, font, 30, 6)
    assert tuple(on[45, 1200]) == DARK_BGR            # lane band is 57 px tall
    assert on[30:57].any()                            # the lane line text is there


def test_video_text_overlay_off_and_on():
    font = _font()
    ov = VideoTextOverlay(font, 30, 6)
    off = ov.draw(np.zeros((H, W, 3), np.uint8), _payload('disabled'), [], [], 0.0)
    assert not off[30:57].any()
    on = ov.draw(np.zeros((H, W, 3), np.uint8), _payload('ufld_v2'), [], [], 0.0)
    assert tuple(on[45, 1200]) == DARK_BGR and on[30:57].any()


def test_safe_overlay_off_leaves_second_row_empty_and_on_draws_it():
    font = _font()
    ov = SafeVideoTextOverlay(font, 30, 6)
    off = ov.draw(np.zeros((H, W, 3), np.uint8), _payload('disabled'), [], [], 0.0)
    assert not off[30:55, 10:300].any()
    on = ov.draw(np.zeros((H, W, 3), np.uint8), _payload('ufld_v2'), [], [], 0.0)
    assert on[30:55, 10:300].any()
