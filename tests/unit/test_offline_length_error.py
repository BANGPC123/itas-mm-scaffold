from backend.routers.offline_video import MAX_VIDEO_FRAMES, length_error


def test_video_within_frame_limit_is_accepted():
    assert length_error(MAX_VIDEO_FRAMES, 30.0) is None


def test_video_over_frame_limit_asks_for_trimming():
    message = length_error(MAX_VIDEO_FRAMES + 1, 30.0)
    assert message is not None
    assert 'cắt ngắn' in message and str(MAX_VIDEO_FRAMES) in message
