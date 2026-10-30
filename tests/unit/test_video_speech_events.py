from src.interaction.ai_observation import video_speech_events
from src.utils.types import SignDetection

BOX = (50, 50, 110, 110)


def sign(label, track_id, source='temporal_best_crop', recognition=.95):
    s = SignDetection(label, .9, BOX, recognition_confidence=recognition)
    s.recognition_source = source
    s.track_id = track_id
    return s


def test_each_track_is_cued_once():
    announced = set()
    first = video_speech_events([sign('P.131a', 7)], 2.0, announced)
    again = video_speech_events([sign('P.131a', 7)], 2.5, announced)
    assert len(first) == 1 and first[0]['t_seconds'] == 2.0 and first[0]['codes'] == ['P.131a']
    assert 'P.131a' in first[0]['text'] and again == []


def test_untracked_or_unsupported_signs_are_not_cued():
    announced = set()
    assert video_speech_events([sign('P.131a', None)], 1.0, announced) == []
    assert video_speech_events([sign('P.131a', 8, source='frame_only')], 1.0, announced) == []
    assert video_speech_events([sign('unknown', 9)], 1.0, announced) == []
    assert announced == set()


def test_low_recognition_confidence_is_not_cued():
    assert video_speech_events([sign('P.131a', 3, recognition=.5)], 1.0, set()) == []
