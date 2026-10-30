from src.interaction.ai_observation import group_speech_events, video_speech_events
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


def test_cues_within_a_second_and_a_half_are_read_together():
    events = [
        {'t_seconds': 3.167, 'codes': ['P.112'], 'text': 'a'},
        {'t_seconds': 3.667, 'codes': ['W.245a'], 'text': 'b'},
    ]
    grouped = group_speech_events(events)
    assert len(grouped) == 1
    assert grouped[0]['t_seconds'] == 3.167 and grouped[0]['codes'] == ['P.112', 'W.245a']
    assert 'P.112' in grouped[0]['text'] and 'W.245a' in grouped[0]['text']


def test_third_sign_in_a_burst_gets_its_own_cue_after_the_burst():
    events = [
        {'t_seconds': 3.0, 'codes': ['P.112'], 'text': 'a'},
        {'t_seconds': 3.5, 'codes': ['W.245a'], 'text': 'b'},
        {'t_seconds': 4.0, 'codes': ['P.117'], 'text': 'c'},
    ]
    grouped = group_speech_events(events)
    assert [g['codes'] for g in grouped] == [['P.112', 'W.245a'], ['P.117']]
    assert grouped[1]['t_seconds'] == 4.0


def test_separate_bursts_stay_separate():
    events = [
        {'t_seconds': 3.0, 'codes': ['P.112'], 'text': 'a'},
        {'t_seconds': 13.0, 'codes': ['P.117'], 'text': 'b'},
    ]
    assert [g['t_seconds'] for g in group_speech_events(events)] == [3.0, 13.0]
