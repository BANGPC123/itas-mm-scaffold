"""QCVN zone transition and ambiguous supplementary fail-closed tests."""
from src.context.verified_zone_transitions import ZoneSignState
from src.context.sign_applicability import SignApplicability
from src.interaction.sign_speech import sign_speech
from src.utils.types import SignDetection,LaneInfo
from src.perception.sign_catalog import NAMES_VI
import pytest

def sign(code,box=(100,100,180,160),**attrs):
    return SignDetection(code,.98,box,**attrs)
def verified(code):
    return sign(code,panel_role='primary',
        context_verified=True,label_verified=True,
        placement='roadside',lane_scope='all',
        applicability='applies')

def test_zone_transition_starts_and_ends_from_only_verified_signs():
    s=ZoneSignState()
    assert s.update([sign('R.420')],stream_id='routeA',timestamp=1)=='unknown'
    assert s.update([verified('R.420')],stream_id='routeA',timestamp=2)=='populated'
    assert s.update([],stream_id='routeA',timestamp=3)=='populated'
    assert s.update([verified('R.421')],stream_id='routeA',timestamp=4)=='outside'
    assert s.last_valid_code=='R.421'

def test_zone_changes_stream_or_seek_fail_closed():
    s=ZoneSignState()
    s.update([verified('R.420')],stream_id='a',timestamp=80)
    assert s.update([],stream_id='b',timestamp=81)=='unknown'
    s.update([verified('R.420')],stream_id='b',timestamp=100)
    assert s.update([],stream_id='b',timestamp=5)=='unknown'

def test_zone_rejects_unverified_vehicle_or_bad_scope():
    s=ZoneSignState()
    r=verified('R.420')
    r.applicability='uncertain'
    assert s.update([r],stream_id='a',timestamp=1)=='unknown'
    r.applicability='applies';r.lane_scope='unknown'
    assert s.update([r],stream_id='a',timestamp=2)=='unknown'

def test_opposing_transitions_same_frame_abstain():
    s=ZoneSignState()
    assert s.update([verified('R.420'),verified('R.421')],stream_id='a',timestamp=3)=='unknown'

@pytest.mark.parametrize('timestamp',[-1,float('nan'),float('inf')])
def test_bad_video_timestamps_rejected(timestamp):
    with pytest.raises(ValueError):ZoneSignState().update([],stream_id='a',timestamp=timestamp)

def test_regulation_codes_present_but_never_infer_speed_limit():
    assert 'R.420' in NAMES_VI and 'R.421' in NAMES_VI
    s=ZoneSignState()
    s.update([verified('R.420')],stream_id='a',timestamp=2)
    assert not hasattr(s,'speed_kmh')

def test_ambiguous_supplement_prevents_main_signs_becoming_audible():
    primary_a=sign('P.130',(100,100,200,200),panel_role='primary',
        context_verified=True,label_verified=True,placement='roadside',lane_scope='all')
    primary_b=sign('P.131a',(115,100,215,200),panel_role='primary',
        context_verified=True,label_verified=True,placement='roadside',lane_scope='all')
    panel=sign('unknown',(125,205,190,225),panel_role='supplementary')
    output=SignApplicability().apply([primary_a,primary_b,panel],LaneInfo([]),(720,1280,3))
    mains=[s for s in output if s.label in ('P.130','P.131a')]
    assert len(mains)==2
    assert all(s.applicability=='uncertain' for s in mains)
    assert sign_speech(output)==''
