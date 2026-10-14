from dataclasses import replace
from datetime import datetime
import pytest
from src.context.sign_applicability import SignApplicability
from src.context.time_scope import time_scope
from src.interaction.sign_speech import sign_speech
from src.utils.types import SignDetection, LaneInfo


def primary(code='P.131a'):
    return SignDetection(code,.99,(100,100,200,200),panel_role='primary',
                         placement='roadside',lane_scope='all',context_verified=True,label_verified=True)


def apply(signs,vehicle='car',at=None):
    return SignApplicability().apply(signs,LaneInfo([]),(720,1280,3),vehicle,observed_at=at)


def test_motorbike_alias_matches_reviewed_motorcycle_panel_and_is_spoken():
    sub=SignDetection('S.505a',.9,(110,210,190,240),panel_role='supplementary',
                      allowed_vehicles=['motorbike'],context_verified=True,label_verified=True)
    result=apply([primary(),sub],'motorcycle')
    assert result[0].applicability=='applies'
    assert 'xe máy' in sign_speech(result)
    assert apply([primary(),sub],'car')[0].applicability=='ignored'


def test_multiple_reviewed_vehicle_panels_cannot_widen_previous_restriction():
    first=SignDetection('car plate',.9,(110,210,190,230),panel_role='supplementary',
                       allowed_vehicles=['car'],context_verified=True,label_verified=True)
    second=replace(first,label='truck plate',bbox_xyxy=(110,235,190,250),allowed_vehicles=['truck'])
    result=apply([primary(),first,second],'truck')
    assert result[0].applicability!='applies'
    assert primary().allowed_vehicles==[]


@pytest.mark.parametrize('code,day,expected',[('P.131b',11,'applies'),('P.131b',12,'ignored'),('P.131c',11,'ignored'),('P.131c',12,'applies')])
def test_odd_even_parking_uses_observed_vietnam_date(code,day,expected):
    assert apply([primary(code)],at=datetime.fromisoformat(f'2026-10-{day}T08:00:00+07:00'))[0].applicability==expected
    assert apply([primary(code)])[0].applicability=='uncertain'


def test_observation_date_converts_utc_to_vietnam_before_parity():
    assert time_scope('P.131c',[],datetime.fromisoformat('2026-10-11T19:00:00+00:00')) is True


def test_verified_time_plate_requires_time_and_respects_windows():
    sub=SignDetection('S.508a',.9,(110,210,190,240),panel_role='supplementary',
                      time_windows=[('07:00','09:00'),('16:00','18:00')],
                      context_verified=True,label_verified=True)
    for hour,status in [(8,'applies'),(12,'ignored'),(17,'applies')]:
        out=apply([primary(),sub],at=datetime.fromisoformat(f'2026-10-11T{hour:02}:00:00+07:00'))
        assert out[0].applicability==status
        if status=='applies': assert '07:00 đến 09:00' in sign_speech(out)
    assert apply([primary(),sub])[0].applicability=='uncertain'
    unverified=replace(sub,label_verified=False)
    assert apply([primary(),unverified],at=datetime.fromisoformat('2026-10-11T08:00:00+07:00'))[0].applicability=='uncertain'


def test_cross_midnight_window_and_endpoint():
    cond=[[('22:00','06:00')]]
    for hour,expected in [(23,True),(1,True),(6,False),(12,False)]:
        assert time_scope('P.130',cond,datetime.fromisoformat(f'2026-10-11T{hour:02}:00:00+07:00')) is expected
    assert time_scope('P.130',cond,datetime(2026,10,11,23)) is None
