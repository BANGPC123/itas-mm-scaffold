"""Logic contract matrix for human-verified scopes, not a QCVN legal audit."""
import pytest
from src.context.sign_applicability import SignApplicability
from src.interaction.sign_speech import sign_speech
from src.utils.types import SignDetection,LaneInfo

@pytest.mark.parametrize('code',['P.130','P.131a','P.102','W.225'])
@pytest.mark.parametrize('scope,expected',[('all','applies'),('ego','applies'),
                                           ('other','ignored'),('unknown','uncertain')])
def test_reviewed_roadside_scope_matrix(code,scope,expected):
    sign=SignDetection(code,.99,(200,100,255,145),panel_role='primary',
         context_verified=True,label_verified=True,placement='roadside',lane_scope=scope)
    result=SignApplicability().apply([sign],LaneInfo([]),(720,1280,3),'car')[0]
    assert result.applicability==expected
    assert bool(sign_speech([result]))==(expected=='applies')


@pytest.mark.parametrize('vehicle,allowed,expected',[
    ('car',['truck'],'ignored'),('car',['motorcycle'],'ignored'),
    ('truck',['truck'],'applies'),('motorcycle',['motorcycle'],'applies'),
    ('truck',['car'],'ignored'),('car',[],'applies')])
def test_reviewed_vehicle_scope_matrix(vehicle,allowed,expected):
    sign=SignDetection('P.130',.99,(100,100,140,150),panel_role='primary',
        context_verified=True,label_verified=True,placement='roadside',lane_scope='all',
        allowed_vehicles=allowed)
    result=SignApplicability().apply([sign],LaneInfo([]),(720,1280,3),vehicle)[0]
    assert result.applicability==expected


@pytest.mark.parametrize('verified',[False,True])
@pytest.mark.parametrize('code',['P.130','P.127','unknown'])
def test_unresolved_or_numeric_context_cannot_generate_audio(verified,code):
    sign=SignDetection(code,.99,(100,100,140,150),panel_role='primary',
        context_verified=verified,label_verified=verified,
        placement='roadside',lane_scope='all')
    result=SignApplicability().apply([sign],LaneInfo([]),(720,1280,3))[0]
    if not verified or code=='unknown':assert result.applicability=='uncertain'
    if not verified or code in ('P.127','unknown'):assert sign_speech([result])==''
