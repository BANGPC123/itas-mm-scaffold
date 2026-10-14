"""QCVN 41:2024 Appendix B.30/B.31, F.5 source-specific scope examples.

Legal scope logic is separate from AI confidence, parent panel verification and
roadside placement. A sign does not create a legal command based on a box alone.
"""
from src.context.sign_applicability import SignApplicability
from src.context.sign_scope_principles import base_vehicle_scope
from src.utils.types import SignDetection,LaneInfo
from src.interaction.sign_speech import sign_speech

def sign(code,xy=(100,100,210,210),**kwargs):
    return SignDetection(code,.98,xy,**kwargs)
def primary():
    return sign('P.131a',panel_role='primary',placement='roadside',
        lane_scope='all',context_verified=True,label_verified=True)
def plate(*,vehicle='car',verified=True):
    return sign('S.505a',(115,219,193,251),panel_role='supplementary',
        allowed_vehicles=[vehicle] if vehicle else [],
        context_verified=verified,label_verified=verified)
def check(signs,vehicle):
    return SignApplicability().apply(signs,LaneInfo([]),(720,1280,3),vehicle)
def test_only_primary_parking_sign_follows_base_motor_scope():
    for vehicle in ('car','truck','bus','motorbike'):
        out=check([primary()],vehicle)
        assert out[0].applicability=='applies',(vehicle,out[0])
        assert not out[0].is_compound
    assert check([primary()],'bicycle')[0].applicability=='ignored'

def test_reviewer_verified_vehicle_plate_narrows_scope_to_car():
    both=[primary(),plate(vehicle='car')]
    car=check(both,'car')
    bike=check(both,'motorbike')
    truck=check(both,'truck')
    assert len(car)==len(bike)==len(truck)==1
    assert car[0].is_compound and car[0].panel_labels==['P.131a','S.505a']
    assert car[0].applicability=='applies'
    assert bike[0].applicability=='ignored'
    assert truck[0].applicability=='ignored'
    assert car[0].allowed_vehicles==['car']
    assert not both[0].is_compound  # immutable caller inputs

def test_vehicle_aliases_cannot_change_scope():
    assert check([primary(),plate(vehicle='ô tô')],'car')[0].applicability=='applies'
    assert check([primary(),plate(vehicle='car')],'xe máy')[0].applicability=='ignored'

def test_unreadable_s505a_never_widens_primary_sign_to_all_vehicles():
    for plate_data in [plate(vehicle=None),plate(vehicle='car',verified=False)]:
        x=check([primary(),plate_data],'car')
        assert x[0].applicability=='uncertain'
        assert sign_speech(x)==''

def test_temporally_unknown_vehicle_class_is_not_ignored():
    assert check([primary()],'electric_scooter_unknown')[0].applicability=='uncertain'

def test_other_modifier_time_rule_is_not_silently_ignored():
    time_sub=sign('S.508a',(114,219,190,250),
        panel_role='supplementary',label_verified=True,context_verified=True)
    assert check([primary(),time_sub],'car')[0].applicability=='uncertain'

def test_legal_scope_base_not_universal_for_every_vehicle():
    assert base_vehicle_scope('P.131a','bicycle')=='excluded'
    assert base_vehicle_scope('P.130','motorbike')=='included'
    assert base_vehicle_scope('P.131a','unknown_thing')=='unknown'
