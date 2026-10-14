"""Regression tests for vehicle-specific main + S.505a companion sign."""
from src.context.sign_applicability import SignApplicability
from src.utils.types import SignDetection,LaneInfo

def make_sign(code,box=(100,80,190,175),**kwargs):
    return SignDetection(code,.98,box,**kwargs)
def main():
    return make_sign('P.131a',panel_role='primary',label_verified=True,context_verified=True,
                     placement='roadside',lane_scope='all')
def plate(vehicles,*,verified=True):
    return make_sign('S.505a',(105,190,185,218),panel_role='supplementary',
                     context_verified=verified,label_verified=verified,allowed_vehicles=vehicles)
def infer(vehicle,items):
    return SignApplicability().apply(items,LaneInfo([]),(720,1280,3),vehicle=vehicle)[0]

def test_main_only_keeps_intrinsic_motor_vehicle_scope():
    assert infer('car',[main()]).applicability=='applies'
    assert infer('motorcycle',[main()]).applicability=='applies'
    assert infer('truck',[main()]).applicability=='applies'
    assert infer('bicycle',[main()]).applicability=='ignored'

def test_verified_s505a_car_narrows_main_scope():
    for vehicle,expected in [('car','applies'),('motorcycle','ignored'),
                             ('truck','ignored'),('bus','ignored')]:
        found=infer(vehicle,[main(),plate(['car'])])
        assert found.is_compound
        assert found.applicability==expected, (vehicle,found.applicability_reason)

def test_undecoded_car_picture_must_not_guess_scope():
    for vehicle in ('car','truck'):
        assert infer(vehicle,[main(),plate([],verified=False)]).applicability=='uncertain'

def test_unknown_vehicle_not_treated_as_car():
    assert infer('unrecognized_new_vehicle',[main(),plate(['car'])]).applicability!='applies'

def test_unrelated_unreviewed_panel_does_not_erase_intrinsic_scope():
    assert infer('bicycle',[main()]).applicability=='ignored'
