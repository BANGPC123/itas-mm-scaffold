"""Regression tests: road-edge rejection and broad vs passenger-car S.505a."""
import numpy as np
from src.perception.road_edge_veto import suppress_road_edge_contours
from src.context.sign_applicability import SignApplicability
from src.context.sign_scope_principles import supplementary_vehicle_scope
from src.utils.types import SignDetection,LaneInfo


def test_long_road_edge_is_not_displayed_as_lane():
    ground=np.zeros((300,400),np.float32)
    ground[50:,:]=1
    edge=[(25,y) for y in range(85,265,6)]
    center=[(200,y) for y in range(85,265,6)]
    short=[(25,y) for y in range(100,124,6)]
    retained,rejected=suppress_road_edge_contours([edge,center,short],ground)
    assert rejected==1
    assert retained==[center,short]


def test_no_semantic_road_cannot_suppress_by_guess():
    poly=[(20,y) for y in range(90,250,6)]
    assert suppress_road_edge_contours([poly],np.zeros((300,400),np.float32))==([poly],0)


def test_scaled_predictions_are_not_mutated():
    ground=np.ones((300,400),np.float32)
    poly=[(40,y) for y in range(160,480,10)]
    retained,rejected=suppress_road_edge_contours([poly],ground,coordinate_scale=.5)
    assert retained==[] and rejected==1
    assert poly[0]==(40,160)


def _primary(allowed=()):
    return SignDetection('P.131a',.98,(100,70,190,165),panel_role='primary',
        label_verified=True,context_verified=True,placement='roadside',
        lane_scope='all',allowed_vehicles=list(allowed))


def _plate(allowed):
    return SignDetection('S.505a',.98,(105,175,185,200),panel_role='supplementary',
        label_verified=True,context_verified=True,allowed_vehicles=list(allowed))


def _scope(vehicle,primary,plate=None):
    return SignApplicability().apply(
        [primary]+([plate] if plate else []),LaneInfo([]),(720,1280,3),vehicle=vehicle)[0].applicability


def test_generic_automobile_symbol_includes_car_truck_and_bus():
    for vehicle in ('car','truck','bus','van'):
        assert supplementary_vehicle_scope(['automobile'],vehicle) is True
        assert _scope(vehicle,_primary(),_plate(['ô tô']))=='applies'
    assert supplementary_vehicle_scope(['automobile'],'motorcycle') is False
    assert _scope('motorcycle',_primary(),_plate(['ô tô']))=='ignored'


def test_car_front_symbol_means_passenger_car_not_all_automobiles():
    assert _scope('car',_primary(),_plate(['car']))=='applies'
    assert _scope('truck',_primary(),_plate(['car']))=='ignored'


def test_restricted_parent_intersection_of_broad_auto_and_car():
    assert _scope('car',_primary(['automobile']),_plate(['car']))=='applies'
    assert _scope('truck',_primary(['automobile']),_plate(['car']))=='ignored'


def test_unknown_plate_vehicle_must_abstain():
    assert _scope('car',_primary(),_plate(['unresolved_ocr']))=='uncertain'
