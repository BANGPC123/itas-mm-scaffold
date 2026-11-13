"""M1.2: with the lane stage off, lane-dependent sign scope must stay uncertain (never 'applies')."""
import pytest
from src.context.sign_applicability import SignApplicability
from src.utils.types import LaneInfo, SignDetection


def _overhead_ego_sign():
    return SignDetection('P.130', .9, (100, 20, 300, 120), panel_role='primary',
                         label_verified=True, context_verified=True,
                         placement='overhead', lane_scope='ego')


def test_disabled_lane_gives_uncertain_for_overhead_ego_sign():
    lane = LaneInfo([], backend='disabled')
    out = SignApplicability({}).apply([_overhead_ego_sign()], lane, (720, 1280, 3), 'car')
    assert out[0].applicability == 'uncertain'
    assert out[0].applicability != 'applies'


def test_empty_ego_boundaries_gives_uncertain_for_overhead_ego_sign():
    lane = LaneInfo([], ego_boundaries=[], backend='ufld_v2')
    out = SignApplicability({}).apply([_overhead_ego_sign()], lane, (720, 1280, 3), 'car')
    assert out[0].applicability == 'uncertain'


def test_disabled_lane_does_not_crash_on_empty_signs():
    lane = LaneInfo([], backend='disabled')
    assert SignApplicability({}).apply([], lane, (720, 1280, 3), 'car') == []
