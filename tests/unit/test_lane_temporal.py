import pytest
from src.perception.lane_temporal import LaneStabilizer,LaneStreamPool


def line(x):
    return [(x-20,200),(x-10,320),(x,440),(x+10,570)]


def test_jitter_is_smoothed_only_for_present_lane():
    s=LaneStabilizer(alpha=.65)
    s.update([line(200)],0,(600,800,3))
    out=s.update([line(225)],.1,(600,800,3))
    assert 200 < out[0][2][0] < 225
    assert abs(out[0][2][0]-200)<abs(225-200)
    assert s.update([],.2,(600,800,3))==[]
    assert s.update([line(225)],.3,(600,800,3))==[line(225)]


def test_seek_or_resize_forgets_all_temporal_geometry():
    s=LaneStabilizer()
    s.update([line(200)],10,(600,800,3))
    assert s.update([line(225)],9,(600,800,3))==[line(225)]
    assert s.update([line(255)],9.1,(700,800,3))==[line(255)]
    assert s.update([line(275)],10,(700,800,3))==[line(275)]


def test_distant_lane_does_not_snap_to_previous_lane():
    s=LaneStabilizer()
    s.update([line(150)],1,(600,800,3))
    assert s.update([line(400)],1.1,(600,800,3))==[line(400)]


def test_display_stabilization_sessions_isolated():
    pool=LaneStreamPool(max_sessions=2)
    a=pool.get('a',(600,800,3),.65)
    a.update([line(150)],0,(600,800,3))
    assert pool.get('b',(600,800,3),.65).previous==[]
    assert pool.get('a',(600,800,3),.8).previous==[]
    pool.get('c',(600,800,3),.65)
    assert len(pool.sessions)==2


def test_invalid_lane_time_rejected():
    with pytest.raises(ValueError):LaneStabilizer().update([],float('nan'),(600,800,3))
