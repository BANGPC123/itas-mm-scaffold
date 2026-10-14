"""Conservative lane geometry checks for an uncalibrated front-facing camera.

This is a reliability filter, not an alternative ground-truth lane detector.
We deliberately keep untrusted line evidence visible but never infer ego-lane
applicability from a geometrically impossible pair of boundaries.
"""
from __future__ import annotations
import math
import numpy as np


def x_at_y(polyline, y):
    points = sorted({(float(x), float(yy)) for x, yy in polyline}, key=lambda p: p[1])
    if len(points) < 3:
        return None
    ys = np.array([p[1] for p in points])
    if ys[-1] - ys[0] < 1 or y < ys[0] or y > ys[-1]:
        return None
    return float(np.interp(y, ys, [p[0] for p in points]))


def credible_ego(boundaries, shape):
    """Require two distinct, non-crossing road boundaries spanning nearby y."""
    if len(boundaries) != 2:
        return False
    h, w = shape[:2]
    if h <= 0 or w <= 0:
        return False
    for y in (.78*h, .87*h):
        xs = [x_at_y(line, y) for line in boundaries]
        if any(x is None or not math.isfinite(x) for x in xs):
            return False
        gap = xs[1] - xs[0]
        if not .10*w <= gap <= .90*w:
            return False
    return True


def reject_obvious_artifacts(lane, shape):
    """Remove only clearly invalid short/looping polylines; leave curves intact."""
    h, w = shape[:2]
    kept = []
    for points in lane.polylines:
        if len(points) < 3:
            continue
        a = np.asarray(points, dtype=float)
        if not np.isfinite(a).all() or np.any(a[:, 0] < 0) or np.any(a[:, 0] >= w):
            continue
        if np.any(a[:, 1] < 0) or np.any(a[:, 1] >= h):
            continue
        ordered = a[np.argsort(a[:, 1])]
        span = np.ptp(ordered[:, 1])
        if span < .12*h:
            continue
        # A virtually horizontal or repeatedly reversed path is not a lane.
        if np.ptp(ordered[:, 0]) > 3*span:
            continue
        if len(ordered) >= 5:
            dx = np.diff(ordered[:, 0])
            large = dx[np.abs(dx) > .055*w]
            if len(large) >= 3 and np.sum(large[:-1] * large[1:] < 0) >= 2:
                continue
        kept.append(points)
    lane.polylines = kept
    lane.lane_lines = [(*p, *q) for line in kept for p, q in zip(line, line[1:])]
    if not credible_ego(lane.ego_boundaries, shape):
        lane.ego_boundaries = []
        lane.lane_center_offset_px = None
    elif any(line not in kept for line in lane.ego_boundaries):
        lane.ego_boundaries = []
        lane.lane_center_offset_px = None
    return lane
