"""Fail-closed visual-only lane curve quality checks for dashboard-heavy MP4.

Reject obvious horizontal/car-outline artifacts; don't alter ML raw
polylines, ego lane geometry or application rules. Heuristics cannot
establish GT correctness.
"""
import numpy as np

def display_quality(line,shape):
    if len(line)<5:return False
    h,w=shape[:2]
    xy=np.asarray(line,dtype=float)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():return False
    xy=xy[np.argsort(xy[:,1])]
    ys=xy[:,1];xs=xy[:,0]
    span=float(np.ptp(ys));run=float(np.ptp(xs))
    if span<.16*h or run>1.28*span:return False
    if ys.min()<0 or ys.max()>h:return False
    # Straight/curved lane can bend, but car bumper silhouettes and
    # dashboard edges have implausibly strong zig-zag changes.
    diff=np.diff(xs)
    if len(diff)>=3:
        signs=np.sign(diff[np.abs(diff)>2])
        if len(signs)>=4 and np.count_nonzero(signs[1:]*signs[:-1]<0)>=3:
            return False
    y_steps=np.diff(ys)
    if len(y_steps)>=3:
        positive=y_steps>0
        if np.count_nonzero(positive)<len(y_steps)*.7:return False
        slopes=np.diff(xs)/np.maximum(y_steps,1)
        if np.quantile(np.abs(slopes),.9)>2.0:return False
    return True

def visual_polylines(lane,shape):
    return [list(p) for p in lane.polylines if display_quality(p,shape)]
