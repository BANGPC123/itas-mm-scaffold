"""Conservative visual-only suppression of long road-edge contours.

A semantic road mask gives rough carriageway boundaries, not validated legal
lanes. Long paint next to either edge is more likely an edge marking than a
vehicle lane divider. Short road dashes are preserved for visual review.
No new geometry is ever created.
"""
import numpy as np


def suppress_road_edge_contours(polylines, ground_probability, *,
                                coordinate_scale=1.0,
                                min_span_ratio=0.12,
                                edge_margin_ratio=0.22):
    """Return retained original polylines and number of suppressed contours."""
    ground=np.asarray(ground_probability)
    if ground.ndim != 2 or not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError('Ground mask or coordinate scale is invalid')
    h,w=ground.shape
    retained=[];rejected=0
    for poly in polylines:
        points=np.asarray(poly,dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            rejected+=1
            continue
        # A short dashed divider is not a continuous road-edge contour.
        if len(points)<8 or np.ptp(points[:,1])*coordinate_scale < h*min_span_ratio:
            retained.append(poly)
            continue
        margins=[]
        # Occluding cars may leave a narrow semantic road strip by a median.
        # Only a long, densely sampled contour may use this narrower support.
        narrow_candidate=(len(points)>=12 and np.ptp(points[:,1])*coordinate_scale>=h*.16)
        narrow_rows=0
        for x,y in points[::max(1,len(points)//30)]:
            x=int(round(x*coordinate_scale));y=int(round(y*coordinate_scale))
            if not (0<=x<w and 0<=y<h):
                continue
            road_x=np.flatnonzero(ground[y]>.55)
            if len(road_x)<w*.12:
                if not narrow_candidate or len(road_x)<w*.045:
                    continue
                narrow_rows+=1
            left,right=int(road_x[0]),int(road_x[-1])
            if right-left <= w*(.045 if narrow_candidate else .12):
                continue
            margins.append(min(x-left,right-x)/(right-left))
        # Insufficient semantic mask: abstain instead of deleting markings.
        if len(margins)<6:
            retained.append(poly)
            continue
        near=float(np.mean(np.asarray(margins)<edge_margin_ratio))
        # Narrow masks are weaker evidence: require a strong, consistently
        # one-sided contour instead of suppressing intermittent center dashes.
        threshold=.78 if narrow_rows>=6 else .67
        if near>=threshold and float(np.median(margins))<edge_margin_ratio:
            rejected+=1
        else:
            retained.append(poly)
    return retained,rejected
