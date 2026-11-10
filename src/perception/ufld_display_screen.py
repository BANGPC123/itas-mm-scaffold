"""Precision-first screen for *display-only* UFLD v2 guesses.

Grounded by inspection of user video: model sometimes predicts horizontal /
zigzag polylines across cars and instrument panel. Do not call any screened
polyline verified lane GT or apply to legal sign-to-lane matching.
"""
from __future__ import annotations
import numpy as np
from src.utils.types import LaneInfo

def screen_ufld_for_demo(image,original, *,
                         road_limit=.72,min_road_span=.19,max_slope=1.30,
                         max_linear_rms=.015):
    """Screen UFLD-v2 polygons only when user's MP4 demo option is enabled.

    Fully visible road crop is intentionally conservative. Dashboard/vehicle
    lines are suppressed; lower recall is preferred over false lines.
    """
    if original.backend!='ufld_v2' or not original.polylines:
        return original,False
    height,width=image.shape[:2]
    top_bottom_y=height*road_limit
    accepted=[]
    rejected=0
    for line in original.polylines:
        xy=np.asarray(line,dtype=np.float64)
        if len(xy)<9 or not np.isfinite(xy).all():
            rejected+=1;continue
        xy=xy[xy[:,1]<=top_bottom_y]
        if len(xy)<9:
            rejected+=1;continue
        xy=xy[np.argsort(xy[:,1])]
        y=xy[:,1];x=xy[:,0]
        span=float(np.ptp(y))
        if span<min_road_span*height:
            rejected+=1;continue
        # Parallel to road-facing camera perspective, not horizontal vehicle edges.
        slope,offset=np.polyfit(y,x,1)
        if abs(slope)>max_slope:
            rejected+=1;continue
        linear_error=float(np.sqrt(np.mean((x-(slope*y+offset))**2)))
        if linear_error>max_linear_rms*width:
            rejected+=1;continue
        # Road lines have smooth, almost monotone variation in x.
        dx=np.diff(x)
        wiggle=float(np.sum(np.abs(dx)) / max(1.,float(np.ptp(x))))
        if wiggle>1.14:
            rejected+=1;continue
        if np.max(np.abs(dx))>.080*width:
            rejected+=1;continue
        # Keep real observed rows, do not extrapolate or invent missing points.
        accepted.append([(int(round(x)),int(round(y))) for x,y in xy])
    if not rejected and len(accepted)==len(original.polylines):
        # Still constrain dashboard visualization and never feed an unreviewed
        # ego pair into legal inference in this user-facing demo.
        pass
    original.polylines=accepted
    original.lane_lines=[(*a,*b) for line in accepted for a,b in zip(line,line[1:])]
    original.display_polylines=accepted
    original.ego_boundaries=[]
    original.lane_center_offset_px=None
    original.rejection_reason=('screened_unverified_visual_geometry'
                                if accepted else 'no_credible_ufld_road_polyline')
    return original,bool(rejected)
