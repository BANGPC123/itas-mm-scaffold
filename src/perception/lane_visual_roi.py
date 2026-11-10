"""Optional display-only lane region probe for MP4 demos.

Uses existing UFLD weights; only when the full-frame UFLD returns NO lanes.
ROIs can improve visibility on dashboard-heavy videos but can also yield
spurious polylines, so NEVER use it for lane scope or law.
"""
from src.perception.lane_consistency import reject_obvious_artifacts

def optional_lane_visual_roi(detector, frame, original, enabled=False):
    if (not enabled or original.backend!='ufld_v2' or
        original.polylines or frame is None or frame.shape[0]<240):
        return original,False
    learned=getattr(detector,'learned',None)
    if learned is None:return original,False
    roi_h=round(frame.shape[0]*.72)
    cropped=frame[:roi_h]
    visual=learned.detect(cropped,filter_geometry=True)
    if not visual.polylines:return original,False
    visual=reject_obvious_artifacts(visual,cropped.shape)
    if not visual.polylines:return original,False
    # No trusted ego-lane association: detections are only optional graphics.
    original.polylines=visual.polylines
    original.lane_lines=visual.lane_lines
    original.display_polylines=visual.polylines
    original.ego_boundaries=[]
    original.lane_center_offset_px=None
    original.backend='ufld_v2_roi_visual_only'
    original.rejection_reason='roi_lane_visual_candidate_not_scope_evidence'
    return original,True
