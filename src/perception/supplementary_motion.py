"""Map asynchronous plate reads to the current tracked primary sign."""
import math


def map_plate_rows(rows,old_primary_boxes,visible_primary_boxes,shape):
    h,w=shape[:2];result=[]
    for row in rows:
        pair=(row.get('primary_label'),row.get('primary_track_id'))
        old=old_primary_boxes.get(pair);current=visible_primary_boxes.get(pair)
        if not old or not current or pair[1] is None:continue
        values=[*old,*current,*row.get('bbox_xyxy',[])]
        if len(values)!=12 or not all(math.isfinite(v) for v in values):continue
        ow,oh=old[2]-old[0],old[3]-old[1];cw,ch=current[2]-current[0],current[3]-current[1]
        if min(ow,oh,cw,ch)<=0:continue
        sx,sy=cw/ow,ch/oh
        if not .5<=sx<=2.5 or not .5<=sy<=2.5:continue
        x1,y1,x2,y2=row['bbox_xyxy']
        mapped=[max(0.,current[0]+(x1-old[0])*sx),max(0.,current[1]+(y1-old[1])*sy),
                min(float(w),current[0]+(x2-old[0])*sx),min(float(h),current[1]+(y2-old[1])*sy)]
        if mapped[0]>=mapped[2] or mapped[1]>=mapped[3]:continue
        result.append({**row,'bbox_xyxy':mapped,'source_bbox_xyxy':list(row['bbox_xyxy']),
                       'association_motion_compensated':True,'association_verified':False})
    return result
