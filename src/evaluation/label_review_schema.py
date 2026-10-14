"""Validation for reviewer-submitted field labels; AI labels never equal ground truth."""
import math
JUDGMENTS={'unreviewed','sign','not_sign','uncertain'}
LANES={'unreviewed','annotated','no_visible_lane','uncertain'}
ROLES={'unknown','primary','supplementary'}

def validate_review(data,manifest=None):
    if data.get('schema_version')!='itasmm-field-review-v1':raise ValueError('Schema version')
    frames=data.get('frames')
    if not isinstance(frames,list) or not frames:raise ValueError('No frames')
    source={f['id']:f for f in manifest['frames']} if manifest else None
    ids=set();reviewed=exhaustive=0
    signs=0;negatives=0
    for f in frames:
        fid=f.get('id')
        if not isinstance(fid,str) or fid in ids:raise ValueError('Duplicate frame')
        ids.add(fid)
        if source is not None:
            if fid not in source or source[fid]['image_sha256']!=f.get('image_sha256'):
                raise ValueError('Frame provenance mismatch')
            for key in ('width','height','source_group','image','timestamp_seconds'):
                if source[fid].get(key)!=f.get(key):
                    raise ValueError('Source metadata mismatch: '+key)
        w,h=f.get('width'),f.get('height')
        if type(w)!=int or type(h)!=int or not 0<w<=12000 or not 0<h<=12000:
            raise ValueError('Image dimensions')
        labels=f.get('signs')
        if not isinstance(labels,list):raise ValueError('No signs list')
        seen=set()
        for s in labels:
            sid=s.get('id')
            if not isinstance(sid,str) or sid in seen:raise ValueError('Duplicate sign')
            seen.add(sid)
            if s.get('judgment') not in JUDGMENTS or s.get('panel_role') not in ROLES:
                raise ValueError('Invalid sign judgment/role')
            box=s.get('bbox_xyxy')
            if not isinstance(box,list) or len(box)!=4 or not all(type(v) in (int,float) and math.isfinite(v) for v in box):
                raise ValueError('Invalid bbox')
            x1,y1,x2,y2=box
            if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('BBox out of bounds')
            text=s.get('plate_text')
            verified=s.get('plate_text_verified',False)
            if type(verified) is not bool or (text is not None and
                (not isinstance(text,str) or len(text)>300)):
                raise ValueError('Invalid supplementary text metadata')
            if verified is True and (s.get('judgment')!='sign' or
                s.get('panel_role')!='supplementary' or
                not isinstance(text,str) or not text.strip()):
                raise ValueError('Verified plate text requires a real supplementary sign with nonempty transcription')
        lanes=f.get('lanes')
        if not isinstance(lanes,list):raise ValueError('No lane list')
        for lane in lanes:
            pts=lane.get('points')
            if lane.get('role') not in {'ego_left','ego_right','other'} or not isinstance(pts,list) or not 2<=len(pts)<=100:
                raise ValueError('Bad lane annotation')
            if any(not isinstance(p,list) or len(p)!=2 or not all(type(v) in (int,float) and math.isfinite(v) for v in p) or not 0<=p[0]<=w or not 0<=p[1]<=h for p in pts):
                raise ValueError('Lane point out of bounds')
        if f.get('lane_status') not in LANES:raise ValueError('Bad lane review status')
        if f['lane_status']=='annotated' and not lanes:raise ValueError('Empty annotated lanes')
        if f['lane_status']=='no_visible_lane' and lanes:raise ValueError('Contradictory lanes')
        if f.get('human_confirmed') is True:
            reviewed+=1
            if not str(f.get('reviewer','')).strip() or not str(f.get('reviewed_at','')).strip():
                raise ValueError('Missing reviewer attribution')
            if any(s['judgment']=='unreviewed' for s in labels) or f['lane_status']=='unreviewed':
                raise ValueError('Cannot confirm unreviewed objects/lane status')
            if any(s['judgment']=='sign' and not str(s.get('label','')).strip() for s in labels):
                raise ValueError('Confirmed sign must have a label; use unknown if code is unreadable')
            if f.get('all_visible_signs_annotated') is True:
                if any(s['judgment']=='uncertain' for s in labels):
                    raise ValueError('Exhaustive frame has unresolved sign')
                exhaustive+=1
            signs+=sum(s['judgment']=='sign' for s in labels)
            negatives+=sum(s['judgment']=='not_sign' for s in labels)
    return {'frames':len(frames),'human_confirmed':reviewed,'exhaustive':exhaustive,
        'confirmed_signs':signs,'confirmed_negatives':negatives,
        'detector_recall_evaluable':bool(exhaustive),
        'independent_source_groups':len({f.get('source_group') for f in frames if f.get('human_confirmed')})}
