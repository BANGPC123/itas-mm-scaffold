"""Strict, human-label-aware end-to-end field evaluation (research prototype).

Scores are only meaningful on source-disjoint human-reviewed video frames.
Detection precision/recall requires exhaustive ground truth. Unknown labels,
uncertain regions and partially reviewed frames are NEVER assumed negatives.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from math import isfinite
import numpy as np
import cv2

def iou(a,b):
    if len(a)!=4 or len(b)!=4:return 0.0
    x=max(0.,min(a[2],b[2])-max(a[0],b[0]))
    y=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    inter=x*y
    aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])
    ab=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    return inter/max(aa+ab-inter,1e-9)

def associate(gt,pred,threshold=.5):
    """Greedy highest-IoU one-to-one matching; no duplicate-credit."""
    scored=[(iou(g['bbox_xyxy'],p['bbox_xyxy']),i,j)
            for i,g in enumerate(gt) for j,p in enumerate(pred)]
    scored.sort(reverse=True)
    used_g=set();used_p=set();matched=[]
    for score,i,j in scored:
        if score<threshold:break
        if i in used_g or j in used_p:continue
        used_g.add(i);used_p.add(j);matched.append((i,j,score))
    return matched

def f1(precision,recall):
    return 2*precision*recall/(precision+recall) if precision+recall else 0.

def render_lane_mask(shape,lanes,thickness=3):
    h,w=shape[:2]
    mask=np.zeros((h,w),np.uint8)
    for lane in lanes:
        raw=lane['points'] if isinstance(lane,dict) else lane
        if len(raw)<2:continue
        pts=np.asarray(raw,dtype=np.float64)
        if pts.ndim!=2 or pts.shape[1]!=2 or not np.isfinite(pts).all():continue
        pts=np.rint(pts).astype(np.int32)
        cv2.polylines(mask,[pts],False,1,thickness,cv2.LINE_8)
    return mask

def lane_pixel_counts(shape,gt,pred,tolerance=6):
    """Tolerance-based symmetric lane-pixel TP. Predictions not smoothed."""
    if not(1<=tolerance<=20):raise ValueError('Unsupported lane tolerance')
    gtmask=render_lane_mask(shape,gt).astype(bool)
    pdmask=render_lane_mask(shape,pred).astype(bool)
    if not(gtmask.any() or pdmask.any()):return None
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*tolerance+1,2*tolerance+1))
    gtdilate=cv2.dilate(gtmask.astype(np.uint8),k).astype(bool)
    pddilate=cv2.dilate(pdmask.astype(np.uint8),k).astype(bool)
    return {'matched_pred_pixels':int(np.count_nonzero(pdmask&gtdilate)),
            'pred_pixels':int(np.count_nonzero(pdmask)),
            'matched_gt_pixels':int(np.count_nonzero(gtmask&pddilate)),
            'gt_pixels':int(np.count_nonzero(gtmask))}

def _score_lane(counts,tolerance=6):
    if not counts:return None
    tp=sum(c['matched_pred_pixels'] for c in counts)
    pred=sum(c['pred_pixels'] for c in counts)
    rg=sum(c['matched_gt_pixels'] for c in counts)
    gt=sum(c['gt_pixels'] for c in counts)
    precision=tp/pred if pred else 0.
    recall=rg/gt if gt else 0.
    return {'precision':round(precision,6),'recall':round(recall,6),
            'pixel_tolerance_f1':round(f1(precision,recall),6),
            'annotated_frames':len(counts),'pixel_tolerance':tolerance}

def score_frames(samples, *, threshold=.5, lane_tolerance=6):
    """Each sample {gt: imported_record, pred:{signs,lanes}, image_shape}.

    NOTE: only confirmed human review sources may be passed by the caller.
    Insufficiently exhaustive frames contribute matched class QA but NOT
    detection false-positive/negative metrics.
    """
    overall=Counter()
    tp=fp=fn=0
    neg_accept=neg_total=0
    code_correct=code_total=0
    lane_counts=[]
    false_positive_details=[]
    for entry in samples:
        record=entry['gt'];pred=entry['pred']
        if not(record.get('reviewer') and record.get('reviewed_at')):
            raise ValueError('No human review attribution')
        labeled=[s for s in record['signs'] if s['judgment']=='sign']
        negative=[s for s in record['signs'] if s['judgment']=='not_sign']
        uncertain=[s for s in record['signs'] if s['judgment']=='uncertain']
        detections=pred.get('signs',[])
        pairs=associate(labeled,detections,threshold)
        for i,j,_ in pairs:
            gt_label=labeled[i].get('label','unknown')
            if gt_label and gt_label!='unknown':
                code_total+=1
                code_correct+=int(detections[j].get('label')==gt_label)
        overall['human_reviewed_frames']+=1
        overall['human_sign_boxes']+=len(labeled)
        overall['human_negative_boxes']+=len(negative)
        overall['human_unknown_boxes']+=len(uncertain)
        if record.get('all_visible_signs_annotated') is True and not uncertain:
            tp+=len(pairs)
            fn+=len(labeled)-len(pairs)
            matched_pred={j for _,j,_ in pairs}
            for j,det in enumerate(detections):
                if j in matched_pred:continue
                # Do not count proposals overlapping explicitly labeled
                # uncertain regions as confirmed false positives.
                if any(iou(det['bbox_xyxy'],z['bbox_xyxy'])>=.3 for z in uncertain):continue
                fp+=1
                false_positive_details.append({'frame':record['frame_id'],
                    'label':det.get('label','unknown'),'bbox_xyxy':det['bbox_xyxy']})
            overall['exhaustive_frames']+=1
        # A negative-region false accept is whether a *sign class* was
        # predicted on a human-reviewed non-sign crop, not just a YOLO box.
        for neg in negative:
            neg_total+=1
            if any(d.get('label','unknown')!='unknown' and
                   iou(neg['bbox_xyxy'],d['bbox_xyxy'])>=.3 for d in detections):
                neg_accept+=1
        if record.get('lane_status') in ('annotated','no_visible_lane'):
            counts=lane_pixel_counts(entry['image_shape'],record.get('lanes',[]),
                                     pred.get('lanes',[]),lane_tolerance)
            if counts:lane_counts.append(counts)
            overall['lane_reviewed_frames']+=1
    result={'frames_scored':overall['human_reviewed_frames'],
            'exhaustive_detection_frames':overall['exhaustive_frames'],
            'reviewed_sign_boxes':overall['human_sign_boxes'],
            'reviewed_negative_boxes':overall['human_negative_boxes'],
            'unresolved_regions':overall['human_unknown_boxes']}
    if overall['exhaustive_frames']:
        precision=tp/(tp+fp) if tp+fp else None
        recall=tp/(tp+fn) if tp+fn else None
        result['detection']={'true_positive':tp,'false_positive':fp,'false_negative':fn,
            'precision':precision,'recall':recall,
            'f1':f1(precision,recall) if precision is not None and recall is not None else None,
            'iou_threshold':threshold}
    else:result['detection']={'status':'blocked_no_exhaustive_review_frames'}
    result['sign_code']={'evaluated':code_total,
        'accuracy':code_correct/code_total if code_total else None}
    result['non_sign_regions']={'evaluated':neg_total,
        'false_known_sign_accepts':neg_accept,
        'false_accept_rate':neg_accept/neg_total if neg_total else None}
    result['lane'] = _score_lane(lane_counts,lane_tolerance) or {'status':'blocked_no_reviewed_lane_annotations'}
    result['false_positive_examples']=false_positive_details[:30]
    return result

def field_gate(records,minimum_test_groups=2,minimum_exhaustive_frames=30):
    """Explicit minimum-evidence policy; PASS is not road-safety certification."""
    test=[r for r in records if r.get('split')=='TEST']
    development=[r for r in records if r.get('split') in ('TRAIN','VAL')]
    test_groups={r['source_group'] for r in test}
    dev_groups={r['source_group'] for r in development}
    issues=[]
    if len(test_groups)<minimum_test_groups:
        issues.append('need_at_least_two_distinct_test_route_groups')
    if test_groups&dev_groups:issues.append('test_route_leakage')
    if any(not r.get('source_video_sha256') for r in test):
        issues.append('missing_original_video_sha256_for_test_route')
    dev_video_hashes={r['source_video_sha256'] for r in development if r.get('source_video_sha256')}
    test_video_hashes={r['source_video_sha256'] for r in test if r.get('source_video_sha256')}
    if dev_video_hashes & test_video_hashes:
        issues.append('same_video_bytes_in_development_and_test')
    if len(test_video_hashes)<len(test_groups):
        issues.append('test_route_groups_not_video_unique')
    exhaustive=sum(r.get('all_visible_signs_annotated') is True and
                  not any(s['judgment']=='uncertain' for s in r['signs']) for r in test)
    if exhaustive<minimum_exhaustive_frames:
        issues.append('need_more_human_exhaustive_test_frames')
    if not any(r.get('lane_status')=='annotated' for r in test):
        issues.append('lane_geometry_unassessed')
    return {'field_evidence_ready':not issues,'blockers':issues,
            'test_route_groups':len(test_groups),'reviewed_test_frames':len(test),
            'exhaustive_test_frames':exhaustive,
            'min_test_groups_policy':minimum_test_groups,
            'min_exhaustive_test_frames_policy':minimum_exhaustive_frames,
            'source_video_hash_groups':len(test_video_hashes),
            'note':'Data sufficiency gate only; passing is NOT safety validation.'}
