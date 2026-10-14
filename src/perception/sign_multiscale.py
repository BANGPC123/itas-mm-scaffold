"""Recover uncertain crops from another scale without replacing clear labels."""
from copy import deepcopy
import math
from src.perception.temporal_signs import overlap


def crop_side(box, shape):
    h,w=shape[:2];a,b,c,d=box
    return min(min(w,math.ceil(c))-max(0,math.floor(a)),
               min(h,math.ceil(d))-max(0,math.floor(b)))


def recognize_views(recognizer,image,primary,secondary,*,min_short_side=16):
    first=recognizer.recognize(image,deepcopy(primary),min_short_side=min_short_side)
    other=recognizer.recognize(image,deepcopy(secondary),min_short_side=min_short_side)
    matches=[[j for j,s in enumerate(other) if overlap(p.bbox_xyxy,s.bbox_xyxy)>=.5] for p in first]
    usage=[sum(j in candidates for candidates in matches) for j in range(len(other))]
    used=set()
    for p,indices in zip(first,matches):
        q=other[indices[0]] if len(indices)==1 and usage[indices[0]]==1 else None
        if q is not None:used.add(indices[0])
        # Very small physical crops need the same code at both scales.
        if crop_side(p.bbox_xyxy,image.shape)<16:
            if not (q and p.label!='unknown' and q.label==p.label and
                    min(p.recognition_confidence or 0,q.recognition_confidence or 0)>=.95 and
                    max(p.confidence,q.confidence)>=.5):
                p.label='unknown';p.recognition_source='tiny_detail_unsupported'
            continue
        # A strong reference or symbol contradiction must never be rescued.
        if p.label=='unknown' and p.recognition_source in ('low_confidence','crop_too_small') and q:
            if q.label!='unknown' and (q.recognition_confidence or 0)>=.90 and q.confidence>=.60 and crop_side(q.bbox_xyxy,image.shape)>=16:
                p.label=q.label;p.recognition_confidence=q.recognition_confidence
                p.bbox_xyxy=q.bbox_xyxy;p.recognition_source='multiscale_rescue'
    for j,q in enumerate(other):
        if j in used or any(overlap(q.bbox_xyxy,p.bbox_xyxy)>=.30 for p in first):continue
        if q.label!='unknown' and (q.recognition_confidence or 0)>=.97 and q.confidence>=.65 and crop_side(q.bbox_xyxy,image.shape)>=16:
            q.recognition_source='multiscale_rescue';first.append(q)
    return first
