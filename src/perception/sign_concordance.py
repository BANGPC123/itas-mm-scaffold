"""Audit independent recognizer concordance on existing secondary YOLO11 proposals.

Agreement is diagnostic only and cannot validate vehicle scope or a rule.
Map only unambiguous exported classes to the released 30-code vocabulary.
"""
from __future__ import annotations
from copy import deepcopy
from src.utils.types import SignDetection

# Index labels are pinned to best.pt SHA256 and the 59-class manifest.
SECONDARY_TO_CODE={
    0:'W.224',2:'P.102',3:'R.302a',
    5:'W.207',6:'W.201a',7:'P.123a',9:'R.303',
    10:'P.130',11:'I.409',13:'W.245a',16:'P.117',
    17:'P.124a',19:'P.124d',27:'W.225',
    32:'P.123b',45:'I.407a',46:'P.131a',48:'W.210',
}
MIN_YOLO_CONFIDENCE=.80
MIN_CODE_CONFIDENCE=.90
MIN_BOX_SIDE=24


def label_agreement(recognizer,image,candidates,*,min_yolo=MIN_YOLO_CONFIDENCE,
                    min_recognition=MIN_CODE_CONFIDENCE):
    """Return new detached candidate dicts; never promote them to legal signs."""
    vetted=deepcopy(candidates)
    eligible=[]
    indices=[]
    for i,row in enumerate(vetted):
        row['agreement_status']='insufficient_evidence'
        row['recognizer_label']='unknown'
        row['recognizer_confidence']=None
        row['legal_warning_allowed']=False
        row['car_applicability_verified']=False
        row['applicability']='uncertain'
        code=SECONDARY_TO_CODE.get(row.get('class_id'))
        a,b,c,d=row['bbox_xyxy']
        if (code is None or row['confidence']<min_yolo or
            min(c-a,d-b)<MIN_BOX_SIDE):
            continue
        indices.append(i)
        eligible.append(SignDetection('unknown',row['confidence'],tuple(row['bbox_xyxy'])))
    if eligible:
        accepted=recognizer.recognize(image,eligible,min_short_side=MIN_BOX_SIDE)
        for idx,sign in zip(indices,accepted):
            row=vetted[idx]
            row['recognizer_label']=sign.label
            row['recognizer_confidence']=sign.recognition_confidence
            # Independent classifier and YOLO must agree on a pinned code.
            if (sign.label==SECONDARY_TO_CODE[row['class_id']]
                and (sign.recognition_confidence or 0)>=min_recognition
                and sign.recognition_source not in (
                    'background_rejected','crop_too_small','low_confidence')):
                row['agreement_status']='cross_model_visual_agreement'
    return vetted
