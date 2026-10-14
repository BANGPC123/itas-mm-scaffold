"""Optional YOLO11 review-only sign proposals from the user-provided archive.

The model was trained on a different data domain and loses precision when
fused with the existing detector on canonical VAL. Its hypotheses MUST NOT
enter SignApplicability, RAG guidance, speech, or temporal regulatory tracks.
"""
from __future__ import annotations
from pathlib import Path
import json
import hashlib
import math
from typing import Any

import numpy as np


def box_iou(a, b) -> float:
    if not all(math.isfinite(float(x)) for x in (*a, *b)):
        return 0.0
    x1=max(a[0],b[0]);y1=max(a[1],b[1])
    x2=min(a[2],b[2]);y2=min(a[3],b[3])
    intersection=max(0,x2-x1)*max(0,y2-y1)
    area_a=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    area_b=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return intersection/max(area_a+area_b-intersection,1e-6)


def review_candidates(raw_detections, names: dict, image_shape, primary_boxes=(), *,
                      min_confidence=.65, overlap_iou=.35, max_candidates=12,
                      min_side=16):
    """Filter new boxes using frozen conservative thresholds; never invent scope."""
    h,w=image_shape[:2]
    accepted=[]
    kept=list(primary_boxes)
    for bbox, confidence, class_id in sorted(raw_detections,key=lambda a:-a[1]):
        if not math.isfinite(confidence) or confidence<min_confidence:
            continue
        if len(bbox)!=4 or not all(math.isfinite(float(x)) for x in bbox):
            continue
        x1,y1,x2,y2=[float(x) for x in bbox]
        if x1<0 or y1<0 or x2>w or y2>h or x2<=x1 or y2<=y1:
            continue
        if min(x2-x1,y2-y1)<min_side:
            continue
        key=str(int(class_id))
        if key not in names:
            continue
        if any(box_iou((x1,y1,x2,y2),existing)>=overlap_iou for existing in kept):
            continue
        found={'bbox_xyxy':[round(v,2) for v in (x1,y1,x2,y2)],
               'confidence':round(float(confidence),4),
               'class_id':int(class_id),'label_vi':names[key],
               'model':'yolo11_optimized_user_import',
               'recognition_status':'hypothesis_unverified',
               'applicability':'uncertain',
               'car_applicability_verified':False,
               'human_verified':False,
               'legal_warning_allowed':False}
        accepted.append(found)
        kept.append((x1,y1,x2,y2))
        if len(accepted)>=max_candidates:break
    return accepted


class AuxiliarySignReviewer:
    def __init__(self, config:dict[str,Any]):
        self.enabled=bool(config.get('enabled',False))
        self.weights=Path(config.get('weights_path',''))
        self.names_path=Path(config.get('labels_path',''))
        self.confidence=float(config.get('confidence_threshold',.65))
        self.iou=float(config.get('overlap_iou',.35))
        self.imgsz=int(config.get('image_size',416))
        self.max_candidates=int(config.get('max_candidates',12))
        self.model=None
        self.names={}
        self.status='disabled' if not self.enabled else 'missing_model'
        if not self.enabled or not self.weights.is_file() or not self.names_path.is_file():
            return
        manifest=json.loads(self.names_path.read_text(encoding='utf-8'))
        if manifest.get('num_classes')!=59 or len(manifest.get('class_names',{}))!=59:
            self.status='invalid_labels'
            return
        if hashlib.sha256(self.weights.read_bytes()).hexdigest()!=manifest['source_sha256']:
            self.status='weights_hash_mismatch'
            return
        self.names=manifest['class_names']
        from ultralytics import YOLO
        self.model=YOLO(str(self.weights))
        actual=getattr(self.model.model,'names',{})
        if len(actual)!=len(self.names):
            self.model=None
            self.status='model_class_count_mismatch'
            return
        self.status='ready'

    def detect(self, image:np.ndarray, primary_signs=()):
        if self.model is None or image is None or image.size==0:
            return []
        import torch
        device='0' if torch.cuda.is_available() else 'cpu'
        raw=[]
        predictions=self.model.predict(
            image,conf=self.confidence,iou=.45,imgsz=self.imgsz,
            device=device,verbose=False,max_det=60)
        for result in predictions:
            for obj in result.boxes:
                raw.append((obj.xyxy[0].tolist(),float(obj.conf[0]),int(obj.cls[0])))
        return review_candidates(raw,self.names,image.shape,
            [s.bbox_xyxy for s in primary_signs],
            min_confidence=self.confidence,
            overlap_iou=self.iou,max_candidates=self.max_candidates)
