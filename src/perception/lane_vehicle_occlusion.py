"""COCO vehicle-occlusion guard for lane DISPLAY ONLY (not legal lane scope).

Reject polyline candidates that predominantly overlap car/motorcycle/bus/truck
boxes. COCO object boxes are hypotheses, not proof of road markings.
This guard never adds polylines and cannot improve ground truth metrics alone.
"""
from pathlib import Path
import numpy as np

VEHICLE_IDS=(0,1,2,3,5,7)  # person, bicycle, car, motorcycle, bus, truck

def outside_boxes_fraction(line,boxes,shape,margin_ratio=.008):
    if not line:return 1.
    h,w=shape[:2]
    count=0
    for x,y in line:
        covered=any((x1-margin_ratio*w<=x<=x2+margin_ratio*w and
                    y1-margin_ratio*h<=y<=y2+margin_ratio*h)
                    for x1,y1,x2,y2 in boxes)
        count+=int(not covered)
    return count/max(1,len(line))

def suppress_over_vehicle_boxes(polylines,boxes,shape,min_visible_fraction=.68):
    """Keep only lines primarily outside detected occluder bounding boxes."""
    return [pts for pts in polylines
            if outside_boxes_fraction(pts,boxes,shape)>=min_visible_fraction]

class LaneObjectOcclusionGuard:
    def __init__(self,weight='models/lane/candidates/coco_yolo11n_20261011.pt',device='auto',conf=.35):
        self.weight=Path(weight)
        self.device=device
        self.conf=float(conf)
        self.model=None
        self.failure=None
    def get_boxes(self,image):
        if self.model is None:
            if not self.weight.is_file():
                self.failure='missing COCO weight'
                return []
            try:
                from ultralytics import YOLO
                self.model=YOLO(str(self.weight))
            except Exception as exc:
                self.failure=type(exc).__name__
                return []
        try:
            result=self.model.predict(image,device=self.device,
               classes=list(VEHICLE_IDS),conf=self.conf,
               imgsz=640,verbose=False,max_det=35)[0]
            return [tuple(map(float,box))
                    for box in result.boxes.xyxy.cpu().numpy()]
        except Exception as exc:
            self.failure=type(exc).__name__
            return []
    def apply(self,polylines,image):
        if not polylines:return polylines,[]
        boxes=self.get_boxes(image)
        return suppress_over_vehicle_boxes(polylines,boxes,image.shape),boxes
