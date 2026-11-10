"""Conservative candidate-only detection of rectangular supplementary traffic panels.

This is NOT OCR and NOT an independent traffic-sign detector. It searches only
directly below an already proposed primary sign. Never alters legal scope, TTS
or selected checkpoints; results require human review.
"""
from __future__ import annotations
import cv2
import numpy as np


def bbox_iou(a,b):
    w=max(0,min(a[2],b[2])-max(a[0],b[0]))
    h=max(0,min(a[3],b[3])-max(a[1],b[1]))
    i=w*h
    ar=lambda z:max(0,z[2]-z[0])*max(0,z[3]-z[1])
    return i/max(1e-9,ar(a)+ar(b)-i)


def _candidate_regions(gray, hsv):
    # Two variants recover either bordered printed plates or faded/light plates.
    bright=cv2.inRange(hsv,(0,0,86),(179,150,255))
    bright=cv2.morphologyEx(bright,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    edges=cv2.Canny(gray,55,160)
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    for channel in (bright, edges):
        contours,_=cv2.findContours(channel,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            _,_,cw,ch=cv2.boundingRect(contour)
            if cw<10 or ch<5:continue
            yield cv2.boundingRect(contour)


def propose_plates(image, primary_bbox, *, max_candidates=3):
    """List [{bbox_xyxy, heuristic_score}]; caller must review.

    Rects are image-pixel coordinates. No semantic claims on time/vehicle.
    """
    h,w=image.shape[:2]
    x1,y1,x2,y2=map(float,primary_bbox)
    pw=x2-x1;ph=y2-y1
    if pw<16 or ph<16 or pw>min(500,w*.35) or ph>min(500,h*.5):return []
    xmin=max(0,int(x1-.8*pw));xmax=min(w,int(x2+.8*pw)+1)
    ymin=max(0,int(y2-.12*ph));ymax=min(h,int(y2+1.6*ph)+1)
    if xmax-xmin<20 or ymax-ymin<12:return []
    crop=image[ymin:ymax,xmin:xmax]
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    results=[]
    for cx,cy,cw,ch in _candidate_regions(gray,hsv):
        xx1=xmin+cx; yy1=ymin+cy; xx2=xx1+cw; yy2=yy1+ch
        center=(xx1+xx2)/2
        overlap=max(0,min(x2,xx2)-max(x1,xx1))/max(1,min(pw,cw))
        gap=yy1-y2
        if not (.42*pw<=cw<=2.35*pw and .13*ph<=ch<=1.3*ph):continue
        if not (-.14*ph<=gap<=.85*ph):continue
        if overlap<.60 or not(x1-.3*pw<=center<=x2+.3*pw):continue
        if cw/ch<.95:continue
        inner=hsv[cy:cy+ch,cx:cx+cw]
        light=((inner[:,:,1]<130)&(inner[:,:,2]>110))
        if light.mean()<.44:continue
        # Prioritize compact close-underneath boxes with light background.
        score=round(float(light.mean()-.15*max(0,gap)/ph+.09*overlap),4)
        box=[float(xx1),float(yy1),float(xx2),float(yy2)]
        if bbox_iou(box,primary_bbox)>.24:continue
        results.append({'bbox_xyxy':box,'score':score})
    results.sort(key=lambda p:p['score'],reverse=True)
    uniq=[]
    for item in results:
        if all(bbox_iou(item['bbox_xyxy'],z['bbox_xyxy'])<.45 for z in uniq):
            uniq.append(item)
        if len(uniq)>=max_candidates:break
    return uniq
