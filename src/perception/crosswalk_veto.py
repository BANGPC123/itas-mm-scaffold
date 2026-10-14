"""Reject road-paint pieces on rows of broad white crossing stripes.

This runs after semantic-road and occlusion checks. It only removes existing
pieces; it never creates or extends a lane hypothesis.
"""
import cv2
import numpy as np


def crossing_mask(image,ground_probability=None):
    h,w=image.shape[:2]
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);sat,value=hsv[:,:,1],hsv[:,:,2]
    local=value.astype(np.float32)-cv2.GaussianBlur(value,(51,51),0).astype(np.float32)
    mask=((sat<30)&(value>130)&(local>6)).astype(np.uint8)
    if ground_probability is not None:
        if ground_probability.shape!=(h,w):raise ValueError('Ground mask shape mismatch')
        ground=cv2.dilate((ground_probability>.55).astype(np.uint8),np.ones((5,5),np.uint8))
        mask[ground==0]=0
    mask[:round(h*.30)]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,5),np.uint8))
    n,ids,stats,_=cv2.connectedComponentsWithStats(mask,8)
    candidates=[]
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if (bw>=max(20,w*.035) and bw>=1.5*bh and bw<w*.26 and
                4<=bh<h*.12 and area>=max(100,h*w*.0004)):
            candidates.append(i)
    keep=np.zeros(n,np.uint8)
    for i in candidates:
        yc=stats[i,1]+stats[i,3]/2
        group=[j for j in candidates if abs(stats[j,1]+stats[j,3]/2-yc)<h*.05]
        if len(group)>=3 and np.ptp([stats[j,0]+stats[j,2]/2 for j in group])>w*.24:
            keep[group]=1
    return cv2.dilate(keep[ids],np.ones((5,5),np.uint8))


def reject_crosswalk_segments(image,polylines,*,ground_probability=None,coordinate_scale=1.):
    mask=crossing_mask(image,ground_probability);h,w=mask.shape;result=[]
    if not np.isfinite(coordinate_scale) or coordinate_scale<=0:raise ValueError('Invalid coordinate scale')
    for line in polylines:
        piece=[]
        for point in line:
            x,y=(round(v*coordinate_scale) for v in point)
            if 0<=x<w and 0<=y<h and not mask[y,x]:piece.append(point)
            else:
                if len(piece)>=2:result.append(piece)
                piece=[]
        if len(piece)>=2:result.append(piece)
    return result
