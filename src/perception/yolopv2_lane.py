"""Official YOLOPv2 BDD100K lane-segmentation inference; independent of UFLD.

The raw binary mask is only a candidate. Conservative connected-component
traces are visualization-only; lane-scope inference requires human review.
"""
from pathlib import Path
import cv2, torch, numpy as np
from src.utils.types import LaneInfo
from src.perception.image_quality import insufficient_information

def traces_from_mask(mask, *, threshold_area=48, min_span_ratio=.14):
    """Do not invent straight lines from empty masks. Output observed pixels only."""
    h,w=mask.shape
    n,ids,stats,centroids=cv2.connectedComponentsWithStats((mask>0).astype('uint8'),8)
    polylines=[]
    for idx in range(1,n):
        area=int(stats[idx,cv2.CC_STAT_AREA])
        x,y,bw,bh=map(int,stats[idx,:4])
        if area<max(threshold_area,int(w*h*.000045)) or bh<min_span_ratio*h:
            continue
        if bw>2.3*bh or bw>int(.74*w):
            continue
        if x+bw<.035*w or x>.965*w:
            continue
        points=[]
        # Only trace rows with actual positive pixels; never interpolate missing range.
        for yy in range(y,y+bh,max(3,round(h/110))):
            sx=np.flatnonzero(ids[yy,max(0,x):min(w,x+bw)]==idx)
            if len(sx):
                # If a connected component spans much of the road horizontally,
                # likely an artifact or intersection; reject.
                if sx.max()-sx.min()>.065*w:continue
                xx=int(np.median(sx)+x)
                points.append((xx,yy))
        if len(points)<9 or points[-1][1]-points[0][1]<min_span_ratio*h:continue
        # Reject very jagged, car-edge lines and horizontal text.
        xs=np.asarray([p[0] for p in points],dtype=float)
        ys=np.asarray([p[1] for p in points],dtype=float)
        if np.ptp(xs)>2.5*np.ptp(ys):continue
        d=np.abs(np.diff(xs))
        if np.quantile(d,.9)>.085*w:continue
        polylines.append(points)
    return sorted(polylines,key=lambda line:len(line),reverse=True)[:6]

class YOLOPv2Lane:
    def __init__(self,weight,device='cpu',half=True,threshold=.56,roi=.80):
        if not Path(weight).is_file():raise FileNotFoundError(weight)
        self.device=torch.device(device)
        self.model=torch.jit.load(str(weight),map_location='cpu').to(self.device).eval()
        self.half=bool(half and self.device.type=='cuda')
        if self.half:self.model.half()
        self.threshold=float(threshold)
        self.roi=float(roi)

    def scene_probabilities(self,image):
        """Full-frame road/paint probabilities with aspect-preserving letterbox."""
        h,w=image.shape[:2]
        scale=min(640/w,384/h);rw,rh=round(w*scale),round(h*scale)
        left,top=(640-rw)//2,(384-rh)//2
        canvas=np.full((384,640,3),114,np.uint8)
        canvas[top:top+rh,left:left+rw]=cv2.resize(image,(rw,rh))
        x=torch.from_numpy(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy())
        x=x.to(self.device,dtype=torch.float16 if self.half else torch.float32)[None]/255
        with torch.inference_mode():
            outputs=self.model(x)
            road=outputs[1][0,1,top:top+rh,left:left+rw].float().cpu().numpy()
            paint=outputs[2][0,0,top:top+rh,left:left+rw].float().cpu().numpy()
        return (cv2.resize(road,(w,h)),cv2.resize(paint,(w,h)))
    def segment(self,image):
        if image is None or image.size==0:
            raise ValueError('Empty image')
        h,w=image.shape[:2]
        # Crop dashboard only if specified. Map the recovered mask to its true
        # image coordinates; do NOT stretch the crop back across the dashboard.
        roi_h=max(1,round(h*self.roi))
        scene=image[:roi_h]
        target_h=round(640*roi_h/w)
        if target_h>640:target_h=640
        # resize canvas into standard 640x384 padded input (16:9 reference)
        input_h=384
        resized=cv2.resize(scene,(640,min(input_h,round(640*roi_h/w))))
        rh=resized.shape[0]
        top=(input_h-rh)//2
        bottom=input_h-rh-top
        if rh<32:raise ValueError('ROI too small')
        padded=cv2.copyMakeBorder(resized,top,bottom,0,0,cv2.BORDER_CONSTANT,value=(114,114,114))
        rgb=cv2.cvtColor(padded,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()
        dtype=torch.float16 if self.half else torch.float32
        tensor=torch.from_numpy(rgb).to(self.device,dtype=dtype)[None]/255
        with torch.inference_mode():
            outputs=self.model(tensor)
            lane=outputs[2]
            prob=lane[0,0,top:top+rh].float().cpu().numpy()
        # down/up conversion before connected-component tracing
        mask=(prob>=self.threshold).astype('uint8')
        mask=cv2.resize(mask,(w,roi_h),interpolation=cv2.INTER_NEAREST)
        return mask
    def detect(self,image):
        if image is None or image.size==0:return LaneInfo([],backend='yolopv2')
        if insufficient_information(image):
            return LaneInfo([],backend='yolopv2',rejection_reason='insufficient_image_information')
        mask=self.segment(image)
        polylines=traces_from_mask(mask)
        edges=[(*p,*q) for line in polylines for p,q in zip(line,line[1:])]
        # Even when marks exist, do not assert which side is the legally valid
        # ego-lane boundary: BDD100K mask has no lane IDs.
        return LaneInfo(edges,None,polylines,[],'yolopv2_visual_candidate' if polylines else 'yolopv2',
                        'segmentation_candidate_unverified' if polylines else None)
