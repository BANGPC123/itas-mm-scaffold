"""Opt-in UFLDv2 + YOLOPv2 visual-only lane fallback for MP4 university demo.

The selected CULane UFLD checkpoint remains the authority on any frame on
which it returns lanes. BDD YOLOPv2 is only consulted when UFLD has NO lanes.
Its polygons are never used for legal lane scope, only amber dashed overlay.

The fallback does not hallucinate lines when neither model supports them.
"""
from pathlib import Path
import numpy as np
from src.utils.types import LaneInfo
from src.perception.yolopv2_lane import YOLOPv2Lane

class SecondaryLaneVisual:
    def __init__(self,model_path,*,device='cuda:0',roi=.8,threshold=.56,min_span_ratio=.20):
        self.path=Path(model_path)
        self.device=device
        self.roi=roi
        self.threshold=threshold
        self.min_span_ratio=min_span_ratio
        self.candidate=None
        self.load_failure=None

    def detect_if_needed(self,image,baseline):
        if baseline.backend!='ufld_v2' or baseline.polylines:
            return baseline,False
        if not self.path.is_file():
            self.load_failure='official_yolopv2_checkpoint_unavailable'
            return baseline,False
        if self.candidate is None:
            try:
                self.candidate=YOLOPv2Lane(self.path,device=self.device,
                  half=True,roi=self.roi,threshold=self.threshold)
            except Exception as exc:
                self.load_failure=f'{type(exc).__name__}: {str(exc)[:90]}'
                return baseline,False
        try:
            result=self.candidate.detect(image)
        except Exception as exc:
            self.load_failure=f'{type(exc).__name__}: {str(exc)[:90]}'
            return baseline,False
        h,w=image.shape[:2]
        good=[]
        grayscale=None
        for line in result.polylines:
            xy=np.asarray(line,dtype=float)
            if (len(line)<9 or
                np.ptp(xy[:,1]) < self.min_span_ratio*h or
                xy[:,1].max()>.84*h or
                np.ptp(xy[:,0])>2.4*np.ptp(xy[:,1])):
                continue
            if grayscale is None:
                import cv2
                grayscale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            probes=xy[::max(1,len(xy)//10)].astype(int)
            brightness=np.median([float(grayscale[max(0,min(h-1,y)),max(0,min(w-1,x))])
                                  for x,y in probes])
            if brightness<90:continue
            # Checked failure in the user's video at 21.5s: one YOLOPv2
            # line crossed cars/motorcycles instead of following road paint.
            signed_slope=float(np.polyfit(xy[:,1],xy[:,0],1)[0])
            if abs(signed_slope)>1.25:continue
            good.append([(int(x),int(y)) for x,y in line])
        if not good:
            return baseline,False
        # YOLOPv2 sometimes returns two nearly parallel contours of ONE
        # white paint stripe. Do not display/count them as two lane lines.
        distinct=[]
        for candidate in good:
            a=np.asarray(candidate,dtype=float)
            duplicate=False
            for kept in distinct:
                b=np.asarray(kept,dtype=float)
                lower=max(a[:,1].min(),b[:,1].min())
                upper=min(a[:,1].max(),b[:,1].max())
                if upper-lower<.11*h:continue
                ys=np.linspace(lower,upper,12)
                d=np.abs(np.interp(ys,a[:,1],a[:,0])-np.interp(ys,b[:,1],b[:,0]))
                if float(np.median(d))<.065*w:
                    duplicate=True
                    break
            if not duplicate:distinct.append(candidate)
        good=distinct
        # Genuine new output; preserve backend traceability and strip any
        # pseudo-ego boundaries from the original empty baseline.
        lines=[(*a,*b) for pts in good[:3] for a,b in zip(pts,pts[1:])]
        output=LaneInfo(lane_lines=lines,lane_center_offset_px=None,
          polylines=good[:3],ego_boundaries=[],backend='yolopv2_visual_candidate',
          rejection_reason='unverified_BDD_mask_not_lane_scope_evidence')
        return output,True
