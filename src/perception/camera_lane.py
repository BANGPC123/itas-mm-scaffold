"""Camera-aware road-paint display. Never establishes legal lane scope.

Learned candidates are cropped to the observed road, checked against paint
pixels and vehicle occlusion, and refined only over their supported y range.
"""
from pathlib import Path
import cv2
import numpy as np
from src.utils.types import LaneInfo


def paint_evidence(image,road_probability,vehicle_boxes=(),paint_probability=None,ground_probability=None):
    h,w=image.shape[:2]
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hue,sat,value=cv2.split(hsv)
    contrast=cv2.morphologyEx(value,cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT,(max(21,round(w*.05))|1,1)))
    local=value.astype(np.float32)-cv2.GaussianBlur(value,(51,51),0).astype(np.float32)
    smooth=cv2.GaussianBlur(value,(5,5),0).astype(np.float32)
    radius=max(12,round(w*.026))
    padded=cv2.copyMakeBorder(smooth,0,0,radius,radius,cv2.BORDER_REPLICATE)
    peak=smooth-np.maximum(padded[:,:w],padded[:,2*radius:2*radius+w])
    # Both sides must be darker: one-sided vehicle shadows are not paint.
    white=(sat<95)&(value>150)&(peak>16)&((contrast>13)|(local>18))
    yellow=(hue>=10)&(hue<=40)&(sat>12)&(value>105)&(local>12)&(peak>8)
    if paint_probability is not None:
        # Faded white paint can be darker than a fixed daylight threshold.
        # Require both local pixel contrast and a strong independent paint head.
        white|=(sat<95)&(value>90)&(peak>8)&(local>8)&(paint_probability>.70)
    road=cv2.dilate((road_probability>.45).astype(np.uint8),np.ones((11,11),np.uint8))>0
    if paint_probability is not None:
        # The drivable head often classifies painted pixels as non-drivable.
        # Include the independent paint head, while requiring visible color.
        road|=paint_probability>.5
    mask=((white|yellow)&road).astype(np.uint8)
    if ground_probability is not None:
        # The semantic road head includes lane paint, unlike the drivable
        # head. Do not allow a bright dashboard/car edge to create support.
        ground=cv2.dilate((ground_probability>.55).astype(np.uint8),np.ones((5,5),np.uint8))
        mask[ground==0]=0
    if paint_probability is not None:
        strong_white=(sat<80)&(value>185)&(local>25)
        # The BDD-trained paint head can miss Vietnamese yellow dashes.
        # Keep clear yellow color/contrast on semantic road, with all occluder
        # and transverse-mark checks still applied below.
        strong_yellow=(hue>=10)&(hue<=40)&(sat>45)&(value>105)&(local>12)&(peak>8)
        mask[(paint_probability<.35)&~(strong_white|strong_yellow)]=0
    mask[:round(h*.30)]=0
    horizontal=cv2.morphologyEx(mask,cv2.MORPH_OPEN,
        np.ones((1,max(25,round(w*.05))),np.uint8))
    mask[horizontal>0]=0
    for x1,y1,x2,y2 in vehicle_boxes:
        mask[max(0,int(y1)):min(h,int(y2)+1),max(0,int(x1)):min(w,int(x2)+1)]=0
    # Transverse zebra stripes and road text are not longitudinal lane marks.
    n,ids,stats,_=cv2.connectedComponentsWithStats(mask,8)
    keep=np.zeros(n,np.uint8)
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if area>=6 and bh>=max(3,h*.004) and bw<=4.5*bh and bw<w*.10:
            keep[i]=1
    # Zebra stripes form a row of broad marks across the road. The individual
    # stripe direction can look longitudinal under perspective, so a slope
    # check alone cannot distinguish a crosswalk from a lane dash.
    broad=[i for i in range(1,n) if keep[i] and stats[i,2]>=w*.020 and stats[i,4]>=45]
    for i in broad:
        yc=stats[i,1]+stats[i,3]/2
        group=[j for j in broad if abs(stats[j,1]+stats[j,3]/2-yc)<h*.025]
        if len(group)>=3 and np.ptp([stats[j,0]+stats[j,2]/2 for j in group])>w*.22:
            keep[group]=0
    return keep[ids]


def refine_candidates(candidates,evidence,shape):
    h,w=shape[:2];accepted=[]
    radius=max(7,round(w*.028));stride=max(2,round(h/180))
    for line in candidates:
        xy=np.asarray(line,dtype=float)
        if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<5 or not np.isfinite(xy).all():continue
        xy=xy[np.argsort(xy[:,1])];ys,indices=np.unique(xy[:,1],return_index=True);xs=xy[indices,0]
        if len(ys)<5 or np.ptp(ys)<h*.05:continue
        if np.ptp(xs)>2*np.ptp(ys):continue
        hits=[];tested=0
        for yy in range(max(round(h*.30),int(ys[0])),min(h,int(ys[-1])+1),stride):
            xx=int(np.interp(yy,ys,xs));lo=max(0,xx-radius);hi=min(w,xx+radius+1)
            if lo>=hi:continue
            tested+=1;columns=np.flatnonzero(evidence[yy,lo:hi])
            if not len(columns):continue
            groups=np.split(columns,np.where(np.diff(columns)>1)[0]+1)
            centers=[float(np.median(g)+lo) for g in groups]
            hits.append((min(centers,key=lambda x:abs(x-xx)),yy))
        if len(hits)<6 or len(hits)/max(1,tested)<.12:continue
        a=np.asarray(hits);span=np.ptp(a[:,1])
        if span<h*.05:continue
        coeff=np.polyfit(a[:,1]/h,a[:,0]/w,2 if len(hits)>10 else 1)
        pred=np.polyval(coeff,a[:,1]/h)*w
        residual=np.abs(pred-a[:,0])
        if np.quantile(residual,.90)>w*.016:continue
        valid=residual<w*.020;a=a[valid]
        if len(a)<6 or np.ptp(a[:,1])<h*.05:continue
        coeff=np.polyfit(a[:,1]/h,a[:,0]/w,2 if len(a)>10 else 1)
        yy=np.arange(int(a[:,1].min()),int(a[:,1].max())+1,stride)
        xx=np.polyval(coeff,yy/h)*w
        if np.quantile(np.abs(np.diff(xx)/stride),.9)>2.5:continue
        curve=[(int(round(x)),int(y)) for x,y in zip(xx,yy) if 0<=x<w]
        if len(curve)<6:continue
        duplicate=False
        for existing in accepted:
            b=np.asarray(existing);lo=max(curve[0][1],b[0,1]);hi=min(curve[-1][1],b[-1,1])
            if hi-lo<h*.015:continue
            samples=np.linspace(lo,hi,10)
            distance=np.median(np.abs(np.interp(samples,yy,xx)-np.interp(samples,b[:,1],b[:,0])))
            if distance<w*.035:duplicate=True;break
        if not duplicate:accepted.append(curve)
    return accepted[:4]


def visible_paint_segments(polylines,evidence,*,min_points=2):
    """Draw supported paint pieces only; never bridge a car or empty road."""
    h,w=evidence.shape;radius=max(3,round(w*.007));segments=[]
    for line in polylines:
        current=[]
        for x,y in line:
            supported=(0<=y<h and w*.06<=x<=w*.94 and
                evidence[y,max(0,x-radius):min(w,x+radius+1)].any())
            if supported:current.append((x,y))
            else:
                if len(current)>=min_points:segments.append(current)
                current=[]
        if len(current)>=min_points:segments.append(current)
    return segments[:16]


def camera_view_from_ground(ground,image=None):
    h,w=ground.shape
    region=ground[round(h*.30):round(h*.75)]>.65
    yy,xx=np.where(region)
    if len(xx)<h*w*.01:return 'unknown'
    middle=region[:,round(w*.35):round(w*.65)].mean()
    center=float(np.median(xx))/w
    offcenter=(center<.29 or center>.76) and middle<.10
    if offcenter and np.quantile(yy,.10)+round(h*.30)>h*.60:
        # A low strip of road through a side window cannot establish a
        # forward view, even when bright reflections hide the dark cabin.
        return 'unknown'
    if not offcenter:return 'forward'
    if image is None:return 'unknown'
    # Traffic in front of the camera can hide the central asphalt. Require
    # a broad dark interior region as well; road displacement alone is not
    # evidence that the camera has turned toward the side window.
    left,right=(.60,.97) if center<.29 else (.03,.40)
    region=image[round(h*.08):round(h*.60),round(w*left):round(w*right)]
    value=cv2.cvtColor(region,cv2.COLOR_BGR2HSV)[:,:,2]
    dark=float(np.mean(value<55))
    return 'side_or_cabin' if dark>.50 else 'forward'


def observed_paint_seeds(evidence):
    """Short dashed paint can be invisible to models trained on long stripes."""
    h,w=evidence.shape
    # Align separated dashes using their component centers. A single small
    # white road symbol cannot establish a longitudinal paint hypothesis.
    n,ids,stats,centers=cv2.connectedComponentsWithStats(evidence,8)
    components=[i for i in range(1,n) if stats[i,4]>=6 and
        stats[i,3]>=max(3,h*.004) and stats[i,2]<=4.5*stats[i,3] and
        w*.06<centers[i,0]<w*.94]
    components=sorted(components,key=lambda i:stats[i,4],reverse=True)[:70]
    points=centers[components] if components else np.empty((0,2))
    aligned=[]
    for i,a in enumerate(points):
        for b in points[i+1:]:
            dy=b[1]-a[1]
            if not h*.025<=abs(dy)<=h*.35 or abs(b[0]-a[0])>1.8*abs(dy):continue
            slope=(b[0]-a[0])/dy
            distance=np.abs(points[:,0]-(a[0]+slope*(points[:,1]-a[1])))
            selected=points[distance<max(4,w*.006)]
            if len(selected)<3 or np.ptp(selected[:,1])<h*.065:continue
            coeff=np.polyfit(selected[:,1],selected[:,0],1)
            lo,hi=int(selected[:,1].min()-h*.012),int(selected[:,1].max()+h*.025)
            yy=np.arange(max(round(h*.30),lo),min(h,hi)+1,max(2,round(h/180)))
            line=[(round(x),int(y)) for x,y in zip(np.polyval(coeff,yy),yy)]
            if any(abs(np.polyval(coeff,h*.5)-v[1])<w*.016 for v in aligned):continue
            aligned.append((len(selected),float(np.polyval(coeff,h*.5)),line))
    seeds=[r[2] for r in sorted(aligned,key=lambda r:r[0],reverse=True)[:12]]
    segments=cv2.HoughLinesP(evidence*255,1,np.pi/180,threshold=18,
        minLineLength=max(20,round(h*.07)),maxLineGap=round(h*.09))
    if segments is None:return seeds
    for x1,y1,x2,y2 in segments[:,0]:
        if abs(y2-y1)<h*.06 or abs(x2-x1)>1.8*abs(y2-y1):continue
        lo,hi=sorted((int(y1),int(y2)));yy=np.arange(lo,hi+1,max(2,round(h/180)))
        xx=x1+(yy-y1)*(x2-x1)/(y2-y1)
        seeds.append([(int(round(x)),int(y)) for x,y in zip(xx,yy)])
    return sorted(seeds,key=lambda p:p[-1][1]-p[0][1],reverse=True)[:24]


class CameraLaneVisual:
    def __init__(self,ufld,device='cpu',surface_path=None):
        from src.perception.yolopv2_lane import YOLOPv2Lane
        root=Path(__file__).resolve().parents[2]
        self.ufld=ufld
        self.scene=YOLOPv2Lane(root/'models/lane/candidates/yolopv2_official_20261011/yolopv2.pt',device,roi=1)
        self.clr=None;self.failure=None
        try:
            from src.perception.clrnet_research import CLRNetResearch
            self.clr=CLRNetResearch(device=device)
        except (ImportError,RuntimeError,FileNotFoundError) as e:self.failure=str(e)[:100]
        from src.perception.lane_vehicle_occlusion import LaneObjectOcclusionGuard
        self.occlusion=LaneObjectOcclusionGuard(device=device)
        from src.perception.road_surface_veto import RoadSurfaceVeto
        self.surface=RoadSurfaceVeto(surface_path or root/'models/lane/candidates/segformer_cityscapes_20261011',device)
        self.uses_road_surface_mask=True

    def detect(self,image,baseline):
        h,w=image.shape[:2]
        from copy import deepcopy
        if w>1280:
            # Pixel processing is bounded; recognition still receives the
            # original frame. Map every lane coordinate back explicitly.
            scale=1280/w
            small=cv2.resize(image,(1280,round(h*scale)),interpolation=cv2.INTER_AREA)
            proposal=deepcopy(baseline)
            proposal.polylines=[[(round(x*scale),round(y*scale)) for x,y in p] for p in baseline.polylines]
            proposal.ego_boundaries=[]
            result=self.detect(small,proposal)
            result.polylines=[[(round(x/scale),round(y/scale)) for x,y in p] for p in result.polylines]
            result.lane_lines=[(*a,*b) for p in result.polylines for a,b in zip(p,p[1:])]
            result.display_polylines=result.polylines
            return result
        road,paint=self.scene.scene_probabilities(image)
        ground,sidewalk=self.surface.scene_probabilities(image)
        camera_view=camera_view_from_ground(ground,image)
        if camera_view!='forward':
            return LaneInfo([],backend='camera_road_paint_visual',
                rejection_reason='camera_direction_uncertain',camera_view=camera_view)
        # Disconnected semantic road islands on dashboard reflections are
        # not the visible carriageway. Keep the largest road component.
        n,ids,stats,_=cv2.connectedComponentsWithStats((ground>.55).astype(np.uint8),8)
        if n>1:
            main=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
            ground=np.where(ids==main,ground,0)
        yy,xx=np.where(ground[:,round(w*.15):round(w*.85)]>.65)
        if len(yy)<h*w*.01:
            return LaneInfo([],backend='camera_road_paint_visual',rejection_reason='road_not_visible',camera_view=camera_view)
        bottom=float(np.clip(np.quantile(yy,.99)/h+.01,.5,.97))
        crop=image[:round(h*bottom)]
        seeds=list(baseline.polylines)
        if bottom<.94 and self.ufld is not None:seeds+=self.ufld.detect(crop).polylines
        if self.clr is not None:seeds+=self.clr.detect(crop).polylines
        from src.perception.yolopv2_lane import traces_from_mask
        seeds+=traces_from_mask((paint>.56).astype(np.uint8),min_span_ratio=.09)
        boxes=self.occlusion.get_boxes(image)
        # A whole cabin/viewport COCO box is not a localized road occluder.
        boxes=[b for b in boxes if not ((b[2]-b[0])*(b[3]-b[1])>h*w*.40 and b[3]>h*.94)]
        evidence=paint_evidence(image,road,boxes,paint,ground)
        observed=observed_paint_seeds(evidence)
        lines=refine_candidates(observed+seeds,evidence,image.shape)
        from src.perception.road_surface_veto import filter_sidewalk_overlap
        lines,_=filter_sidewalk_overlap(lines,sidewalk)
        lines=visible_paint_segments(lines,evidence)
        from src.perception.crosswalk_veto import reject_crosswalk_segments
        lines=reject_crosswalk_segments(image,lines,ground_probability=ground)
        from src.perception.road_edge_veto import suppress_road_edge_contours
        lines,_=suppress_road_edge_contours(lines,ground)
        # No ego pair is created from display candidates or inferred road mask.
        return LaneInfo([(*a,*b) for p in lines for a,b in zip(p,p[1:])],
            polylines=lines,ego_boundaries=[],backend='camera_road_paint_visual',
            rejection_reason='visual_paint_not_verified_lane_scope' if lines else 'no_supported_road_paint',
            display_polylines=lines,camera_view=camera_view)
