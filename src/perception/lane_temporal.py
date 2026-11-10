"""Small temporal lane-display smoother, not a lane detector or legal scope engine.

Only smooth polylines present in BOTH consecutive frames. Never extrapolate
missing lanes, never alter raw lane coordinates used by downstream reasoning.
"""
from __future__ import annotations
from collections import OrderedDict
import math
import time
import numpy as np


def _curve_match(current, previous, width):
    a=np.asarray(current,dtype=float)
    b=np.asarray(previous,dtype=float)
    if len(a)<3 or len(b)<3:return None
    a=a[np.argsort(a[:,1])];b=b[np.argsort(b[:,1])]
    lo=max(a[0,1],b[0,1]);hi=min(a[-1,1],b[-1,1])
    if hi-lo < .25*min(np.ptp(a[:,1]),np.ptp(b[:,1])):return None
    ys=np.linspace(lo,hi,10)
    dist=np.mean(np.abs(np.interp(ys,a[:,1],a[:,0])-np.interp(ys,b[:,1],b[:,0])))
    return float(dist) if dist<.065*width else None


class LaneStabilizer:
    def __init__(self,alpha=.65):
        self.alpha=alpha
        self.previous=[]
        self.last=None
        self.shape=None
        self.scene_descriptor=None

    def update(self,polylines,timestamp,shape,image=None):
        if not math.isfinite(timestamp) or timestamp<0:raise ValueError('Bad timestamp')
        scene=None
        if image is not None:
            import cv2
            # Scene change detector is a temporal reset, not a lane classifier.
            scene=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(32,18)).astype(float)/255.0
        hard_cut=(scene is not None and self.scene_descriptor is not None and
                  float(np.mean(np.abs(scene-self.scene_descriptor)))>.24)
        if self.last is None or timestamp<=self.last or timestamp-self.last>.6 or self.shape!=shape or hard_cut:
            self.previous=[]
        self.scene_descriptor=scene
        self.last=timestamp
        self.shape=shape
        w=shape[1]
        previous=self.previous
        output=[]
        used=set()
        for current in polylines:
            options=sorted((distance,i) for i,older in enumerate(previous)
                if i not in used and (distance:=_curve_match(current,older,w)) is not None)
            choice=options[0][1] if len(options)==1 or (len(options)>1 and options[1][0]-options[0][0]>.015*w) else None
            if choice is None:
                output.append(list(current))
                continue
            used.add(choice)
            old=np.asarray(previous[choice],dtype=float)
            old=old[np.argsort(old[:,1])]
            lo=max(min(y for x,y in current),old[0,1]);hi=min(max(y for x,y in current),old[-1,1])
            taper=max(1.,(hi-lo)*.15)
            result=[]
            for x,y in current:
                if old[0,1]<=y<=old[-1,1]:
                    weight=self.alpha*min(1.,max(0.,(y-lo)/taper),max(0.,(hi-y)/taper))
                    x=int(round(weight*np.interp(y,old[:,1],old[:,0])+(1-weight)*x))
                result.append((x,y))
            output.append(result)
        # Must never produce ghost lanes when detector returns none.
        self.previous=[list(line) for line in output]
        return output


class LaneStreamPool:
    def __init__(self,max_sessions=8):
        self.max_sessions=max_sessions
        self.sessions=OrderedDict()

    def get(self,key,shape,roi):
        now=time.monotonic()
        self.sessions=OrderedDict((k,v) for k,v in self.sessions.items() if now-v[0]<15)
        context=(shape,roi)
        if key in self.sessions and self.sessions[key][1]==context:
            st=self.sessions.pop(key)[2]
        else:
            st=LaneStabilizer()
        self.sessions[key]=(now,context,st)
        while len(self.sessions)>self.max_sessions:
            self.sessions.popitem(last=False)
        return st

class OfflineLaneEvidence:
    """Show *current-frame* UFLD proposals only after consecutive agreement.

    Never emits missing-frame ghosts. It is solely for an amber annotation
    channel and is never connected to ego boundaries or sign applicability.
    """
    def __init__(self,min_hits=2,max_age=.5,max_lines=8):
        if min_hits<2:raise ValueError('Need at least two independent frames')
        self.min_hits=min_hits;self.max_age=max_age;self.max_lines=max_lines
        self.last=None;self.shape=None;self.scene=None;self.tracks=[]

    def update(self,polylines,timestamp,shape,image=None):
        if not math.isfinite(timestamp) or timestamp<0:
            raise ValueError('Bad timestamp')
        scene=None
        if image is not None:
            import cv2
            scene=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),
                              (32,18)).astype(float)/255.
        cut=(scene is not None and self.scene is not None and
             float(np.abs(scene-self.scene).mean())>.24)
        if (self.last is None or timestamp<=self.last or
            timestamp-self.last>self.max_age or shape!=self.shape or cut):
            self.tracks=[]
        self.last=timestamp;self.shape=shape;self.scene=scene
        old=self.tracks
        new=[];output=[];used=set()
        for poly in list(polylines)[:self.max_lines*2]:
            options=sorted((distance,i) for i,(last_poly,votes) in enumerate(old)
                if i not in used and
                (distance:=_curve_match(poly,last_poly,shape[1])) is not None)
            if len(options)>1 and options[1][0]-options[0][0]<=.015*shape[1]:
                options=[]
            if options:
                _,idx=options[0]
                used.add(idx)
                votes=old[idx][1]+1
            else:votes=1
            new.append((list(poly),min(votes,self.min_hits)))
            if votes>=self.min_hits:output.append(list(poly))
            if len(new)>=self.max_lines:break
        self.tracks=new
        return output


def offline_visual_fallback_allowed(lane):
    """Fail closed if independent road/paint fusion rejects a proposed lane.

    In particular, UFLD orange contours through dashboards and vehicles must
    not be resurrected after camera-aware paint verification rejects them.
    """
    return (lane.backend=='ufld_v2' and lane.camera_view=='forward'
            and not lane.polylines and not lane.ego_boundaries)
