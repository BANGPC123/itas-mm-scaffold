"""Bounded, per-stream sign evidence; no carry-over between video sessions."""
from dataclasses import dataclass
import math
import time
from src.perception.image_quality import crop_quality, insufficient_information


def overlap(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    area=max(0,x2-x1)*max(0,y2-y1)
    return area/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def nearby_same_sign(sign,track,timestamp):
    if sign.label=='unknown' or sign.recognition_confidence is None or sign.label!=track.label or timestamp-track.seen>.35:
        return False
    a,b=sign.bbox_xyxy,track.box
    aw,ah=a[2]-a[0],a[3]-a[1];bw,bh=b[2]-b[0],b[3]-b[1]
    if min(aw,ah,bw,bh)<=0 or not .5<=aw*ah/(bw*bh)<=2:
        return False
    distance=math.hypot((a[0]+a[2]-b[0]-b[2])/2,(a[1]+a[3]-b[1]-b[3])/2)
    return distance<=.8*math.hypot(bw,bh)


@dataclass
class Evidence:
    identifier:int
    box:tuple
    first:float
    seen:float
    label:str='unknown'
    votes:int=0
    quality:float=0.
    confidence:float|None=None
    best_time:float=0.
    confirmed:float|None=None


class TemporalSigns:
    def __init__(self,max_age=.75,max_tracks=64):
        self.max_age=max_age;self.max_tracks=max_tracks
        self.tracks=[];self.last=None;self.shape=None;self.next_id=1

    def update(self,image,signs,timestamp):
        if not math.isfinite(timestamp) or timestamp<0:raise ValueError('Invalid frame timestamp')
        signs=[s for s in signs if s.recognition_source!='background_rejected']
        if self.last is not None and (timestamp<=self.last or timestamp-self.last>self.max_age or self.shape!=image.shape):
            self.tracks=[]
        self.last=timestamp;self.shape=image.shape
        if insufficient_information(image):
            self.tracks=[]
            for sign in signs:sign.label='unknown';sign.recognition_confidence=None
            return signs
        old=[t for t in self.tracks if timestamp-t.seen<=self.max_age]
        matches=[[i for i,t in enumerate(old) if overlap(s.bbox_xyxy,t.box)>=.3] for s in signs]
        reserved={i for candidates in matches for i in candidates}
        for index,sign in enumerate(signs):
            if not matches[index]:
                matches[index]=[i for i,t in enumerate(old) if i not in reserved and nearby_same_sign(sign,t,timestamp)]
        use_count=[sum(i in candidates for candidates in matches) for i in range(len(old))]
        current=[]
        for index,sign in enumerate(signs):
            hard_veto=sign.recognition_source in ('symbol_shape_unsupported','outside_supported_reference')
            sign.recognition_source='temporal_pending'
            if index>=self.max_tracks:
                sign.label='unknown';sign.recognition_confidence=None;continue
            candidates=matches[index]
            track=old[candidates[0]] if len(candidates)==1 and use_count[candidates[0]]==1 else None
            if track is None:
                track=Evidence(self.next_id,sign.bbox_xyxy,timestamp,timestamp);self.next_id+=1
            track.box=sign.bbox_xyxy;track.seen=timestamp
            if hard_veto:
                track.label='unknown';track.votes=0;track.quality=0.;track.confidence=None;track.confirmed=None
            if timestamp-track.best_time>self.max_age and sign.label=='unknown':
                track.label='unknown';track.votes=0;track.quality=0.;track.confirmed=None
            if sign.label!='unknown' and sign.recognition_confidence is not None:
                if sign.label!=track.label:
                    track.label=sign.label;track.votes=0;track.quality=0.;track.confirmed=None
                track.votes+=1
                x1,y1,x2,y2=sign.bbox_xyxy;h,w=image.shape[:2]
                crop=image[max(0,math.floor(y1)):min(h,math.ceil(y2)),max(0,math.floor(x1)):min(w,math.ceil(x2))]
                quality=crop_quality(crop)
                if quality>=track.quality or timestamp-track.best_time>self.max_age:
                    track.quality=quality;track.confidence=sign.recognition_confidence;track.best_time=timestamp
                if track.votes>=2 and track.confirmed is None:track.confirmed=timestamp
            sign.track_id=track.identifier
            if track.confirmed is not None and timestamp-track.best_time<=self.max_age:
                sign.label=track.label;sign.recognition_confidence=track.confidence
                sign.recognition_source='temporal_best_crop'
                sign.crop_age_ms=(timestamp-track.best_time)*1000
                sign.confirmation_delay_ms=(track.confirmed-track.first)*1000
            else:
                sign.label='unknown';sign.recognition_confidence=None
            current.append(track)
        # A missing detection terminates the track; old boxes are never emitted.
        self.tracks=current
        return signs


class StreamEvidence:
    def __init__(self):self.sessions={}

    def get(self,key,shape,road_bottom):
        now=time.monotonic()
        self.sessions={k:v for k,v in self.sessions.items() if now-v[0]<15}
        context=(shape,road_bottom)
        if key not in self.sessions or self.sessions[key][1]!=context:
            if len(self.sessions)>=8:
                del self.sessions[min(self.sessions,key=lambda k:self.sessions[k][0])]
            self.sessions[key]=(now,context,TemporalSigns())
        else:self.sessions[key]=(now,context,self.sessions[key][2])
        return self.sessions[key][2]
