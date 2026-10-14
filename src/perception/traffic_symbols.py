"""Pixel evidence for supported QCVN symbols; not reviewed legal scope.

Templates come from the official regulation, never the user's test frames.
Ambiguous shapes abstain rather than assigning a new primary sign class.
"""
from functools import lru_cache
from pathlib import Path
import base64,json,cv2,numpy as np


@lru_cache(maxsize=1)
def templates():
    path=Path(__file__).resolve().parents[2]/'data/reference/qcvn2024_symbols.json'
    doc=json.loads(path.read_text(encoding='utf-8'))
    return {k:cv2.imdecode(np.frombuffer(base64.b64decode(v['mask_png_base64']),np.uint8),0)>0
            for k,v in doc['templates'].items()}


def shape_score(mask,reference):
    """Normalized silhouette Dice, tolerant to pixel loss and aspect distortion."""
    if not mask.any():return 0.
    yy,xx=np.where(mask);mask=mask[yy.min():yy.max()+1,xx.min():xx.max()+1]
    a=cv2.resize(mask.astype(np.uint8),(48,48),interpolation=cv2.INTER_NEAREST)>0
    b=cv2.resize(reference.astype(np.uint8),(48,48),interpolation=cv2.INTER_NEAREST)>0
    scores=[]
    for v in (b,np.fliplr(b)):
        for dx in (-2,0,2):
            for dy in (-2,0,2):
                shifted=cv2.warpAffine(v.astype(np.uint8),np.float32([[1,0,dx],[0,1,dy]]),(48,48))>0
                scores.append(2*np.count_nonzero(a&shifted)/max(1,a.sum()+shifted.sum()))
    return float(max(scores))


def blue_symbol_verdict(crop,code,*,minimum=.53):
    if code!='I.409' or min(crop.shape[:2])<20:return None
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);h,w=crop.shape[:2]
    blue=(hsv[:,:,0]>=85)&(hsv[:,:,0]<=135)&(hsv[:,:,1]>65)&(hsv[:,:,2]>50)
    if blue.mean()<.30:return None
    # Exclude the reflective border: a U-turn symbol lies inside the blue face.
    white=((hsv[:,:,1]<100)&(hsv[:,:,2]>max(100,float(np.median(hsv[:,:,2][blue]))+25))).astype(np.uint8)
    white[:max(1,round(h*.07))]=0;white[-max(1,round(h*.07)):]=0
    white[:,:max(1,round(w*.07))]=0;white[:,-max(1,round(w*.07)):]=0
    n,ids,stats,_=cv2.connectedComponentsWithStats(white,8)
    eligible=[i for i in range(1,n) if stats[i,4]>=max(8,h*w*.018)]
    if not eligible:return 'symbol_shape_unsupported'
    i=max(eligible,key=lambda i:stats[i,4]);m=ids==i
    return None if shape_score(m,templates()['I.409'])>=minimum else 'symbol_shape_unsupported'


def read_vehicle_icon(crop):
    """Recognize a clear truck/car-front glyph on a light supplementary plate.

    Direction arrows are not vehicle classes. Small or competing silhouettes
    stay unresolved; OCR is needed to detect textual exclusions separately.
    """
    if crop.size==0 or min(crop.shape[:2])<12:return None
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);h,w=gray.shape
    votes=[]
    for limit in (85,110):
        m=(gray<limit).astype(np.uint8)
        n,ids,stats,_=cv2.connectedComponentsWithStats(m,8);choices=[]
        for i in range(1,n):
            x,y,bw,bh,area=map(int,stats[i])
            if area<12 or bh<6 or bw<8 or bw/bh<.65 or bw/bh>3.1:continue
            if x<=0 or y<=0 or x+bw>=w or y+bh>=h:continue
            if area/(bw*bh)<.22:continue
            mask=ids[y:y+bh,x:x+bw]==i
            scores={k:shape_score(mask,v) for k,v in templates().items() if k in ('truck','car_front')}
            label=max(scores,key=scores.get);runner=min(scores.values())
            if scores[label]>=.78 and scores[label]-runner>=.09:
                choices.append((scores[label],label,[x,y,x+bw,y+bh]))
        votes.append(max(choices) if choices else None)
    if any(v is None for v in votes) or votes[0][1]!=votes[1][1]:return None
    return {'vehicle':'car' if votes[0][1]=='car_front' else 'truck',
            'confidence':min(v[0] for v in votes),'variant_agreement':True,
            'source':'qcvn2024_symbol_template','human_verified':False}
