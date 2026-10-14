"""CULane ResNet18 inference adapter for cfzd/Ultra-Fast-Lane-Detection-v2.

Architecture and local-expectation decoding follow the author's MIT-licensed
model/model_culane.py and demo.py. See third_party/UFLD-v2-LICENSE.
"""
import cv2
import numpy as np
import torch
import zipfile
from torch import nn
from torchvision.models import resnet18
from PIL import Image
from src.utils.types import LaneInfo
from src.perception.image_quality import insufficient_information


class Backbone(nn.Module):
    def __init__(self):
        super().__init__();m=resnet18(weights=None)
        for key in ['conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4']:
            setattr(self,key,getattr(m,key))
    def forward(self,x):
        for key in ['conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4']:
            x=getattr(self,key)(x)
        return x


class UFLDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=Backbone()
        self.pool=nn.Conv2d(512,8,1)
        self.dims=[200*72*4,100*81*4,2*72*4,2*81*4]
        self.cls=nn.Sequential(nn.LayerNorm(4000),nn.Linear(4000,2048),nn.ReLU(),nn.Linear(2048,sum(self.dims)))
    def forward(self,x):
        out=self.cls(self.pool(self.model(x)).flatten(1))
        parts=out.split(self.dims,dim=1)
        return [parts[0].reshape(1,200,72,4),parts[1].reshape(1,100,81,4),parts[2].reshape(1,2,72,4),parts[3].reshape(1,2,81,4)]


def local_expectation(logits):
    """Decode all anchor/lane coordinates before transferring them off the GPU."""
    grids = logits.shape[1]
    maxima = logits.argmax(1, keepdim=True)
    cells = maxima + torch.arange(-1, 2, device=logits.device).view(1, 3, 1, 1)
    valid = (cells >= 0) & (cells < grids)
    values = logits.gather(1, cells.clamp(0, grids-1)).float().masked_fill(~valid, -torch.inf)
    return ((values.softmax(1)*cells).sum(1)[0]+.5)/(grids-1)


def plausible_lane(points, width, height):
    """Reject short/horizontal zigzags; this is a geometry check, not lane truth."""
    if len(points) < 3:
        return False
    xy = np.asarray(points, dtype=float)
    vertical_span = np.ptp(xy[:, 1])
    if vertical_span < .18*height or np.ptp(xy[:, 0]) > 2.5*vertical_span:
        return False
    if np.abs(np.diff(xy[:, 1])).sum() > 1.3*vertical_span:
        return False
    return True


class UFLDv2:
    def __init__(self,path,device='cpu',half=False):
        self.device=device
        # The 825 MB checkpoint must not coexist with an initialized CPU copy.
        with torch.device('meta'):
            self.model=UFLDNet()
        checkpoint=torch.load(path,map_location='cpu',weights_only=True,mmap=zipfile.is_zipfile(path))
        state=checkpoint['model'] if 'model' in checkpoint else checkpoint
        state={k.removeprefix('module.'):v for k,v in state.items()}
        self.model.load_state_dict(state,strict=True,assign=True)
        self.dtype = torch.float16 if half and str(device).startswith('cuda') else torch.float32
        self.model.to(device=device,dtype=self.dtype).eval()
        self.mean = torch.tensor([.485,.456,.406], device=device, dtype=self.dtype)[:,None,None]
        self.std = torch.tensor([.229,.224,.225], device=device, dtype=self.dtype)[:,None,None]

    def detect(self,image,*,filter_geometry=True):
        if image is None or image.size==0:return LaneInfo([],backend='ufld_v2')
        if insufficient_information(image):
            return LaneInfo([],backend='ufld_v2',rejection_reason='insufficient_image_information')
        h,w=image.shape[:2]
        rgb=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        rgb=np.array(rgb.resize((1600,int(320/.6)),Image.Resampling.BILINEAR))[-320:]
        x=torch.from_numpy(rgb.copy()).permute(2,0,1).to(device=self.device,dtype=self.dtype)/255
        x=(x-self.mean)/self.std
        with torch.inference_mode():
            pred = self.model(x.unsqueeze(0))
            coordinates = [local_expectation(v).cpu().numpy() for v in pred[:2]]
            existence = [v.argmax(1)[0].cpu().numpy() for v in pred[2:]]
        polylines=[]
        for coords,valid,indices,anchors,scale,other_scale,isrow in [
            (coordinates[0],existence[0],[1,2],np.linspace(.42,1,72),w,h,True),
            (coordinates[1],existence[1],[0,3],np.linspace(0,1,81),h,w,False)]:
            for lane in indices:
                if int(valid[:,lane].sum())<=len(anchors)/(2 if isrow else 4):continue
                points=[]
                for k,anchor in enumerate(anchors):
                    if not valid[k,lane]:continue
                    value=float(coords[k,lane])*scale
                    point=(int(value),int(anchor*other_scale)) if isrow else (int(anchor*other_scale),int(value))
                    points.append((max(0,min(w-1,point[0])),max(0,min(h-1,point[1]))))
                if len(points)>=3 and (not filter_geometry or plausible_lane(points,w,h)):polylines.append(points)
        yref=h*.9;candidates=[]
        for points in polylines:
            ordered=sorted(points,key=lambda p:p[1]);ys=[p[1] for p in ordered]
            if min(ys)<=yref<=max(ys):candidates.append((float(np.interp(yref,ys,[p[0] for p in ordered])),points))
        left=[v for v in candidates if v[0]<w/2];right=[v for v in candidates if v[0]>w/2]
        ego=[];offset=None
        if left and right:
            l=max(left,key=lambda v:v[0]);r=min(right,key=lambda v:v[0]);ego=[l[1],r[1]];offset=(l[0]+r[0])/2-w/2
        lines=[(*p1,*p2) for points in polylines for p1,p2 in zip(points,points[1:])]
        return LaneInfo(lines,offset,polylines,ego,'ufld_v2')
