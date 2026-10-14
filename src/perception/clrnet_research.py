"""Opt-in CLRNet adapter. Does not change the deployed lane model."""
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from src.utils.types import LaneInfo
ROOT=Path(__file__).resolve().parents[2]

class CLRNetResearch:
    def __init__(self,path=None,device='cpu'):
        upstream=ROOT/'third_party/CLRNet_research'
        if not (upstream/'ITAS_PROVENANCE.json').is_file():
            raise RuntimeError('Run scripts/setup_clrnet_research.py first')
        sys.path.insert(0,str(ROOT/'.clr-runtime'));sys.path.insert(0,str(upstream))
        from clrnet.utils.config import Config
        from clrnet.models.registry import build_net
        import clrnet.models
        import runpy
        self.cfg=Config({k:v for k,v in runpy.run_path(str(upstream/'configs/clrnet/clr_resnet18_culane.py')).items() if not k.startswith('__')})
        self.cfg.backbone.pretrained=False
        self.device=device
        self.model=build_net(self.cfg)
        path=Path(path or ROOT/'models/lane/candidates/clrnet_official/culane_r18.pth')
        state=torch.load(path,map_location='cpu',weights_only=True)
        state=state.get('net',state.get('state_dict',state.get('model',state)))
        state={k.removeprefix('module.'):v for k,v in state.items()}
        expected=self.model.state_dict()
        derived={'heads.sample_x_indexs','heads.prior_feat_ys','heads.prior_ys','heads.criterion.weight'}
        missing=set(expected)-set(state)
        if not missing <= derived:raise ValueError('Unexpected missing checkpoint tensors: '+str(missing))
        state.update({k:expected[k] for k in missing})
        self.model.load_state_dict(state,strict=True)
        self.model.to(device).eval()
    def tensor(self,image):
        h,w=image.shape[:2];cut=int(h*270/590)
        a=cv2.resize(image[cut:],(800,320),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255
        return torch.from_numpy(a.transpose(2,0,1).copy()).unsqueeze(0).to(self.device)
    def detect(self,image):
        h,w=image.shape[:2]
        with torch.inference_mode():
            out=self.model(self.tensor(image));lanes=self.model.heads.get_lanes(out)[0]
        lines=[]
        for ln in lanes:
            pts=np.asarray(ln.points)
            if not np.isfinite(pts).all():continue
            valid=(pts[:,0]>=0)&(pts[:,0]<1)&(pts[:,1]>=0)&(pts[:,1]<1)
            p=[(int(x*w),int(y*h)) for x,y in pts[valid]]
            if len(p)>=3:lines.append(p)
        return LaneInfo([(*a,*b) for p in lines for a,b in zip(p,p[1:])],polylines=lines,backend='clrnet_research_cpu_nms')
