"""Road-surface veto for display candidates; never establishes lane scope."""
from pathlib import Path
import cv2,numpy as np

def filter_sidewalk_overlap(polylines,sidewalk_probability,*,fraction=.35):
    h,w=sidewalk_probability.shape;kept=[];removed=0
    for line in polylines:
        points=np.asarray(line,dtype=float)
        if len(points)<3 or not np.isfinite(points).all():removed+=1;continue
        x=np.clip(np.rint(points[:,0]).astype(int),0,w-1)
        y=np.clip(np.rint(points[:,1]).astype(int),0,h-1)
        if np.mean(sidewalk_probability[y,x]>.5)>fraction:removed+=1
        else:kept.append(line)
    return kept,removed

class RoadSurfaceVeto:
    def __init__(self,path,device='cpu'):
        from transformers import SegformerForSemanticSegmentation
        self.device=device
        self.model=SegformerForSemanticSegmentation.from_pretrained(str(Path(path)),local_files_only=True).to(device).eval()
        if self.model.config.id2label[0]!='road' or self.model.config.id2label[1]!='sidewalk':
            raise ValueError('Unexpected semantic class mapping')

    def scene_probabilities(self,image):
        import torch
        h,w=image.shape[:2];height=max(160,min(512,round(h*512/w/32)*32))
        rgb=cv2.cvtColor(cv2.resize(image,(512,height)),cv2.COLOR_BGR2RGB).astype(np.float32)/255
        normalized=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        x=torch.from_numpy(normalized.transpose(2,0,1).copy())[None].to(self.device)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=str(self.device).startswith('cuda')):
            logits=self.model(pixel_values=x).logits
        probabilities=logits.float().softmax(1)[0,:2].cpu().numpy()
        return tuple(cv2.resize(value,(w,h)) for value in probabilities)

    def filter(self,image,polylines):
        if not polylines:return [],0
        _,sidewalk=self.scene_probabilities(image)
        return filter_sidewalk_overlap(polylines,sidewalk)
