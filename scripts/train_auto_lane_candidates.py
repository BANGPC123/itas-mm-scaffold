"""Bounded weak-label adaptation; candidates never replace deployed weights.

Separate head adaptation for UFLD/CLRNet, and partial-mask SegFormer training.
AI labels are explicitly weak, not human GT. External sources are excluded.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,random,time,gc,hashlib,cv2,numpy as np,torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from src.perception.ufld_v2 import UFLDv2
from src.perception.clrnet_research import CLRNetResearch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/retraining_auto_20261011'

def dataset():
 rows=[json.loads(l) for l in (ROOT/'data/processed/lane_auto_consensus_20261011/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
 train=[r for r in rows if r['eligible_for_weak_training'] and r['split']=='TRAIN']
 if len(train)<30:raise RuntimeError('Too few consensus training samples')
 assert not any(r['split'].startswith('TEST') for r in train)
 return train

def ufld_tensor(image,device):
 from PIL import Image
 a=np.array(Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).resize((1600,533),Image.Resampling.BILINEAR))[-320:].astype(np.float32)/255
 a=(a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
 return torch.from_numpy(a.transpose(2,0,1).copy()).unsqueeze(0).to(device)

def train_ufld(rows,dev,epochs):
 adapter=UFLDv2(ROOT/'models/lane/selected_20261009.pth',dev,half=False)
 m=adapter.model;m.eval()
 for p in m.parameters():p.requires_grad_(False)
 for p in m.cls[3].parameters():p.requires_grad_(True)
 opt=torch.optim.SGD(m.cls[3].parameters(),lr=1e-5,momentum=.9)
 scaler=torch.amp.GradScaler('cuda',enabled=dev.startswith('cuda'));history=[]
 for ep in range(epochs):
  losses=[]
  for r in rows:
   im=cv2.imread(r['image']);h,w=im.shape[:2]
   target=torch.full((1,72,4),-1,dtype=torch.long,device=dev)
   # Anchor IDs follow UFLD's two inner row heads. Only known coordinates
   # contribute to loss; missing paint never becomes a negative example.
   used=set()
   for l in sorted(r['lanes'],key=lambda a:a['points'][-1][0]):
    a=np.asarray(l['points'],float);ys,ix=np.unique(a[:,1],return_index=True);xs=a[ix,0]
    slot=1 if xs[-1]<w/2 else 2
    if slot in used:continue
    used.add(slot);ra=np.linspace(.42,1,72)*h;valid=(ra>=ys.min())&(ra<=ys.max())
    ids=np.flatnonzero(valid);v=np.interp(ra[valid],ys,xs)/w*199
    target[0,ids,slot]=torch.tensor(np.clip(v,0,199).astype('int64'),device=dev)
   if not (target>=0).any():continue
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',enabled=dev.startswith('cuda')):
    parts=m(ufld_tensor(im,dev));loss=F.cross_entropy(parts[0].float(),target,ignore_index=-1)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite UFLD loss')
   scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(m.cls[3].parameters(),1)
   scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
  history.append({'epoch':ep+1,'loss':float(np.mean(losses))});print('UFLD_AUTO',history[-1],flush=True)
 d=OUT/'ufld';d.mkdir(exist_ok=True)
 torch.save({'model':{k:v.detach().cpu() for k,v in m.state_dict().items()}},d/'candidate.pth')
 return {'method':'positive row-anchor weak adaptation; final projection only','history':history,'path':str(d/'candidate.pth')}

def train_clr(rows,dev,epochs):
 a=CLRNetResearch(device=dev);m=a.model;m.eval()
 for key in ['priors','priors_on_featmap']:
  setattr(m.heads,key,getattr(m.heads,key).detach())
 for p in m.parameters():p.requires_grad_(False)
 params=list(m.heads.reg_layers.parameters())+list(m.heads.cls_layers.parameters())
 for p in params:p.requires_grad_(True)
 opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01);history=[]
 for ep in range(epochs):
  losses=[]
  for r in rows:
   im=cv2.imread(r['image']);h,w=im.shape[:2]
   yy=(1-np.arange(72)/71)*(1-270/590)+270/590
   truth=[];valid=[]
   for l in r['lanes']:
    p=np.asarray(l['points'],float);ys,ix=np.unique(p[:,1]/h,return_index=True);xs=p[ix,0]/w
    truth.append(np.interp(yy,ys,xs));valid.append((yy>=ys.min())&(yy<=ys.max()))
   gt=torch.tensor(np.array(truth),dtype=torch.float32,device=dev)
   val=torch.tensor(np.array(valid),device=dev)
   if not val.any():continue
   opt.zero_grad(set_to_none=True);pred=m(a.tensor(im))[0]
   cost=(((pred[:,None,6:]-gt[None,:,:]).abs()*val[None,:,:]).sum(2)/val.sum(1).clamp(min=1)).detach().cpu().numpy()
   ii,jj=linear_sum_assignment(cost)
   ii=torch.tensor(ii,device=dev);jj=torch.tensor(jj,device=dev)
   loc=(pred[ii,6:]-gt[jj]).abs()[val[jj]].mean()
   # Partial labels do not license treating unmatched priors as background.
   cls=F.cross_entropy(pred[ii,:2],torch.ones(len(ii),dtype=torch.long,device=dev))
   loss=loc+.05*cls
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite CLR loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(params,1);opt.step();losses.append(float(loss.detach()))
  history.append({'epoch':ep+1,'loss':float(np.mean(losses))});print('CLR_AUTO',history[-1],flush=True)
 d=OUT/'clrnet';d.mkdir(exist_ok=True)
 torch.save({'net':{k:v.detach().cpu() for k,v in m.state_dict().items()}},d/'candidate.pth')
 return {'method':'positive-only Hungarian weak coordinate adaptation; final regression/classification projections','history':history,'path':str(d/'candidate.pth')}

def train_seg(rows,dev,epochs):
 from transformers import SegformerForSemanticSegmentation
 from scripts.train_segformer_verified_lanes_20261011 import DenseLanes,metrics
 from torch.utils.data import DataLoader
 model=SegformerForSemanticSegmentation.from_pretrained(str(ROOT/'output/retraining_20261011/segformer_culane_b0/best')).to(dev)
 dl=DataLoader(DenseLanes([(Path(r['image']),Path(r['mask'])) for r in rows],'vietnam'),batch_size=4,shuffle=True)
 opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.015);history=[]
 scaler=torch.amp.GradScaler('cuda',enabled=dev.startswith('cuda'))
 for ep in range(epochs):
  model.train();ls=[]
  for x,gt in dl:
   x=x.to(dev);gt=gt.to(dev);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',enabled=dev.startswith('cuda')):
    p=model(pixel_values=x).logits;p=F.interpolate(p.float(),size=gt.shape[-2:],mode='bilinear',align_corners=False)
    loss=F.cross_entropy(p,gt,weight=torch.tensor([.15,1.,.8],device=dev),ignore_index=255)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite SegFormer loss')
   scaler.scale(loss).backward();scaler.step(opt);scaler.update();ls.append(float(loss.detach()))
  history.append({'epoch':ep+1,'loss':float(np.mean(ls))});print('SEG_AUTO',history[-1],flush=True)
 d=OUT/'segformer';model.save_pretrained(str(d/'candidate'),safe_serialization=True)
 return {'method':'partial corridor-mask weak adaptation; unknown pixels ignored','history':history,'path':str(d/'candidate')}

def main():
 torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(391);random.seed(391);np.random.seed(391)
 OUT.mkdir(exist_ok=True)
 if (OUT/'report.json').exists():raise FileExistsError('Immutable experiment')
 rows=dataset();dev='cuda:0' if torch.cuda.is_available() else 'cpu';start=time.perf_counter()
 report={'training_samples':len(rows),'epochs':3,'human_ground_truth':False,'label_type':'frozen_teacher_consensus_partial','independent_accuracy':None,
 'external_training_samples':0,'production_model_changed':False,'models':{}}
 for name,fun in [('UFLD-v2',train_ufld),('CLRNet',train_clr),('SegFormer-B0',train_seg)]:
  partial=OUT/'report.partial.json'
  if partial.is_file():
   prior=json.loads(partial.read_text(encoding='utf-8'))
   report['models'].update(prior.get('models',{}))
  if name in report['models']:continue
  report['models'][name]=fun(rows,dev,3)
  gc.collect();torch.cuda.empty_cache()
  (OUT/'report.partial.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 report['final_session_seconds']=time.perf_counter()-start
 report['total_training_wall_time_not_measured']=True
 (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('TRAIN_AUTO_DONE',json.dumps(report),flush=True)
if __name__=='__main__':main()
