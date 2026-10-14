"""Automatic weak labels: frozen UFLD/CLRNet agreement plus paint support.

Never populates human_confirmed, never turns a model-empty frame into a GT
negative, never admits source-disjoint external frames to training.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,hashlib,time,cv2,numpy as np,torch
from scipy.optimize import linear_sum_assignment
from src.perception.ufld_v2 import UFLDv2
from src.perception.clrnet_research import CLRNetResearch
ROOT=Path(__file__).resolve().parents[1]

def sampled(line):
 a=np.asarray(line,float);a=a[np.argsort(a[:,1])]
 ys,ix=np.unique(a[:,1],return_index=True)
 return ys,a[ix,0]

def consensus(image,left,right,tolerance=.022):
 h,w=image.shape[:2];cost=np.full((len(left),len(right)),999.)
 pairs={}
 gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
 for i,a in enumerate(left):
  ay,ax=sampled(a)
  for j,b in enumerate(right):
   by,bx=sampled(b);lo=max(ay.min(),by.min(),h*.45);hi=min(ay.max(),by.max(),h*.91)
   if hi-lo<h*.16:continue
   ys=np.linspace(lo,hi,32);xs1=np.interp(ys,ay,ax);xs2=np.interp(ys,by,bx)
   d=np.abs(xs1-xs2)/w
   if np.median(d)>tolerance or np.percentile(d,90)>2*tolerance:continue
   xs=(xs1+xs2)/2
   support=[]
   for x,y in zip(xs,ys):
    x=int(x);y=int(y);rad=max(2,round(w*.006));side=max(rad+1,round(w*.016))
    if not side<=x<w-side or not 2<=y<h-2:continue
    center=gray[y-2:y+3,x-rad:x+rad+1].max(axis=1).mean()
    bg=np.median(np.r_[gray[y-2:y+3,x-side:x-rad].flatten(),gray[y-2:y+3,x+rad+1:x+side].flatten()])
    support.append(center>105 and center-bg>12)
   paint=float(np.mean(support)) if support else 0
   if paint<.25:continue
   cost[i,j]=float(np.median(d));pairs[i,j]=([(int(x),int(y)) for x,y in zip(xs,ys)],paint)
 if not len(left) or not len(right):return []
 ii,jj=linear_sum_assignment(cost)
 return [{'points':pairs[i,j][0],'paint_support':pairs[i,j][1],'median_distance_ratio':cost[i,j],
          'source':'frozen_UFLDv2_CLRNet_consensus','human_confirmed':False} for i,j in zip(ii,jj) if (i,j) in pairs]

def mask_from_partial(image,lanes):
 h,w=image.shape[:2];mask=np.full((h,w),255,np.uint8)
 # Only a local corridor is labeled as background. All unknown road pixels
 # remain ignored, including unmatched lanes and every empty frame.
 for ln in lanes:
  p=np.asarray(ln['points'],np.int32)
  cv2.polylines(mask,[p],False,0,max(16,round(w*.055)))
 for ln in lanes:
  cv2.polylines(mask,[np.asarray(ln['points'],np.int32)],False,1,max(4,round(w*.010)))
 return mask

def main():
 torch.set_num_threads(4);cv2.setNumThreads(1)
 out=ROOT/'data/processed/lane_auto_consensus_20261011'
 if (out/'report.json').exists():raise FileExistsError('Immutable completed experiment')
 out.mkdir(parents=True,exist_ok=True);(out/'masks').mkdir(exist_ok=True)
 dev='cuda:0' if torch.cuda.is_available() else 'cpu'
 u=UFLDv2(ROOT/'models/lane/selected_20261009.pth',dev,half=True);c=CLRNetResearch(device=dev)
 start=time.perf_counter();rows=[];counts={}
 for packname in ['lane_vietnam_review_20261011','lane_external_source_review_20261011']:
  pack=ROOT/'data'/packname;meta=json.loads((pack/'manifest.json').read_text(encoding='utf-8'))
  for f in meta['frames']:
   p=pack/f['image'];im=cv2.imread(str(p))
   if im is None or hashlib.sha256(p.read_bytes()).hexdigest()!=f['sha256']:raise ValueError('Image integrity')
   a=u.detect(im).polylines;b=c.detect(im).polylines;ls=consensus(im,a,b)
   mask=out/'masks'/(f['id']+'.png');cv2.imwrite(str(mask),mask_from_partial(im,ls))
   row={'id':f['id'],'image':str(p.resolve()),'image_sha256':f['sha256'],'source_video_id':f['source_video_id'],
     'split':f['split'],'temporal_group':f['temporal_group'],'mask':str(mask.resolve()),'lanes':ls,
     'ufld_lines':a,'clrnet_lines':b,'human_confirmed':False,'label_provenance':'machine_consensus_partial',
     'eligible_for_weak_training':f['split']=='TRAIN' and bool(ls),'exhaustive':False}
   rows.append(row);counts[f['split']]=counts.get(f['split'],0)+int(bool(ls))
   if len(rows)%50==0:print('AUTO_LANES',len(rows),'positive_frames',sum(bool(r['lanes']) for r in rows),flush=True)
 (out/'manifest.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
 report={'schema_version':'itasmm-weak-lane-v1','frames':len(rows),'frames_with_consensus':sum(bool(r['lanes']) for r in rows),
  'positive_by_split':counts,'training_eligible':sum(r['eligible_for_weak_training'] for r in rows),
  'external_sources_training_frames':0,'human_verified_frames':0,'empty_frames_labeled_negative':0,
  'independent_accuracy_available':False,'mask_classes':{'0':'local corridor background','1':'consensus paint','255':'unknown ignored'},
  'teacher_weights_modified':False,'elapsed_seconds':time.perf_counter()-start}
 (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('AUTO_CORPUS_DONE',json.dumps(report),flush=True)
if __name__=='__main__':main()
