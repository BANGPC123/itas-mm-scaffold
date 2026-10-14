"""Paired regression check on fixed CULane VAL and source-disjoint video frames.

Pixel metrics use the same original-size mask for all six model variants.
External Vietnam data has no GT: only predictions/latency are reported.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time,hashlib,gc,cv2,numpy as np,torch
from src.perception.ufld_v2 import UFLDv2
from src.perception.clrnet_research import CLRNetResearch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/evaluation/auto_lane_comparison_20261011'

def main():
 torch.set_num_threads(4);cv2.setNumThreads(1)
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'report.json').exists():raise FileExistsError('Immutable report')
 dev='cuda:0' if torch.cuda.is_available() else 'cpu'
 names=[s.split()[0].lstrip('/') for s in (ROOT/'data/raw/culane/test/list/val.txt').read_text().splitlines() if s.strip()]
 names=sorted(names,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:100]
 external=[json.loads(l) for l in (ROOT/'data/processed/lane_auto_consensus_20261011/manifest.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
 external=[r for r in external if r['split']=='TEST_EXTERNAL_SOURCE_CANDIDATE']
 models=[('UFLD-v2','before',ROOT/'models/lane/selected_20261009.pth'),('UFLD-v2','after',ROOT/'output/retraining_auto_20261011/ufld/candidate.pth'),
 ('CLRNet','before',None),('CLRNet','after',ROOT/'output/retraining_auto_20261011/clrnet/candidate.pth'),
 ('SegFormer-B0','before',ROOT/'output/retraining_20261011/segformer_culane_b0/best'),('SegFormer-B0','after',ROOT/'output/retraining_auto_20261011/segformer/candidate')]
 report={'dataset':'CULane official VAL labels, SHA-selected 100 images','vietnam_independent_gt':False,'metric':'binary paint pixel F1 with original-size 16-pixel GT raster',
 'same_GT_all_models':True,'validation_used_for_selection':False,'production_model_changed':False,'training_labels':'54 automatic partial consensus frames','models':{}}
 for kind,phase,path in models:
  if kind=='UFLD-v2':model=UFLDv2(path,dev,half=True)
  elif kind=='CLRNet':model=CLRNetResearch(path,dev)
  else:
   from transformers import SegformerForSemanticSegmentation
   model=SegformerForSemanticSegmentation.from_pretrained(str(path)).to(dev).eval()
  def predict(im):
   h,w=im.shape[:2]
   if kind in ('UFLD-v2','CLRNet'):
    lane=model.detect(im,filter_geometry=False) if kind=='UFLD-v2' else model.detect(im)
    mask=np.zeros((h,w),np.uint8)
    for ln in lane.polylines:cv2.polylines(mask,[np.asarray(ln,np.int32)],False,1,16)
    return mask,lane.polylines
   a=cv2.resize(im,(512,288),interpolation=cv2.INTER_AREA)
   a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB).astype(np.float32)/255
   a=(a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
   x=torch.from_numpy(a.transpose(2,0,1).copy()).unsqueeze(0).to(dev)
   with torch.inference_mode(),torch.autocast('cuda',enabled=dev.startswith('cuda')):
    p=model(pixel_values=x).logits
   p=torch.nn.functional.interpolate(p.float(),size=(h,w),mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy()
   return (p==1).astype(np.uint8),None
  counts=np.zeros(3,np.int64);times=[];ext=[]
  for i,name in enumerate(names):
   im=cv2.imread(str(ROOT/'data/raw/culane/val'/name));h,w=im.shape[:2];gt=np.zeros((h,w),np.uint8)
   for line in (ROOT/'data/raw/culane/val'/Path(name).with_suffix('.lines.txt')).read_text().splitlines():
    p=np.fromstring(line,sep=' ')
    if len(p)>=4:cv2.polylines(gt,[p.reshape(-1,2).round().astype(np.int32)],False,1,16)
   t=time.perf_counter();pred,_=predict(im);elapsed=(time.perf_counter()-t)*1000
   if i>0:times.append(elapsed)
   positive=pred>0;truth=gt>0
   counts+=np.array([np.count_nonzero(positive&truth),np.count_nonzero(positive&~truth),np.count_nonzero(~positive&truth)])
   if i in (0,25,50,75):
    view=im.copy();view[truth]=(0,220,0);view[positive]=(0,200,255)
    cv2.imwrite(str(OUT/(kind.replace('-','')+'_'+phase+f'_{i:03d}.jpg')),view)
  for row in external:
   im=cv2.imread(row['image']);t=time.perf_counter();mask,lines=predict(im)
   ext.append({'id':row['id'],'source':row['source_video_id'],'prediction_paint_ratio':float((mask>0).mean()),
    'line_count':len(lines) if lines is not None else None,'latency_ms':(time.perf_counter()-t)*1000})
  tp,fp,fn=map(int,counts)
  score={'tp':tp,'fp':fp,'fn':fn,'pixel_precision':tp/max(1,tp+fp),'pixel_recall':tp/max(1,tp+fn),'pixel_f1':2*tp/max(1,2*tp+fp+fn),
    'warm_mean_ms':float(np.mean(times)),'warm_p95_ms':float(np.percentile(times,95)),'external_frames':len(ext),'external_accuracy':None,'external_predictions':ext}
  report['models'][kind+' '+phase]=score;print('MODEL_COMPARE',kind,phase,score['pixel_f1'],flush=True)
  (OUT/'report.partial.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
  del model;gc.collect();torch.cuda.empty_cache()
 report['culane_image_list_sha256']=hashlib.sha256('\n'.join(names).encode()).hexdigest()
 (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('COMPARE_AUTO_COMPLETE',flush=True)
if __name__=='__main__':main()
