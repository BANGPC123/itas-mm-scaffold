"""Windows-compatible supervised UFLD fine-tuning on official CULane TRAIN labels.

Uses upstream focal/mean losses and lane IDs from segmentation masks. This is
a bounded fine-tuning experiment, not a reproduction of the 50-epoch DALI run.
"""
import gc,importlib.util,json,random,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.perception.ufld_v2 import UFLDv2
BASE=ROOT/'data/raw/culane';OUT=ROOT/'output/retraining_20261009/lane'

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
losses=load_module('upstream_losses',ROOT/'third_party/Ultra-Fast-Lane-Detection-v2/utils/loss.py')
metric=load_module('lane_metric',ROOT/'scripts/evaluate_culane.py')

def targets(points,mask):
    row=np.full((72,4),-1,dtype=np.int64);col=np.full((81,4),-1,dtype=np.int64)
    h,w=mask.shape;ra=np.linspace(.42,1,72)*h;ca=np.linspace(0,1,81)*w
    for pts in points:
        pts=np.asarray(pts);ids=[]
        for x,y in pts:
            patch=mask[max(0,int(y)-2):min(h,int(y)+3),max(0,int(x)-2):min(w,int(x)+3)]
            ids.extend(patch[patch>0].tolist())
        if not ids:continue
        lane=int(np.bincount(ids).argmax())-1
        if not 0<=lane<4:continue
        ordered=pts[np.argsort(pts[:,1])];ys,ix=np.unique(ordered[:,1],return_index=True);xs=ordered[ix,0]
        valid=(ra>=ys.min())&(ra<=ys.max())
        xx=np.full(72,-1.);xx[valid]=np.interp(ra[valid],ys,xs)
        valid_ix=np.flatnonzero(xx>0)
        if xx[-1]<=0 and len(valid_ix)>=6:
            latter=valid_ix[len(valid_ix)//2:];fit=np.polyfit(latter,xx[latter],1)
            xx[latter[0]:]=np.polyval(fit,np.arange(latter[0],72))
        valid=(xx>=0)&(xx<=w);row[valid,lane]=(xx[valid]/w*199).astype(int)
        # Column intersections follow polyline segments, not sorting a non-monotonic curve.
        for j,x in enumerate(ca):
            hits=[]
            for (x1,y1),(x2,y2) in zip(pts[:-1],pts[1:]):
                if x1!=x2 and min(x1,x2)<=x<=max(x1,x2):hits.append(y1+(x-x1)/(x2-x1)*(y2-y1))
            if len(hits) and max(hits)-min(hits)<2 and 0<=hits[0]<=h:col[j,lane]=int(hits[0]/h*99)
    return row,col

def truth(path):
    result=[]
    for line in path.read_text().splitlines():
        a=np.fromstring(line,sep=' ')
        if len(a)>=4:result.append(a.reshape(-1,2).tolist())
    return result

class TrainSet(Dataset):
    def __init__(self):
        self.rows=[l.split() for l in (BASE/'train_subset/train_gt.txt').read_text().splitlines()]
        self.labels=[]
        for r in self.rows:
            path=BASE/'train_subset'/r[0].lstrip('/')
            mask=cv2.imread(str(BASE/'train_subset'/r[1].lstrip('/')),0)
            if mask is None:raise RuntimeError(r[1])
            self.labels.append(targets(truth(path.with_suffix('.lines.txt')),mask))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        path=BASE/'train_subset'/self.rows[i][0].lstrip('/')
        with Image.open(path) as im:a=np.array(im.convert('RGB').resize((1600,533),Image.Resampling.BILINEAR))[-320:]
        a=np.clip(a.astype(np.float32)*random.uniform(.85,1.15),0,255)/255
        a=(a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        return torch.from_numpy(a.transpose(2,0,1).copy()),*self.labels[i]

def evaluate(adapter,names,tag):
    from concurrent.futures import ThreadPoolExecutor
    from collections import deque
    rows=[];pending=deque();started=time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for n in names:
            im=cv2.imread(str(BASE/'val'/n));gt=truth((BASE/'val'/n).with_suffix('.lines.txt'))
            pred=adapter.detect(im,filter_geometry=False).polylines
            pending.append(pool.submit(metric.evaluate_prediction,n,[],pred,gt))
            if len(pending)>=4:rows.append(pending.popleft().result())
            if len(rows) and len(rows)%500==0:print(tag,len(rows),flush=True)
        rows.extend(f.result() for f in pending)
    score=metric.aggregate(rows)['raw_adapter']['all']
    (OUT/f'{tag}.json').write_text(json.dumps(dict(n=len(rows),seconds=time.time()-started,**score),indent=2))
    return score

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.manual_seed(391);random.seed(391);np.random.seed(391)
    dataset=TrainSet();loader=DataLoader(dataset,batch_size=2,shuffle=True,num_workers=0)
    adapter=UFLDv2(ROOT/'models/lane/culane_res18.pth','cuda:0',half=False)
    model=adapter.model
    val=[l.split()[0].lstrip('/') for l in (BASE/'test/list/val.txt').read_text().splitlines()]
    subset=sorted(val,key=lambda x:__import__('hashlib').sha256(x.encode()).hexdigest())[:512]
    baseline=evaluate(adapter,subset,'baseline_val512');best=baseline['f1'];selected=None
    opt=torch.optim.SGD(model.parameters(),lr=1e-5,momentum=.9,weight_decay=1e-4)
    focal=losses.SoftmaxFocalLoss(2,ignore_lb=-1);mean=losses.MeanLoss();ce=torch.nn.CrossEntropyLoss()
    scaler=torch.amp.GradScaler('cuda');history=[]
    for epoch in range(1,3):
        model.train()
        # Batch two is too small for stable running batch-normalization statistics.
        for layer in model.modules():
            if isinstance(layer,torch.nn.BatchNorm2d):layer.eval()
        running=[]
        for step,(x,row,col) in enumerate(loader):
            x=x.cuda();row=row.cuda();col=col.cuda();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                out=model.cls(model.pool(model.model(x)).flatten(1));parts=out.split(model.dims,dim=1);b=x.shape[0]
                locr=parts[0].reshape(b,200,72,4).float();locc=parts[1].reshape(b,100,81,4).float()
                er=parts[2].reshape(b,2,72,4).float();ec=parts[3].reshape(b,2,81,4).float()
                loss=ce(er,(row!=-1).long())+ce(ec,(col!=-1).long())
                if (row!=-1).any():loss=loss+focal(locr,row)+.05*mean(locr,row)
                if (col!=-1).any():loss=loss+focal(locc,col)+.05*mean(locc,col)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite lane loss')
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            scaler.step(opt);scaler.update();running.append(float(loss.detach()))
            if step%100==0:print(json.dumps(dict(epoch=epoch,step=step,steps=len(loader),loss=float(np.mean(running)))),flush=True)
        model.eval();score=evaluate(adapter,subset,f'epoch{epoch}_val512')
        history.append(dict(epoch=epoch,loss=float(np.mean(running)),val=score))
        if score['f1']>best:
            best=score['f1'];selected=epoch
            torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()}},OUT/'candidate.pth')
        (OUT/'history.json').write_text(json.dumps(history,indent=2))
    del opt,scaler,model,adapter;gc.collect();torch.cuda.empty_cache()
    result=dict(train_n=len(dataset),epochs=2,baseline_val512=baseline,history=history,selected_epoch=selected,
                training_scope='4096 official TRAIN images from driver_23; supervised fine-tune, not training from scratch',
                implementation='PyTorch Windows loader, upstream focal/mean losses; no DALI spatial augmentation',candidate_selected=False)
    if selected:
        adapter=UFLDv2(OUT/'candidate.pth','cuda:0',half=True)
        full=evaluate(adapter,val,'candidate_val9675')
        before=json.loads((ROOT/'output/evaluation/improvements_20261009/lane_val/results.json').read_text())['guard_without_geometry_filter']
        result.update(candidate_val=full,baseline_val=before,candidate_selected=full['f1']>before['f1'])
    (OUT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
