"""EXPERIMENTAL 33-class MobileNet extension: 30 old + 2 R420/R421 + unknown.

Frozen backbone + last linear only. Source route GT remains UNVERIFIED.
Never overwrite selected production weight; independent field gate must pass.
"""
from pathlib import Path
import json,random,math,time,copy,collections
import numpy as np,pandas as pd,cv2,torch
from torch.utils.data import Dataset,DataLoader
from torchvision.models import mobilenet_v3_small
from src.perception.sign_recognizer import crop_tensor

ROOT=Path(__file__).resolve().parents[1]
LOC=ROOT/'data/processed/canonical/itas_mm_v1/localization'
EXT=ROOT/'data/raw/external/vietspeedyolo_hf_20261011'
CAND=ROOT/'data/processed/candidates/vietspeedyolo_hf_20261011/train_candidate_manifest.jsonl'
OUT=ROOT/'output/retraining_20261011/recognizer_33class_EXPERIMENT'
OUT.mkdir(parents=True,exist_ok=True)
torch.manual_seed(391);random.seed(391);np.random.seed(391)
torch.set_num_threads(4);cv2.setNumThreads(1)
class SignCrops(Dataset):
    def __init__(self,items):self.items=items
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        path,box,cls=self.items[i];img=cv2.imread(str(path))
        if img is None:raise ValueError('Unreadable '+str(path))
        h,w=img.shape[:2];a,b,c,d=[float(v) for v in box]
        crop=img[max(0,math.floor(b)):min(h,math.ceil(d)),max(0,math.floor(a)):min(w,math.ceil(c))]
        if not crop.size:raise ValueError('Bad crop '+str(path))
        return crop_tensor(crop,128),cls

def ext_rows(split,source=None):
    if source is not None:
        data=source
    else:
        data=[]
        for p in sorted((EXT/'images'/split).glob('*')):
            if p.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
            data.append({'file':f'images/{split}/{p.name}',
                         'labels':f'labels/{split}/{p.stem}.txt'})
    results=[]
    for r in data:
        img=cv2.imread(str(EXT/r['file']))
        if img is None:continue
        h,w=img.shape[:2]
        for line in (EXT/r['labels']).read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            items=line.split();n=int(items[0]);x,y,bw,bh=map(float,items[1:])
            if n not in (0,1):continue
            results.append((EXT/r['file'],[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h],30+n))
    return results

df=pd.read_csv(ROOT/'data/audit/semantic/final_v1_1/recognition_manifest_v1.csv',keep_default_na=False)
rng=random.Random(391)
known_train=[]
for index,group in df[(df.split=='TRAIN') & (df.semantic_training_eligible.astype(str).str.upper()=='TRUE')].groupby('class_index'):
    sampled=group.sample(n=min(50,len(group)),random_state=391)
    for r in sampled.itertuples():
        known_train.append((LOC/r.image_path,json.loads(r.bbox_xyxy),int(index)))
known_val=[(LOC/r.image_path,json.loads(r.bbox_xyxy),int(r.class_index))
           for r in df[df.split=='VAL'].itertuples()]
source=[json.loads(x) for x in CAND.read_text(encoding='utf-8').splitlines() if x.strip()]
ext_train=ext_rows('train',source)
ext_val=ext_rows('val')
print('DATA',len(known_train),len(known_val),len(ext_train),len(ext_val),flush=True)

selected=ROOT/'models/sign_recognizer/retrained_20261009.pt'
original=torch.load(selected,map_location='cpu',weights_only=True)
classes=original['classes']
assert len(classes)==31 and classes[-1]=='unknown'
new_classes=classes[:-1]+['R.420','R.421']+['unknown']
device='cuda:0' if torch.cuda.is_available() else 'cpu'
old=mobilenet_v3_small(num_classes=31)
old.load_state_dict(original['state_dict'])
old.to(device).eval()
# Encode frozen features once; the upper classifier-only optimization fits in GPU.
def features(items):
    loader=DataLoader(SignCrops(items),batch_size=64,shuffle=False,num_workers=0)
    feats=[];ys=[]
    with torch.inference_mode():
        for x,y in loader:
            x=x.to(device)
            z=old.features(x);z=old.avgpool(z);z=torch.flatten(z,1)
            z=old.classifier[:-1](z)
            feats.append(z.cpu());ys.append(y)
    return torch.cat(feats),torch.cat(ys)
start=time.time()
Xk,yk=features(known_train)
Xv,yv=features(known_val)
Xe,ye=features(ext_train)
Xev,yev=features(ext_val)
print('FEATURES_EXTRACTED',round(time.time()-start,1),'seconds',flush=True)
W=old.classifier[-1].weight.detach().cpu();b=old.classifier[-1].bias.detach().cpu()
# model head 33 output, keep old unknown at LAST.
head=torch.nn.Linear(W.shape[1],33,device=device)
with torch.no_grad():
    head.weight[:30]=W[:30].to(device)
    head.bias[:30]=b[:30].to(device)
    head.weight[32]=W[30].to(device)
    head.bias[32]=b[30].to(device)
    torch.nn.init.normal_(head.weight[30:32],std=.025)
    head.bias[30:32]=-2.
head.train()
optimizer=torch.optim.AdamW(head.parameters(),lr=2e-3,weight_decay=1e-3)
X=torch.cat([Xk,Xe]).to(device);y=torch.cat([yk,ye]).to(device)
# Balanced new classes, but constrain old-head logits with KL on original model.
teacher=torch.nn.functional.linear(Xk.to(device),W.to(device),b.to(device)).detach()
history=[];best=None
for epoch in range(10):
    order=torch.randperm(len(y),device=device)
    totals=[]
    for batch in order.split(128):
        out=head(X[batch])
        loss=torch.nn.functional.cross_entropy(out,y[batch])
        # Old known logits remain close to selected checkpoint.
        with torch.no_grad(): pass
        kd=torch.nn.functional.mse_loss(head(Xk.to(device))[:,:30],teacher[:,:30])
        value=loss+.12*kd
        optimizer.zero_grad(set_to_none=True);value.backward();optimizer.step()
        totals.append(float(loss.detach()))
    with torch.inference_mode():
        old_pred=old.classifier[-1](Xv.to(device)).argmax(1).cpu()
        new_pred=head(Xv.to(device)).argmax(1).cpu()
        a0=float((old_pred==yv).float().mean())
        a1=float((new_pred==yv).float().mean())
        ev_pred=head(Xev.to(device)).argmax(1).cpu()
        ea=float((ev_pred==yev).float().mean()) if len(yev) else None
        c=collections.Counter(ev_pred.tolist())
        rec={'epoch':epoch+1,'training_loss':float(np.mean(totals)),
             'canonical_VAL_baseline_acc':a0,'canonical_VAL_33class_acc':a1,
             'external_source_VAL_accuracy_NOT_INDEPENDENT':ea,
             'new_classes_confusion':dict(c)}
        history.append(rec)
        print('EPOCH',json.dumps(rec),flush=True)
        if best is None or (a1>=a0-.01 and ea is not None and ea>best[0]):
            if a1>=a0-.01:
                best=(ea,epoch+1,copy.deepcopy(head.state_dict()))
    (OUT/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
if best:
    model=mobilenet_v3_small(num_classes=33)
    state=original['state_dict']
    state['classifier.3.weight']=best[2]['weight'].cpu()
    state['classifier.3.bias']=best[2]['bias'].cpu()
    model.load_state_dict(state,strict=True)
    torch.save({'state_dict':model.state_dict(),'classes':new_classes,'image_size':128,
                'confidence_threshold':.85,'training_provenance':'EXPERIMENT source labeled R420/R421 + sampled canonical TRAIN; external source VAL non-independent',
                'selected_for_production':False,'independent_field_verified':False},
               OUT/'candidate_33classes.pt')
report={'status':'EXPERIMENT_ONLY_NOT_FOR_PRODUCTION',
        'candidate_saved':bool(best),'chosen_epoch':best[1] if best else None,
        'train_canonical_known_n':len(Xk),'train_external_R420_R421_n':len(Xe),
        'val_canonical_known_n':len(Xv),'val_external_nonindependent_n':len(Xev),
        'classes':new_classes,'history':history,'train_seconds':round(time.time()-start,1),
        'release_blockers':['external route/source authenticity unreviewed',
                            'source VAL near duplicate contamination',
                            'missing independently labeled real Vietnam field test',
                            'cannot infer new sign legal scope from classification'],
        'selected_checkpoint_unchanged':True}
(OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('FINISHED',json.dumps({k:v for k,v in report.items() if k not in ('history','classes')},ensure_ascii=False),flush=True)
