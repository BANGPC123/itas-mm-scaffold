"""TRAIN-only augmentation; candidate selection on VAL, never on TEST."""
from pathlib import Path
import sys,json,random,math,time,hashlib,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.models import mobilenet_v3_small
from sklearn.metrics import f1_score
from src.perception.sign_recognizer import crop_tensor

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/evaluation/improvements_20261009/recognizer_small'
LOC=ROOT/'data/processed/canonical/itas_mm_v1/localization'

def degrade(crop,kind):
    if kind=='blur':return cv2.GaussianBlur(crop,(5,5),1.2)
    if kind in ('small','small_blur'):
        scale=min(1.,16/min(crop.shape[:2]))
        small=cv2.resize(crop,(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))),interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small,(3,3),.7) if kind=='small_blur' else small
    return crop

class Crops(Dataset):
    def __init__(self,rows,training=False,condition='clean'):
        self.rows=rows.reset_index(drop=True);self.training=training;self.condition=condition
        self.images=[]
        for r in self.rows.itertuples():
            image=cv2.imread(str(LOC/r.image_path));h,w=image.shape[:2]
            a,b,c,d=json.loads(r.bbox_xyxy)
            self.images.append(image[max(0,math.floor(b)):min(h,math.ceil(d)),max(0,math.floor(a)):min(w,math.ceil(c))].copy())
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        crop=self.images[i]
        if self.training:
            kind=random.choice(['clean','clean','blur','small','small_blur'])
            crop=degrade(crop,kind)
            crop=np.clip(crop.astype(float)*random.uniform(.6,1.4),0,255).astype(np.uint8)
        else:crop=degrade(crop,self.condition)
        return crop_tensor(crop),int(self.rows.iloc[i].class_index)

def evaluate(model,loaders,device):
    model.eval();result={}
    for kind,loader in loaders.items():
        y=[];pred=[]
        with torch.inference_mode():
            for x,target in loader:
                y.extend(target.tolist());pred.extend(model(x.to(device)).argmax(1).cpu().tolist())
        y=np.array(y);pred=np.array(pred);large=loader.dataset.rows.short_side_px.to_numpy()>=32
        result[kind]={'accuracy':float((y==pred).mean()),'macro_f1':float(f1_score(y,pred,labels=list(range(30)),average='macro',zero_division=0)),
                      'ge32_accuracy':float((y[large]==pred[large]).mean()),'n':len(y)}
    result['selection_score']=float(np.mean([r['macro_f1'] for r in result.values()]))
    return result

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'models/sign_recognizer/best.pt')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--epochs',type=int,default=4)
    args=parser.parse_args();OUT=args.output
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.manual_seed(391);np.random.seed(391);random.seed(391)
    df=pd.read_csv(ROOT/'data/audit/semantic/final_v1_1/recognition_manifest_v1.csv',keep_default_na=False)
    train=df[(df.split=='TRAIN')&df.semantic_training_eligible.astype(str).str.upper().eq('TRUE')];val=df[df.split=='VAL']
    assert not set(train.canonical_box_id)&set(val.canonical_box_id)
    source=args.source;checkpoint=torch.load(source,weights_only=True,map_location='cpu')
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    model=mobilenet_v3_small(num_classes=len(checkpoint['classes'])).to(device);model.load_state_dict(checkpoint['state_dict'])
    loaders={k:DataLoader(Crops(val,condition=k),batch_size=64) for k in ['clean','blur','small','small_blur']}
    tl=DataLoader(Crops(train,training=True),batch_size=64,shuffle=True)
    baseline=evaluate(model,loaders,device);best=baseline['selection_score'];history=[];selected=None;started=time.perf_counter()
    print(json.dumps({'baseline':baseline}),flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-4)
    lossfn=torch.nn.CrossEntropyLoss()
    for epoch in range(1,args.epochs+1):
        model.train()
        for x,y in tl:
            optimizer.zero_grad(set_to_none=True);loss=lossfn(model(x.to(device)),y.to(device));loss.backward();optimizer.step()
        result=evaluate(model,loaders,device);history.append({'epoch':epoch,**result});print(json.dumps(history[-1]),flush=True)
        acceptable=(result['clean']['accuracy']>=baseline['clean']['accuracy']-.01 and result['clean']['ge32_accuracy']>=baseline['clean']['ge32_accuracy']-.01)
        if acceptable and result['selection_score']>best:
            best=result['selection_score'];selected=epoch
            candidate={**checkpoint,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'robustness_epoch':epoch}
            torch.save(candidate,OUT/'candidate.pt')
    report={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'train_n':len(train),'val_n':len(val),'baseline':baseline,'history':history,
            'selected_epoch':selected,'selected_score':best,'deployed':False,'test_used':False,'human_verified':False,
            'selection':'Mean VAL macro-F1 clean/blur/small/small_blur, clean accuracy and ge32 accuracy drop <=1 percentage point; keep original if no improvement',
            'conditions':{'small':'resize short side to at most 16px','small_blur':'small then Gaussian 3x3 sigma=.7','blur':'Gaussian 5x5 sigma=1.2'},
            'elapsed_seconds':time.perf_counter()-started}
    (OUT/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
