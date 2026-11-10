"""Research-only MobileNetV3 fine-tune with assistant-reviewed hard negatives.

TRAIN sign set and four development-video negatives only; choose by canonical
VAL, NEVER report source-video examples as independent TEST. No auto-deploy.
"""
from pathlib import Path
import sys,json,random,time,hashlib,importlib.util
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader,Dataset,ConcatDataset
from torchvision.models import mobilenet_v3_small
from sklearn.metrics import f1_score
from src.perception.sign_recognizer import crop_tensor
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/retraining_20261011/recognizer_hardneg'
spec=importlib.util.spec_from_file_location('robust',ROOT/'scripts/finetune_recognizer_robustness.py')
robust=importlib.util.module_from_spec(spec);spec.loader.exec_module(robust)
spec2=importlib.util.spec_from_file_location('revtrainer',ROOT/'scripts/retrain_recognizer_candidate.py')
rev=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(rev)

class Examples(Dataset):
    def __init__(self,training=True):
        self.training=training
        folder=ROOT/'data/audit/development_hard_negatives_20261011'
        self.samples=[cv2.imread(str(folder/name)) for name in
          ['dashboard_gauge_a.png','dashboard_gauge_b.png','commercial_logo.png','vehicle_tail_lamp.png']]
        if any(s is None for s in self.samples):raise RuntimeError('Missing reviewed source crops')
    def __len__(self):return 240 if self.training else 4
    def __getitem__(self,i):
        crop=self.samples[i%4].copy()
        # Safe perturbations: preserve the object, vary exposure and blur.
        if self.training:
            if i%3==0:crop=cv2.GaussianBlur(crop,(3,3),.6)
            crop=np.clip(crop.astype(np.float32)*random.uniform(.7,1.35),0,255).astype(np.uint8)
        return crop_tensor(crop),30

def measure(net,val,background,negative,device):
    net.eval()
    result={}
    with torch.inference_mode():
        for name,loader in [('known',val),('unseen_background_proxy',background),
                            ('development_training_negatives',negative)]:
            ys=[];preds=[];accept=[]
            for x,y in loader:
                p=net(x.to(device)).softmax(1).cpu();conf,cls=p.max(1)
                ys.extend(y.tolist());preds.extend(cls.tolist())
                accept.extend(((cls!=30)&(conf>=.63)).tolist())
            y=np.array(ys);p=np.array(preds);a=np.array(accept)
            result[name]={'n':len(y),'accuracy':float((y==p).mean()),
              'accepted_fraction':float(a.mean()),
              'macro_f1_30':float(f1_score(y,p,labels=list(range(30)),average='macro',zero_division=0)) if name=='known' else None}
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    random.seed(391);np.random.seed(391);torch.manual_seed(391);torch.set_num_threads(4);cv2.setNumThreads(1)
    df=pd.read_csv(ROOT/'data/audit/semantic/final_v1_1/recognition_manifest_v1.csv',keep_default_na=False)
    train=df[(df.split=='TRAIN')&(df.semantic_training_eligible.astype(str).str.upper()=='TRUE')]
    val=df[df.split=='VAL']
    assert not set(train.canonical_box_id)&set(val.canonical_box_id)
    source=ROOT/'models/sign_recognizer/retrained_20261009.pt'
    ckpt=torch.load(source,weights_only=True,map_location='cpu')
    assert len(ckpt['classes'])==31 and ckpt['classes'][-1]=='unknown'
    dev='cuda:0' if torch.cuda.is_available() else 'cpu'
    model=mobilenet_v3_small(num_classes=31).to(dev)
    model.load_state_dict(ckpt['state_dict'])
    # Deterministic capped TRAIN sample, never substitute VAL into training.
    train=train.sample(n=min(2400,len(train)),random_state=391)
    positive=robust.Crops(train,training=True)
    valset=robust.Crops(val,condition='clean')
    backgrounds=rev.Background('train',600)
    bgval=rev.Background('val',400)
    hard=Examples(training=True)
    hard_val=Examples(training=False)
    trainloader=DataLoader(ConcatDataset([positive,backgrounds,hard]),batch_size=48,shuffle=True,num_workers=0)
    knownloader=DataLoader(valset,batch_size=64,num_workers=0)
    bgloader=DataLoader(bgval,batch_size=64,num_workers=0)
    hardloader=DataLoader(hard_val,batch_size=64,num_workers=0)
    baseline=measure(model,knownloader,bgloader,hardloader,dev)
    print('BASELINE '+json.dumps(baseline),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    loss_fn=torch.nn.CrossEntropyLoss()
    history=[];best=None
    for epoch in (1,2):
        model.train();losses=[]
        for x,y in trainloader:
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(x.to(dev)),y.to(dev))
            loss.backward();opt.step();losses.append(float(loss.detach()))
        result=measure(model,knownloader,bgloader,hardloader,dev)
        row={'epoch':epoch,'loss':float(np.mean(losses)),'val':result}
        history.append(row)
        print('EPOCH '+json.dumps(row),flush=True)
        good=(result['known']['macro_f1_30']>=baseline['known']['macro_f1_30']-.005 and
              result['known']['accuracy']>=baseline['known']['accuracy']-.005 and
              result['unseen_background_proxy']['accepted_fraction']<=baseline['unseen_background_proxy']['accepted_fraction']+.001 and
              result['development_training_negatives']['accepted_fraction']<baseline['development_training_negatives']['accepted_fraction'])
        if good and (best is None or result['known']['macro_f1_30']>best):
            best=result['known']['macro_f1_30']
            torch.save({**ckpt,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                         'hard_negative_provenance':'4 assistant-reviewed DEVELOPMENT video crops, NOT a generalization benchmark'},OUT/'candidate.pt')
    report={'baseline':baseline,'history':history,'candidate_exists':(OUT/'candidate.pt').is_file(),
       'deployed':False,'source_checkpoint':str(source),'train_sign_n':len(train),
       'val_sign_n':len(val),'background_train_n':len(backgrounds),
       'background_val_proxy_n':len(bgval),'development_negative_unique':4,
       'independent_vietnam_test':False,
       'warning':'Same-source negatives. They are training examples, not test. No deployment until separate human-labeled ads/traffic video test.'}
    (OUT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('RESULT '+json.dumps(report),flush=True)
if __name__=='__main__':main()
