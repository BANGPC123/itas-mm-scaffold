"""Fine-tune 30 sign classes plus a background rejection class using TRAIN only."""
import gc,json,random,sys,time,importlib.util
from pathlib import Path
import cv2,numpy as np,pandas as pd,torch
from torch.utils.data import Dataset,DataLoader,ConcatDataset
from torchvision.models import mobilenet_v3_small
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.perception.sign_recognizer import crop_tensor
spec=importlib.util.spec_from_file_location('robust',ROOT/'scripts/finetune_recognizer_robustness.py')
robust=importlib.util.module_from_spec(spec);spec.loader.exec_module(robust)
OUT=ROOT/'output/retraining_20261009/recognizer';LOC=robust.LOC

class Background(Dataset):
    def __init__(self,split,n):
        self.crops=[];self.manifest=[];rng=random.Random(391)
        paths=sorted((LOC/f'images/{split}').glob('*'));rng.shuffle(paths)
        for path in paths:
            if len(self.crops)>=n:break
            image=cv2.imread(str(path))
            if image is None:continue
            h,w=image.shape[:2];labels=LOC/f'labels/{split}'/(path.stem+'.txt')
            if not labels.is_file():continue
            boxes=[]
            for line in labels.read_text().splitlines():
                _,cx,cy,bw,bh=map(float,line.split());boxes.append(((cx-bw/2)*w-16,(cy-bh/2)*h-16,(cx+bw/2)*w+16,(cy+bh/2)*h+16))
            for _ in range(30):
                size=rng.randint(32,min(160,h,w));x=rng.randint(0,w-size);y=rng.randint(0,h-size)
                if any(min(x+size,u)>max(x,a) and min(y+size,v)>max(y,b) for a,b,u,v in boxes):continue
                self.crops.append(image[y:y+size,x:x+size].copy());self.manifest.append(dict(image=str(path.relative_to(LOC)),box=[x,y,x+size,y+size]));break
        if len(self.crops)<n:raise RuntimeError(f'Insufficient {split} backgrounds: {len(self.crops)}')
    def __len__(self):return len(self.crops)
    def __getitem__(self,i):return crop_tensor(self.crops[i]),30

def measure(model,loaders,bg):
    model.eval();res={}
    with torch.inference_mode():
        for name,loader in {**loaders,'background':bg}.items():
            ys=[];ps=[];cs=[]
            for x,y in loader:
                prob=model(x.cuda()).softmax(1).cpu();c,p=prob.max(1);ys.extend(y.tolist());ps.extend(p.tolist());cs.extend(c.tolist())
            y=np.array(ys);p=np.array(ps);c=np.array(cs);accepted=(p<30)&(c>=.5)
            res[name]=dict(n=len(y),accuracy=float((y==p).mean()),accepted_fraction=float(accepted.mean()),
                accepted_accuracy=float((y[accepted]==p[accepted]).mean()) if accepted.any() else None)
    return res

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.manual_seed(391);random.seed(391);np.random.seed(391)
    df=pd.read_csv(ROOT/'data/audit/semantic/final_v1_1/recognition_manifest_v1.csv',keep_default_na=False)
    train=df[(df.split=='TRAIN')&df.semantic_training_eligible.astype(str).str.upper().eq('TRUE')];val=df[df.split=='VAL']
    bgtrain=Background('train',1800);bgval=Background('val',400)
    (OUT/'background_manifest.json').write_text(json.dumps(dict(train=bgtrain.manifest,val=bgval.manifest,video_used=False,
        provenance='Random crops outside expanded annotated boxes; assumes localization annotations complete; not human verified'),indent=2))
    loaders={k:DataLoader(robust.Crops(val,condition=k),batch_size=64) for k in ['clean','blur','small','small_blur']}
    bgloader=DataLoader(bgval,batch_size=64)
    checkpoint=torch.load(ROOT/'models/sign_recognizer/robust_val.pt',map_location='cpu',weights_only=True)
    old=mobilenet_v3_small(num_classes=30).cuda();old.load_state_dict(checkpoint['state_dict'])
    baseline=measure(old,loaders,bgloader);del old;gc.collect();torch.cuda.empty_cache()
    model=mobilenet_v3_small(num_classes=31).cuda()
    state=model.state_dict()
    for k,v in checkpoint['state_dict'].items():
        if k in ('classifier.3.weight','classifier.3.bias'):state[k][:30]=v
        else:state[k]=v
    model.load_state_dict(state)
    tl=DataLoader(ConcatDataset([robust.Crops(train,training=True),bgtrain]),batch_size=64,shuffle=True)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=1e-4);lossfn=torch.nn.CrossEntropyLoss()
    history=[];best=None;selected=None;start=time.time()
    for epoch in range(1,6):
        model.train();losses=[]
        for x,y in tl:
            opt.zero_grad(set_to_none=True);loss=lossfn(model(x.cuda()),y.cuda());loss.backward();opt.step();losses.append(float(loss.detach()))
        result=measure(model,loaders,bgloader)
        acceptable=all(result[k]['accuracy']>=baseline[k]['accuracy']-.01 for k in loaders)
        score=float(np.mean([result[k]['accuracy'] for k in loaders]))-.25*result['background']['accepted_fraction']
        row=dict(epoch=epoch,loss=float(np.mean(losses)),val=result,acceptable=acceptable,selection_score=score);history.append(row)
        print(json.dumps(row),flush=True)
        if acceptable and result['background']['accepted_fraction']<baseline['background']['accepted_fraction'] and (best is None or score>best):
            best=score;selected=epoch
            torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                classes=checkpoint['classes']+['unknown'],image_size=128,confidence_threshold=.5,
                background_class=True,training_provenance='TRAIN signs and background crops; no user video'),OUT/'candidate.pt')
        (OUT/'history.json').write_text(json.dumps(history,indent=2))
    result=dict(baseline=baseline,history=history,selected_epoch=selected,candidate_selected=selected is not None,
        training_n=len(tl.dataset),background_train_n=len(bgtrain),background_val_n=len(bgval),video_used_for_training=False,
        selection='All four known-class VAL accuracies drop <=1pp; lower background false acceptance; maximize mean known accuracy minus .25 background FAR',
        background_validation_limit='Backgrounds outside existing boxes, not comprehensive OOD ground truth',seconds=time.time()-start)
    (OUT/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
