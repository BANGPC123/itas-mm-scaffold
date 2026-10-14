from pathlib import Path
import sys,json,math,random,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,confusion_matrix,classification_report
from src.perception.sign_recognizer import crop_tensor

ROOT=Path(__file__).resolve().parents[1]
LOC=ROOT/'data/processed/canonical/itas_mm_v1/localization'
SEM=ROOT/'data/audit/semantic/final_v1_1'

class Crops(Dataset):
    def __init__(self,df,augment=False):self.df=df.reset_index(drop=True);self.augment=augment
    def __len__(self):return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i];im=cv2.imread(str(LOC/r.image_path));h,w=im.shape[:2]
        x1,y1,x2,y2=json.loads(r.bbox_xyxy)
        crop=im[max(0,math.floor(y1)):min(h,math.ceil(y2)),max(0,math.floor(x1)):min(w,math.ceil(x2))]
        if crop.size==0:raise ValueError(r.canonical_box_id)
        if self.augment:
            crop=np.clip(crop.astype(np.float32)*random.uniform(.85,1.15),0,255).astype(np.uint8)
        return crop_tensor(crop),int(r.class_index)

def predict(model,loader):
    model.eval();truth=[];prob=[]
    with torch.inference_mode():
        for x,y in loader:
            prob.extend(model(x.cuda()).softmax(1).cpu().numpy());truth.extend(y.numpy())
    return np.array(truth),np.array(prob)

def metrics(y,p):
    pred=p.argmax(1)
    return {'n':len(y),'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,labels=list(range(30)),average='macro',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred))}

def main():
    torch.manual_seed(391);np.random.seed(391);random.seed(391)
    torch.set_num_threads(4);cv2.setNumThreads(0)
    df=pd.read_csv(SEM/'recognition_manifest_v1.csv',keep_default_na=False)
    classes=pd.read_csv(SEM/'recognition_classes_v1.csv').sort_values('class_index').canonical_sign_code.tolist()
    out=ROOT/'output/evaluation/recognizer';out.mkdir(parents=True,exist_ok=True)
    target=ROOT/'models/sign_recognizer/best.pt';target.parent.mkdir(parents=True,exist_ok=True)
    train=df[(df.split=='TRAIN')&df.semantic_training_eligible.astype(str).str.upper().eq('TRUE')]
    val=df[df.split=='VAL'];test=df[df.split=='TEST']
    loader=lambda d,aug=False,shuffle=False:DataLoader(Crops(d,aug),batch_size=64,shuffle=shuffle,num_workers=0,pin_memory=True)
    vl=loader(val)
    model=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3]=torch.nn.Linear(model.classifier[3].in_features,len(classes));model.cuda()
    counts=np.bincount(train.class_index,minlength=30)
    history=[];best=-1;start=time.time()
    for phase in ['baseline','finetuned']:
        if phase=='finetuned':model.load_state_dict(torch.load(target,weights_only=True)['state_dict'])
        weights=None if phase=='baseline' else torch.tensor(np.sqrt(counts.sum()/(30*counts)),dtype=torch.float32,device='cuda')
        lossfn=torch.nn.CrossEntropyLoss(weight=weights)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001 if phase=='baseline' else .0002,weight_decay=.0001)
        tl=loader(train,phase=='finetuned',True)
        for epoch in range(8):
            model.train();losses=[]
            for x,y in tl:
                optimizer.zero_grad(set_to_none=True);loss=lossfn(model(x.cuda()),y.cuda());loss.backward();optimizer.step();losses.append(float(loss.detach()))
            y,p=predict(model,vl);m=metrics(y,p);row={'phase':phase,'epoch':epoch+1,'train_loss':float(np.mean(losses)),**m};history.append(row)
            print(json.dumps(row),flush=True)
            (out/'history.json').write_text(json.dumps(history,indent=2))
            if m['macro_f1']>best:
                best=m['macro_f1'];torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'classes':classes,'image_size':128,'phase':phase,'epoch':epoch+1},target)
        (out/'history.json').write_text(json.dumps(history,indent=2))
    checkpoint=torch.load(target,weights_only=True);model.load_state_dict(checkpoint['state_dict'])
    y,p=predict(model,vl);conf=p.max(1);correct=p.argmax(1)==y
    thresholds=[]
    for threshold in np.linspace(.5,.99,50):
        keep=conf>=threshold
        if keep.sum():thresholds.append({'threshold':float(threshold),'coverage':float(keep.mean()),'accepted_accuracy':float(correct[keep].mean())})
    usable=[v for v in thresholds if v['accepted_accuracy']>=.9]
    chosen=max(usable,key=lambda v:v['coverage']) if usable else {'threshold':.99,'coverage':0,'accepted_accuracy':None}
    checkpoint['confidence_threshold']=chosen['threshold'];torch.save(checkpoint,target)
    result={'selected_phase':checkpoint['phase'],'selected_epoch':checkpoint['epoch'],'val':metrics(y,p),'rejection_calibration':chosen,'threshold_curve':thresholds,'selection':'VAL macro-F1; threshold from known-class VAL only','ood_validated':False,'history':history}
    y,p=predict(model,loader(test));result['test_all']=metrics(y,p)
    for label,mask in [('ge32',test.short_side_px.to_numpy()>=32),('lt32',test.short_side_px.to_numpy()<32)]:
        result['test_'+label]=metrics(y[mask],p[mask]) if mask.any() else None
    keep=p.max(1)>=chosen['threshold'];result['test_accepted']={'coverage':float(keep.mean()),'accuracy':float((p.argmax(1)[keep]==y[keep]).mean()) if keep.any() else None}
    result['per_class']=classification_report(y,p.argmax(1),labels=list(range(30)),target_names=classes,output_dict=True,zero_division=0)
    result['elapsed_seconds']=time.time()-start
    (out/'results.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame({'box_id':test.canonical_box_id,'true':y,'pred':p.argmax(1),'confidence':p.max(1),'short_side_px':test.short_side_px}).to_csv(out/'test_predictions.csv',index=False)
    np.save(out/'confusion_matrix.npy',confusion_matrix(y,p.argmax(1),labels=list(range(30))))
    print(json.dumps({k:v for k,v in result.items() if k not in ['history','per_class','threshold_curve']},indent=2),flush=True)

if __name__=='__main__':main()
