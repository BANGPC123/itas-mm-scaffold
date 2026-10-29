"""Train recognizer v2 (MobileNetV3-small, 128 px) on TRAIN crops of at least 16 px; select on VAL only.

Initialises shared classes from retrained_20261009.pt. TEST is never read here.
"""
from __future__ import annotations
import argparse,json,random,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np,pandas as pd,torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
from torchvision.models import mobilenet_v3_small
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.perception.recognizer_v2_crops import augment,build_cache,exact_crop,load_cache,save_cache
from src.perception.sign_recognizer import crop_tensor
OUT=ROOT/'output/kiet_wp_20261028/recognizer_v2'
LOC=ROOT/'data/processed/canonical/itas_mm_v1/localization'
OLD=ROOT/'models/sign_recognizer/retrained_20261009.pt'
MIN_SIDE=16


def load_split(out,split,classes,boxes,background):
    """Rows and crop entries for one split: sign boxes of trained classes plus background crops."""
    signs=boxes[(boxes.split==split)&(boxes.short_side_px>=MIN_SIDE)&boxes.code.isin(classes)]
    bg=background[background.split==split]
    rows=pd.concat([signs[['image_path','bbox_xyxy','short_side_px','code','source']],bg[['image_path','bbox_xyxy','short_side_px','code','source']]],ignore_index=True)
    cache=out/f'crop_cache_{split.lower()}.pkl'
    if cache.is_file():entries=load_cache(cache)
        # The cache is keyed by row order, so rebuild if the table changed.
    else:entries=None
    if entries is None or len(entries)!=len(rows):
        entries=build_cache(rows,LOC);save_cache(entries,cache)
    keep=[i for i,e in enumerate(entries) if e is not None]
    return rows.iloc[keep].reset_index(drop=True),[entries[i] for i in keep]


class TrainCrops(Dataset):
    def __init__(self,entries,labels,seed):
        self.entries=entries;self.labels=labels;self.rng=random.Random(seed)
    def __len__(self):return len(self.entries)
    def __getitem__(self,i):
        return crop_tensor(augment(self.entries[i],self.rng)),int(self.labels[i])


def predict(model,entries,device,batch=256):
    model.eval();out=[]
    with torch.inference_mode():
        for start in range(0,len(entries),batch):
            x=torch.stack([crop_tensor(exact_crop(e)) for e in entries[start:start+batch]]).to(device)
            out.append(model(x).float().cpu())
    return torch.cat(out)


def evaluate(logits,y,classes,main,sizes,background_index):
    prob=logits.softmax(1);conf,pred=prob.max(1);y=np.asarray(y);pred=pred.numpy();conf=conf.numpy()
    main_idx=[classes.index(c) for c in main];known=y!=background_index
    result=dict(n=int(len(y)),accuracy_known=float((pred[known]==y[known]).mean()),
        macro_f1_main=float(f1_score(y,pred,labels=main_idx,average='macro',zero_division=0)),
        macro_f1_all_known=float(f1_score(y,pred,labels=sorted(set(y[known])),average='macro',zero_division=0)),
        background_accept_0_5=float(((pred[~known]!=background_index)&(conf[~known]>=.5)).mean()) if (~known).any() else None)
    for name,mask in (('ge32',known&(sizes>=32)),('16_32',known&(sizes<32))):
        result[f'accuracy_{name}']=float((pred[mask]==y[mask]).mean()) if mask.any() else None
        result[f'n_{name}']=int(mask.sum())
    return result


def calibrate(logits,y,sizes,background_index):
    prob=logits.softmax(1);conf,pred=prob.max(1);conf=conf.numpy();pred=pred.numpy();y=np.asarray(y);known=y!=background_index
    curve=[]
    for t in np.linspace(.5,.99,50):
        keep=known&(pred!=background_index)&(conf>=t);row=dict(threshold=float(t))
        for name,mask in (('all',known),('ge32',known&(sizes>=32)),('16_32',known&(sizes<32))):
            sel=keep&mask;row[f'coverage_{name}']=float(sel.sum()/max(1,mask.sum()));row[f'accepted_accuracy_{name}']=float((pred[sel]==y[sel]).mean()) if sel.any() else 0.
        bg=~known;row['background_accepted']=float(((pred[bg]!=background_index)&(conf[bg]>=t)).mean()) if bg.any() else None
        curve.append(row)
    usable=[r for r in curve if r['coverage_all']>=.8 and r['accepted_accuracy_all']>=.99]
    chosen=min(usable,key=lambda r:r['threshold']) if usable else max(curve,key=lambda r:r['accepted_accuracy_all']*r['coverage_all'])
    return chosen,curve,bool(usable)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT/'train');ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--batch',type=int,default=128);ap.add_argument('--epoch-size',type=int,default=14000);ap.add_argument('--lr',type=float,default=4e-4)
    ap.add_argument('--seed',type=int,default=20261028);ap.add_argument('--balance-power',type=float,default=.7);ap.add_argument('--workers',type=int,default=0)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed);torch.set_num_threads(4);cv2.setNumThreads(1)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    report=json.loads((OUT/'class_list.json').read_text(encoding='utf-8'));classes=report['classes'];main_classes=report['main']
    bg=classes.index('unknown');boxes=pd.read_csv(OUT/'boxes.csv',keep_default_na=False);background=pd.read_csv(OUT/'background_boxes.csv',keep_default_na=False)
    started=time.time()
    train_rows,train_entries=load_split(OUT,'TRAIN',classes,boxes,background);val_rows,val_entries=load_split(OUT,'VAL',classes,boxes,background)
    print('crops',len(train_rows),len(val_rows),f'{time.time()-started:.0f}s',flush=True)
    ytr=train_rows.code.map(classes.index).to_numpy();yva=val_rows.code.map(classes.index).to_numpy();sizes=val_rows.short_side_px.to_numpy()
    counts=np.bincount(ytr,minlength=len(classes));weights=(counts[ytr].astype(float))**(-args.balance_power)
    sampler=WeightedRandomSampler(torch.from_numpy(weights),num_samples=args.epoch_size,replacement=True,generator=torch.Generator().manual_seed(args.seed))
    loader=DataLoader(TrainCrops(train_entries,ytr,args.seed),batch_size=args.batch,sampler=sampler,num_workers=args.workers,drop_last=True)
    old=torch.load(OLD,map_location='cpu',weights_only=True);old_index={c:i for i,c in enumerate(old['classes'])}
    model=mobilenet_v3_small(num_classes=len(classes));state=model.state_dict();copied=[]
    for k,v in old['state_dict'].items():
        if k in ('classifier.3.weight','classifier.3.bias'):continue
        state[k]=v
    for new_i,c in enumerate(classes):
        if c in old_index:
            state['classifier.3.weight'][new_i]=old['state_dict']['classifier.3.weight'][old_index[c]];state['classifier.3.bias'][new_i]=old['state_dict']['classifier.3.bias'][old_index[c]];copied.append(c)
    model.load_state_dict(state);model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4);steps=args.epochs*(args.epoch_size//args.batch)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=args.lr,total_steps=steps,pct_start=.08,anneal_strategy='cos')
    lossfn=torch.nn.CrossEntropyLoss(label_smoothing=.05)
    history=[];best=None;best_logits=None
    for epoch in range(1,args.epochs+1):
        model.train();losses=[];t0=time.time()
        for x,y in loader:
            opt.zero_grad(set_to_none=True);loss=lossfn(model(x.to(device)),y.to(device));loss.backward();opt.step();sched.step();losses.append(float(loss.detach()))
        logits=predict(model,val_entries,device);metrics=evaluate(logits,yva,classes,main_classes,sizes,bg)
        row=dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.time()-t0,**metrics);history.append(row);print(json.dumps(row),flush=True)
        key=(metrics['macro_f1_main'],metrics['accuracy_known'])
        if best is None or key>best[0]:
            best=(key,epoch);best_logits=logits
            torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},classes=classes,image_size=128,confidence_threshold=.5,
                background_class=True,main_classes=main_classes,low_support_classes=report['low_support'],
                training_provenance='TRAIN crops >=16 px (released V1.2 codes + AI-reviewed ontology extension) and random background crops; labels are AI-assisted, not human GT; TEST unused',
                seed=args.seed,selected_epoch=epoch),out/'candidate.pt')
        (out/'history.json').write_text(json.dumps(history,indent=1),encoding='utf-8')
    chosen,curve,met=calibrate(best_logits,yva,sizes,bg)
    ckpt=torch.load(out/'candidate.pt',map_location='cpu',weights_only=True);ckpt['confidence_threshold']=float(chosen['threshold']);torch.save(ckpt,out/'candidate.pt')
    np.savez_compressed(out/'val_logits.npz',logits=best_logits.numpy(),y=yva,short_side=sizes)
    (out/'calibration.json').write_text(json.dumps(dict(selected=chosen,curve=curve,criterion_met=met,split='VAL',
        criterion='smallest threshold with accepted accuracy >=0.99 and coverage >=0.80 on VAL known-class crops >=16 px; otherwise maximise accuracy*coverage',test_used=False),indent=1),encoding='utf-8')
    commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,cwd=ROOT).stdout.strip()
    (out/'results.json').write_text(json.dumps(dict(command=' '.join(sys.argv),git_head=commit,seed=args.seed,epochs=args.epochs,batch=args.batch,epoch_size=args.epoch_size,lr=args.lr,
        optimizer='AdamW weight_decay=1e-4, OneCycleLR cosine pct_start=0.08',label_smoothing=.05,balance_power=args.balance_power,sampler='WeightedRandomSampler weights=n_class^-power',
        init=str(OLD.relative_to(ROOT)),copied_classes=copied,n_classes=len(classes),train_crops=int(len(train_rows)),val_crops=int(len(val_rows)),
        selected_epoch=best[1],selected_macro_f1_main=best[0][0],threshold=chosen,criterion_met=met,seconds=time.time()-started,device=torch.cuda.get_device_name(0) if device.startswith('cuda') else 'cpu',
        torch=torch.__version__,test_used=False),indent=1),encoding='utf-8')
    print('DONE best epoch',best[1],best[0],'threshold',chosen['threshold'],flush=True)


if __name__=='__main__':main()
