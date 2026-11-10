"""TEST evaluation of recognizer v2 against the 2026-10-09 recognizer on identical ground-truth crops.

TEST is used for reporting only: nothing here selects a checkpoint, threshold or class list.
Labels on TEST are AI-assisted / ai_reviewed, not human ground truth. Prediction = raw argmax.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import f1_score
from torchvision.models import mobilenet_v3_small
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.perception.recognizer_v2_crops import build_cache,exact_crop,load_cache,save_cache
from src.perception.sign_recognizer import crop_tensor
OUT=ROOT/'output/kiet_wp_20261028/recognizer_v2'
LOC=ROOT/'data/processed/canonical/itas_mm_v1/localization'
OLD=ROOT/'models/sign_recognizer/retrained_20261009.pt'
GROUPS=[('<16',0,16),('16-32',16,32),('32-64',32,64),('>=64',64,1e9)]
SEED=20261028


def load_model(path):
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    model=mobilenet_v3_small(num_classes=len(ckpt['classes']));model.load_state_dict(ckpt['state_dict']);return model.eval().cuda(),ckpt


def predict(model,classes,entries,batch=256):
    out=[]
    with torch.inference_mode():
        for s in range(0,len(entries),batch):
            x=torch.stack([crop_tensor(exact_crop(e)) for e in entries[s:s+batch]]).cuda();out.append(model(x).softmax(1).cpu())
    prob=torch.cat(out);conf,idx=prob.max(1)
    return np.array([classes[i] for i in idx.tolist()]),conf.numpy()


def group_of(side):
    return next(name for name,lo,hi in GROUPS if lo<=side<hi)


def metrics(true,pred):
    labels=sorted(set(true))
    return dict(n=int(len(true)),classes=len(labels),accuracy=float((true==pred).mean()) if len(true) else None,
        macro_f1=float(f1_score(true,pred,labels=labels,average='macro',zero_division=0)) if len(true) else None)


def table(df,pred_col):
    rows=[]
    for name,lo,hi in GROUPS+[('>=32',32,1e9),('all',0,1e9)]:
        sub=df[(df.short_side_px>=lo)&(df.short_side_px<hi)]
        rows.append(dict(group=name,**metrics(sub.code.to_numpy(),sub[pred_col].to_numpy())))
    return rows


def bootstrap(df,draws=1000):
    """Paired image-level bootstrap of new-minus-old accuracy on the same crops."""
    rng=np.random.default_rng(SEED);images=df.image_path.to_numpy();uniq,inv=np.unique(images,return_inverse=True)
    names=['<16','16-32','32-64','>=64','>=32','all'];agg=np.zeros((len(uniq),len(names),3))
    for r,i in zip(df.itertuples(),inv):
        g=[group_of(r.short_side_px)];g+=['>=32'] if r.short_side_px>=32 else [];g+=['all']
        for name in g:
            k=names.index(name);agg[i,k]+=[1,r.old_pred==r.code,r.new_pred==r.code]
    total=agg.sum(0);result={}
    diffs=np.empty((draws,len(names)))
    for d in range(draws):
        w=np.bincount(rng.integers(0,len(uniq),len(uniq)),minlength=len(uniq));s=(w[:,None,None]*agg).sum(0)
        diffs[d]=(s[:,2]-s[:,1])/np.maximum(1,s[:,0])
    for k,name in enumerate(names):
        point=(total[k,2]-total[k,1])/max(1,total[k,0])
        lo,hi=np.percentile(diffs[:,k],[2.5,97.5]);result[name]=dict(n=int(total[k,0]),old_accuracy=float(total[k,1]/max(1,total[k,0])),new_accuracy=float(total[k,2]/max(1,total[k,0])),
            difference=float(point),ci95=[float(lo),float(hi)],images=int((agg[:,k,0]>0).sum()))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,default=OUT/'train/candidate.pt');ap.add_argument('--output',type=Path,default=OUT/'eval')
    args=ap.parse_args();args.candidate=args.candidate.resolve();out=args.output;out.mkdir(parents=True,exist_ok=True)
    report=json.loads((OUT/'class_list.json').read_text(encoding='utf-8'));main_classes=report['main'];low=report['low_support']
    old30=[c for c in torch.load(OLD,map_location='cpu',weights_only=True)['classes'] if c!='unknown']
    boxes=pd.read_csv(OUT/'boxes.csv',keep_default_na=False);test=boxes[boxes.split=='TEST'].reset_index(drop=True)
    cache=OUT/'crop_cache_test_all.pkl'
    entries=load_cache(cache) if cache.is_file() else None
    if entries is None or len(entries)!=len(test):entries=build_cache(test,LOC);save_cache(entries,cache)
    keep=[i for i,e in enumerate(entries) if e is not None];test=test.iloc[keep].reset_index(drop=True);entries=[entries[i] for i in keep]
    old_model,old_ckpt=load_model(OLD);new_model,new_ckpt=load_model(args.candidate)
    test['old_pred'],test['old_conf']=predict(old_model,old_ckpt['classes'],entries);test['new_pred'],test['new_conf']=predict(new_model,new_ckpt['classes'],entries)
    test.drop(columns=['bbox_xyxy']).to_csv(out/'test_predictions.csv',index=False)
    a=test[test.code.isin(old30)];b=test[test.code.isin(main_classes)&~test.code.isin(old30)];c=test[test.code.isin(low)]
    res=dict(candidate=str(args.candidate.relative_to(ROOT)),label_note='TEST labels are AI-assisted (released V1.2 codes, ai_reviewed extension); not human ground truth.',
        selection_note='No checkpoint, threshold or class choice used TEST.',n_test_crops=int(len(test)),
        a_old30={'n_crops':int(len(a)),'old_model':table(a,'old_pred'),'new_model':table(a,'new_pred'),'bootstrap_new_minus_old':bootstrap(a)},
        b_new_main={'n_crops':int(len(b)),'classes':sorted(b.code.unique().tolist()),'new_model':table(b,'new_pred')},
        c_low_support=[],untrained_test_support=test[~test.code.isin(set(main_classes)|set(low))].code.value_counts().to_dict())
    for code in low:
        sub=c[c.code==code];res['c_low_support'].append(dict(code=code,test_support=int(len(sub)),correct=int((sub.new_pred==code).sum()),
            accuracy=float((sub.new_pred==code).mean()) if len(sub) else None,images=int(sub.image_path.nunique()),reliable=bool(len(sub)>=5)))
    per=[]
    for code in sorted(test.code.unique(),key=lambda x:(x not in old30,x)):
        sub=test[test.code==code];row=dict(code=code,category='old30' if code in old30 else 'main' if code in main_classes else 'low_support' if code in low else 'untrained',support=int(len(sub)),images=int(sub.image_path.nunique()),
            new_correct=int((sub.new_pred==code).sum()),new_accuracy=float((sub.new_pred==code).mean()))
        if code in old30:row.update(old_correct=int((sub.old_pred==code).sum()),old_accuracy=float((sub.old_pred==code).mean()))
        per.append(row)
    res['per_class']=per;pd.DataFrame(per).to_csv(out/'per_class.csv',index=False)
    (out/'results.json').write_text(json.dumps(res,indent=1,ensure_ascii=False),encoding='utf-8')
    for name in ('<16','16-32','32-64','>=64','>=32','all'):
        bt=res['a_old30']['bootstrap_new_minus_old'][name];print(f"old30 {name:6s} n={bt['n']:5d} old={bt['old_accuracy']:.4f} new={bt['new_accuracy']:.4f} diff={bt['difference']:+.4f} CI={bt['ci95'][0]:+.4f}..{bt['ci95'][1]:+.4f}")


if __name__=='__main__':main()
