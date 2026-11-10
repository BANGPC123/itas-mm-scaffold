"""Calibrate acceptance on VAL synthetic conditions after candidate selection."""
import importlib.util,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('robust_training',ROOT/'scripts/finetune_recognizer_robustness.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def main():
    torch.set_num_threads(4)
    out=ROOT/'output/evaluation/improvements_20261009/recognizer_small'
    checkpoint=torch.load(out/'candidate.pt',map_location='cpu',weights_only=True)
    model=mobilenet_v3_small(num_classes=len(checkpoint['classes'])).cuda().eval();model.load_state_dict(checkpoint['state_dict'])
    df=pd.read_csv(ROOT/'data/audit/semantic/final_v1_1/recognition_manifest_v1.csv',keep_default_na=False)
    outcomes={}
    for kind in ['clean','blur','small','small_blur']:
        conf=[];correct=[]
        with torch.inference_mode():
            for x,y in DataLoader(module.Crops(df[df.split=='VAL'],condition=kind),batch_size=64):
                p=model(x.cuda()).softmax(1).cpu();v,i=p.max(1);conf.extend(v.tolist());correct.extend((i==y).tolist())
        outcomes[kind]=(np.array(conf),np.array(correct))
    curve=[]
    for threshold in np.linspace(.5,.99,50):
        values={}
        for kind,(confidence,correct) in outcomes.items():
            keep=confidence>=threshold
            values[kind]={'coverage':float(keep.mean()),'accepted_accuracy':float(correct[keep].mean()) if keep.any() else 0.}
        curve.append({'threshold':float(threshold),'conditions':values})
    usable=[r for r in curve if all(v['accepted_accuracy']>=.95 and v['coverage']>=.2 for v in r['conditions'].values())]
    selected=min(usable,key=lambda r:r['threshold']) if usable else None
    report={'split':'VAL','selected':selected,'curve':curve,'ood_validated':False,'independent_holdout_verified':False}
    if selected:
        target=ROOT/'models/sign_recognizer/robust_val.pt'
        checkpoint['confidence_threshold']=selected['threshold'];torch.save(checkpoint,target)
        report['checkpoint']=str(target.relative_to(ROOT))
    (out/'calibration.json').write_text(json.dumps(report,indent=2));print(json.dumps({'selected':selected,'checkpoint':report.get('checkpoint')},indent=2))

if __name__=='__main__':main()
