"""Choose acceptance on known-class VAL; no TEST or user-video labels used."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'output/retraining_20261009/recognizer'

def main():
    torch.set_num_threads(2)
    spec=importlib.util.spec_from_file_location('robust_calibration',ROOT/'scripts/finetune_recognizer_robustness.py')
    robust=importlib.util.module_from_spec(spec);spec.loader.exec_module(robust)
    checkpoint=torch.load(OUT/'candidate.pt',map_location='cpu',weights_only=True)
    model=mobilenet_v3_small(num_classes=len(checkpoint['classes'])).eval();model.load_state_dict(checkpoint['state_dict'])
    df=pd.read_csv(ROOT/'data/audit/semantic/final_v1_1/recognition_manifest_v1.csv',keep_default_na=False);df=df[df.split=='VAL']
    outcomes={}
    for kind in ['clean','blur','small','small_blur']:
        ys=[];ps=[];cs=[]
        with torch.inference_mode():
            for x,y in DataLoader(robust.Crops(df,condition=kind),batch_size=64):
                c,p=model(x).softmax(1).max(1);ys.extend(y.tolist());ps.extend(p.tolist());cs.extend(c.tolist())
        outcomes[kind]=(np.array(ys),np.array(ps),np.array(cs))
    curve=[]
    for threshold in np.linspace(.5,.99,50):
        conditions={}
        for kind,(y,p,c) in outcomes.items():
            keep=(p<30)&(c>=threshold)
            conditions[kind]=dict(coverage=float(keep.mean()),accepted_accuracy=float((y[keep]==p[keep]).mean()) if keep.any() else 0.)
        curve.append(dict(threshold=float(threshold),conditions=conditions))
    usable=[r for r in curve if all(c['coverage']>=.8 and c['accepted_accuracy']>=.99 for c in r['conditions'].values())]
    selected=min(usable,key=lambda r:r['threshold']) if usable else curve[0]
    checkpoint['confidence_threshold']=selected['threshold'];torch.save(checkpoint,OUT/'candidate.pt')
    result=dict(selected=selected,curve=curve,criterion_met=bool(usable),split='VAL',
        criterion='All four conditions: accepted accuracy >=99%, coverage >=80%; smallest passing threshold, otherwise retain .5',
        video_used=False,test_used=False,background_note='Increasing threshold cannot increase background false acceptance')
    (OUT/'calibration.json').write_text(json.dumps(result,indent=2));print(json.dumps(selected),flush=True)

if __name__=='__main__':main()
