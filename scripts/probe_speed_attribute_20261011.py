"""Prototype-only speed-number recognition from frozen MobileNetV3 embeddings.

ONLY train and val P.127 records with speed_training_eligible TRUE. No TEST.
Do not infer a driving/legal speed command. Keeps recognizer backbone unchanged.
"""
from pathlib import Path
import json, math, time
from collections import Counter
import cv2,numpy as np,pandas as pd,torch
from torchvision.models import mobilenet_v3_small
from src.perception.sign_recognizer import crop_tensor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
import joblib

root=Path(__file__).resolve().parents[1]
out=root/'output/evaluation/speed_attribute_probe_20261011'
out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
df=pd.read_csv(root/'data/audit/semantic/final_v1_1/recognition_manifest_v1.csv')
df=df[(df.canonical_sign_code=='P.127')&(df.speed_training_eligible.astype(str).str.lower()=='true')&(df.split.isin(['TRAIN','VAL']))].copy()
assert len(df)==788,(len(df))
train=df[df.split=='TRAIN'].reset_index(drop=True)
val=df[df.split=='VAL'].reset_index(drop=True)
assert len(train)==639 and len(val)==149,(len(train),len(val))
assert not set(train.canonical_box_id)&set(val.canonical_box_id)
ckpt=torch.load(root/'models/sign_recognizer/retrained_20261009.pt',weights_only=True,map_location='cpu')
model=mobilenet_v3_small(num_classes=len(ckpt['classes']))
model.load_state_dict(ckpt['state_dict'])
model.cuda().eval()
image_root=root/'data/processed/canonical/itas_mm_v1/localization'
def extract(rows):
    tensors=[];labels=[]
    for row in rows.itertuples():
        im=cv2.imread(str(image_root/row.image_path))
        if im is None:raise RuntimeError('Could not decode image')
        h,w=im.shape[:2]
        x1,y1,x2,y2=json.loads(row.bbox_xyxy)
        crop=im[max(0,math.floor(y1)):min(h,math.ceil(y2)),max(0,math.floor(x1)):min(w,math.ceil(x2))]
        if crop.size==0:raise RuntimeError('Crop missing '+row.canonical_box_id)
        tensors.append(crop_tensor(crop))
        labels.append(int(row.speed_kmh))
    embeddings=[]
    with torch.inference_mode():
        for start in range(0,len(tensors),48):
            x=torch.stack(tensors[start:start+48]).cuda()
            z=model.avgpool(model.features(x)).flatten(1)
            embeddings.extend(z.cpu().numpy())
    return np.stack(embeddings),np.array(labels)
beg=time.perf_counter()
xtr,ytr=extract(train);xval,yval=extract(val)
print('FEATURES '+json.dumps({'train':list(xtr.shape),'val':list(xval.shape),'seconds':round(time.perf_counter()-beg,1)}),flush=True)
history=[]
for kernel,c in [('linear',.01),('linear',.1),('rbf',1),('rbf',10)]:
    clf=make_pipeline(StandardScaler(),SVC(kernel=kernel,C=c,probability=True,class_weight='balanced',random_state=391))
    clf.fit(xtr,ytr)
    pred=clf.predict(xval);conf=clf.predict_proba(xval).max(1)
    thresholds={}
    for threshold in [.5,.7,.8,.9]:
        accept=conf>=threshold
        thresholds[str(threshold)]={'coverage':float(accept.mean()),
            'accepted_accuracy':float((pred[accept]==yval[accept]).mean()) if accept.any() else None,
            'accepted_n':int(accept.sum())}
    score={'kernel':kernel,'C':c,'val_accuracy':float(accuracy_score(yval,pred)),
           'balanced_accuracy':float(balanced_accuracy_score(yval,pred)),
           'macro_f1':float(f1_score(yval,pred,average='macro',zero_division=0)),
           'selective':thresholds}
    history.append((score,clf))
    print('TRY '+json.dumps(score),flush=True)
winner,clf=max(history,key=lambda pair:pair[0]['macro_f1'])
if winner['macro_f1']>=.90 and winner['selective']['0.8']['coverage']>=.30 and winner['selective']['0.8']['accepted_accuracy']>=.97:
    joblib.dump(clf,out/'research_speed_classifier.joblib',compress=3)
    candidate=True
else:candidate=False
report={'source':'canonical P.127 TRAIN and VAL ONLY; human semantic evidence tiers vary',
        'scope':'attribute feasibility, not field/generalization, not safe-driving speed advice',
        'test_touched':False,'train_n':len(train),'val_n':len(val),
        'classes':sorted(map(int,set(ytr))),'candidates':[x[0] for x in history],
        'selected':winner,'model_saved_as_research_candidate':candidate,
        'deployed':False,'seconds':round(time.perf_counter()-beg,2)}
(out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('RESULT '+json.dumps({k:v for k,v in report.items() if k!='candidates'}),flush=True)
