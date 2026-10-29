"""Box tables, class list and background crops for recognizer v2 (TRAIN/VAL/TEST kept separate).

Reads V1.2 semantic labels and the AI-reviewed extension; writes only local files under output/.
Background crops are random image regions that overlap no labelled box of any disposition
(readable, unreadable or out-of-ontology): IoU < 0.1 with every box and under 30% of any box covered.
"""
from __future__ import annotations
import argparse,json,math,random,sys
from pathlib import Path
import cv2,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.perception.recognizer_v2_labels import (BACKGROUND,build_boxes,build_class_list,read_extension_labels,read_semantic_labels)
LABELS=ROOT/'data/audit/semantic/final_v1_2/semantic_labels_v1_2.csv'
EXTENSION=ROOT/'data/manifests/dataset_v2/ai_ontology_extension_labels.jsonl'
LOC=ROOT/'data/processed/canonical/itas_mm_v1/localization'
OUT=ROOT/'output/kiet_wp_20261028/recognizer_v2'
SEED=20261028
BACKGROUND_COUNTS={'TRAIN':4000,'VAL':800,'TEST':800}
OLD_CHECKPOINT=ROOT/'models/sign_recognizer/retrained_20261009.pt'


def overlaps(crop,boxes):
    x1,y1,x2,y2=crop;area=(x2-x1)*(y2-y1)
    for a,b,c,d in boxes:
        inter=max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))
        if not inter:continue
        box_area=(c-a)*(d-b);union=area+box_area-inter
        if inter/union>=.1 or inter/max(1e-9,box_area)>=.3:return True
    return False


def background_crops(labels,split,count,seed):
    """Random crops 16-200 px (log-uniform side) outside every labelled box of the split's images."""
    rng=random.Random(seed);per_image={}
    for row in labels[labels.split==split].itertuples():
        per_image.setdefault(row.image_path,[]).append(tuple(json.loads(row.bbox_xyxy)))
    paths=sorted(per_image);rng.shuffle(paths);result=[]
    for path in paths:
        if len(result)>=count:break
        image=cv2.imread(str(LOC/path))
        if image is None:continue
        h,w=image.shape[:2];top=min(200,h,w)
        if top<16:continue
        for _ in range(40):
            side=int(round(math.exp(rng.uniform(math.log(16),math.log(top)))))
            x=rng.randint(0,w-side);y=rng.randint(0,h-side);crop=(x,y,x+side,y+side)
            if not overlaps(crop,per_image[path]):
                result.append(dict(image_path=path,split=split,bbox_xyxy=json.dumps([x,y,x+side,y+side]),short_side_px=float(side),code=BACKGROUND,source='random_background'));break
    if len(result)<count:raise RuntimeError(f'Only {len(result)} {split} backgrounds')
    return pd.DataFrame(result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    import torch
    old=torch.load(OLD_CHECKPOINT,map_location='cpu',weights_only=True)['classes']
    labels=read_semantic_labels(LABELS);extension_all=[json.loads(l) for l in EXTENSION.read_text(encoding='utf-8').splitlines()]
    extension=read_extension_labels(EXTENSION);boxes=build_boxes(labels,extension)
    result=build_class_list(boxes,old)
    counts=result.pop('counts');counts['category']=['main' if c in result['main'] else 'low_support' if c in result['low_support'] else 'untrained' for c in counts.index]
    counts['in_old_30']=[c in result['old_classes'] for c in counts.index]
    counts['class_index']=[result['classes'].index(c) if c in result['classes'] else -1 for c in counts.index]
    counts.to_csv(out/'class_counts.csv',index_label='code')
    boxes.to_csv(out/'boxes.csv',index=False)
    both_not_qcvn=[r['canonical_box_id'] for r in extension_all if r['pass_a']==r['pass_b']=='not_qcvn_sign']
    not_sign=labels[labels.canonical_box_id.isin(both_not_qcvn)].assign(code=BACKGROUND,source='ai_not_qcvn_both_passes')
    backgrounds=pd.concat([background_crops(labels,s,n,SEED+i) for i,(s,n) in enumerate(BACKGROUND_COUNTS.items())],ignore_index=True)
    background_boxes=pd.concat([backgrounds,not_sign[['image_path','split','bbox_xyxy','short_side_px','code','source']]],ignore_index=True)
    background_boxes.to_csv(out/'background_boxes.csv',index=False)
    report=dict(seed=SEED,labels=str(LABELS.relative_to(ROOT)),extension=str(EXTENSION.relative_to(ROOT)),min_side_px=16,
        main_rule='>=30 TRAIN and >=5 VAL boxes with short side >=16 px',classes=result['classes'],n_classes=len(result['classes']),
        main=result['main'],low_support=result['low_support'],untrained=result['untrained'],dropped_old=result['dropped_old'],
        boxes=int(len(boxes)),boxes_by_split=boxes.groupby('split').size().to_dict(),boxes_by_source=boxes.groupby('source').size().to_dict(),
        background=background_boxes.groupby(['split','source']).size().rename('n').reset_index().to_dict('records'))
    (out/'class_list.json').write_text(json.dumps(report,ensure_ascii=False,indent=1),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('n_classes','boxes','boxes_by_split','boxes_by_source','dropped_old')},ensure_ascii=False))
    print('main',len(result['main']),'low',len(result['low_support']),'untrained',result['untrained'])


if __name__=='__main__':main()
