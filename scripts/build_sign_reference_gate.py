"""Build reproducible frozen references; never train on canonical VAL/TEST."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,math,cv2,numpy as np,pandas as pd,torch
from src.perception.sign_recognizer import SignRecognizer,crop_tensor
from src.utils.config import load_config


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--negative-manifest',type=Path);ap.add_argument('--source',type=Path);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    cfg=load_config('perception');cfg['sign_recognizer'].pop('reference_gate_path',None)
    rec=SignRecognizer(cfg)
    def embed(crops):
        values=[]
        for start in range(0,len(crops),64):
            batch=torch.stack([crop_tensor(c,rec.size) for c in crops[start:start+64]]).to(rec.device)
            with torch.inference_mode():
                f=rec.model.avgpool(rec.model.features(batch)).flatten(1)
                f=torch.nn.functional.normalize(rec.model.classifier[:-1](f),dim=1)
            values.append(f.cpu().numpy())
        return np.concatenate(values)
    labels=[];crops=[];positive_ids=[]
    gt=pd.read_csv(ROOT/'data/audit/semantic/final_v1_1/semantic_labels_v1_1.csv',keep_default_na=False)
    train=gt[(gt.split=='TRAIN')&gt.semantic_final_disposition.str.startswith('RELEASED')]
    for code in rec.classes:
        group=train[train.canonical_sign_code==code].sort_values('canonical_box_id').head(96)
        for row in group.itertuples():
            image=cv2.imread(str(ROOT/'data/processed/canonical/itas_mm_v1/localization'/row.image_path))
            x1,y1,x2,y2=json.loads(row.bbox_xyxy);crop=image[max(0,math.floor(y1)):math.ceil(y2),max(0,math.floor(x1)):math.ceil(x2)]
            if crop.size and min(crop.shape[:2])>=16:crops.append(crop);labels.append(code);positive_ids.append(row.canonical_box_id)
    positive=embed(crops)
    scan=ROOT/'output/video_completion_20261011/full_v1'
    # Indices refer to the frozen AI-inspected contact sheet. Exclude small or
    # ambiguous crops; retained rows visibly show ads/HUD or vehicle restrictions.
    reviewed=[0,1,2,3,4,*range(6,27),30,31,32,33]
    irrelevant={3,4,6,7,8,14,30}
    if args.negative_manifest:
        manifest=json.loads(args.negative_manifest.read_text(encoding='utf-8'))
        if not args.source:ap.error('--source is required with --negative-manifest')
        if manifest.get('canonical_val_test_used_for_references') is not False:raise ValueError('References must exclude VAL/TEST')
        if manifest['encoder_sha256']!=hashlib.sha256((ROOT/'models/sign_recognizer/retrained_20261009.pt').read_bytes()).hexdigest():raise ValueError('Different frozen encoder')
        if set(manifest['positive_box_ids'])!=set(positive_ids):raise ValueError('TRAIN references differ from the frozen manifest')
        source=str(args.source)
        entries=[(e['contact_index'],{'timestamp':e['timestamp'],'sign':{'bbox_xyxy':e['bbox_xyxy'],'label':e['old_prediction']},'evidence':e}) for e in manifest['negative_evidence']]
        expected_source=manifest.get('source_sha256')
        if expected_source:
            with args.source.open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=expected_source:raise ValueError('Different negative source video')
    else:
        tracks=json.loads((scan/'qa/known_tracks.json').read_text(encoding='utf-8'))
        special=[r for r in tracks if r['sign']['label'] in ['P.127','DP.134','P.123a']]
        source=json.loads((scan/'summary.json').read_text(encoding='utf-8'))['source']
        entries=[(index,special[index]) for index in reviewed]
    cap=cv2.VideoCapture(source)
    negative_crops=[];negative_evidence=[];negative_targets=[];negative_kinds=[]
    for index,row in entries:
        cap.set(cv2.CAP_PROP_POS_MSEC,row['timestamp']*1000);ok,image=cap.read()
        if not ok:raise RuntimeError('Cannot decode reviewed crop')
        x1,y1,x2,y2=map(round,row['sign']['bbox_xyxy']);crop=image[y1:y2,x1:x2]
        target=row['evidence']['veto_target'] if 'evidence' in row else 'P.123a' if index==0 else 'P.127' if index in (1,2) else '*'
        kind=row['evidence']['kind'] if 'evidence' in row else 'irrelevant' if index in irrelevant else 'unsupported'
        negative_targets.append(target)
        negative_kinds.append(kind)
        negative_crops.append(crop);negative_evidence.append({'contact_index':index,'timestamp':row['timestamp'],
            'bbox_xyxy':row['sign']['bbox_xyxy'],'old_prediction':row['sign']['label'],
            'veto_target':target,'kind':kind,'review':'AI visual inspection: wrong arrow direction, non-speed numeric restriction, advertising/HUD or unsupported vehicle restriction',
            'human_verified':False,'independent_test':False})
    cap.release();negative=embed(negative_crops);args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,positive=positive,labels=np.asarray(labels),negative=negative,
        negative_targets=np.asarray(negative_targets),negative_kinds=np.asarray(negative_kinds))
    report={'positive_reference_split':'TRAIN only','positive_count':len(labels),'positive_box_ids':positive_ids,
        'negative_evidence':negative_evidence,'negative_count':len(negative_crops),
        'frozen_encoder':'models/sign_recognizer/retrained_20261009.pt',
        'encoder_sha256':hashlib.sha256((ROOT/'models/sign_recognizer/retrained_20261009.pt').read_bytes()).hexdigest(),
        'canonical_val_test_used_for_references':False,'human_gt_for_development_negatives':False,
        'new_classes_assigned':False}
    args.output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('REFERENCE_BANK',len(labels),len(negative_crops),flush=True)


if __name__=='__main__':main()
