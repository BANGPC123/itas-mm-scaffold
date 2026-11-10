"""Build reproducible frozen references; never train on canonical VAL/TEST."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,math,cv2,numpy as np,pandas as pd,torch
from src.perception.sign_recognizer import SignRecognizer,crop_tensor
from src.utils.config import load_config


def select_positive_rows(boxes,classes,per_class=96,min_side=16):
    """TRAIN boxes of trained codes (never the background class), at most `per_class` per code in box-id order."""
    codes=[c for c in classes if c!='unknown'];train=boxes[(boxes.split=='TRAIN')&(boxes.short_side_px>=min_side)&boxes.code.isin(codes)]
    parts=[train[train.code==c].sort_values('canonical_box_id').head(per_class) for c in codes]
    return pd.concat([p for p in parts if len(p)],ignore_index=True)


def fit_negative_records(records):
    """AI-reviewed not-a-sign crops reserved for the bank; held-out ones only select the margin."""
    return [r for r in records if r['label']=='not_a_sign' and r['split']=='fit']


def negative_crop(record,video_root):
    """Tight floor/ceil crop of a reviewed negative from its video frame or TRAIN image."""
    if record['source']=='video':
        cap=cv2.VideoCapture(str(Path(video_root)/f"{record['video_id']}.mp4"));cap.set(cv2.CAP_PROP_POS_FRAMES,record['frame']);ok,image=cap.read();cap.release()
    else:image=cv2.imread(str(ROOT/'data/processed/canonical/itas_mm_v1/localization'/record['image_path']));ok=image is not None
    if not ok:raise RuntimeError(f"Cannot read negative source {record['id']}")
    x1,y1,x2,y2=record['bbox_xyxy'];h,w=image.shape[:2]
    return image[max(0,math.floor(y1)):min(h,math.ceil(y2)),max(0,math.floor(x1)):min(w,math.ceil(x2))]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_v2(args):
    """Gate bank for the 88-class recognizer: TRAIN positives, AI-reviewed negatives (80% fit split); no VAL/TEST crops."""
    torch.set_num_threads(4);cv2.setNumThreads(2)
    cfg=load_config('perception');cfg['sign_recognizer'].pop('reference_gate_path',None);cfg['sign_recognizer']['weights_path']=str(args.recognizer)
    rec=SignRecognizer(cfg)
    def embed(crops):
        values=[]
        for start in range(0,len(crops),64):
            batch=torch.stack([crop_tensor(c,rec.size) for c in crops[start:start+64]]).to(rec.device)
            with torch.inference_mode():
                f=rec.model.avgpool(rec.model.features(batch)).flatten(1);f=torch.nn.functional.normalize(rec.model.classifier[:-1](f),dim=1)
            values.append(f.cpu().numpy())
        return np.concatenate(values)
    boxes=pd.read_csv(args.boxes_csv,keep_default_na=False);rows=select_positive_rows(boxes,rec.classes,args.per_class)
    crops,labels,ids=[],[],[]
    for row in rows.itertuples():
        image=cv2.imread(str(ROOT/'data/processed/canonical/itas_mm_v1/localization'/row.image_path))
        x1,y1,x2,y2=json.loads(row.bbox_xyxy);crop=image[max(0,math.floor(y1)):math.ceil(y2),max(0,math.floor(x1)):math.ceil(x2)]
        if crop.size and min(crop.shape[:2])>=16:crops.append(crop);labels.append(row.code);ids.append(row.canonical_box_id)
    positive=embed(crops)
    records=fit_negative_records([json.loads(l) for l in args.negatives_jsonl.read_text(encoding='utf-8').splitlines() if l.strip()])
    negative=embed([negative_crop(r,args.video_root) for r in records])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,positive=positive,labels=np.asarray(labels),negative=negative,
        negative_targets=np.asarray(['*']*len(records)),negative_kinds=np.asarray(['irrelevant']*len(records)))
    report={'positive_reference_split':'TRAIN only','positive_count':len(labels),'positive_box_ids':ids,'negative_count':len(records),
        'negative_sources':{s:sum(r['source']==s for r in records) for s in ('video','train')},'negative_ids':[r['id'] for r in records],
        'frozen_encoder':str(args.recognizer),'encoder_sha256':sha256_file(args.recognizer),'negatives_sha256':sha256_file(args.negatives_jsonl),
        'boxes_csv_sha256':sha256_file(args.boxes_csv),'canonical_val_test_used_for_references':False,'development_video_v0_used':False,
        'human_gt_for_negatives':False,'negative_provenance':'ai_reviewed'}
    args.output.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('REFERENCE_BANK_V2',len(labels),len(records),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--negative-manifest',type=Path);ap.add_argument('--source',type=Path)
    ap.add_argument('--recognizer',type=Path,help='v2 mode: checkpoint whose encoder embeds the references')
    ap.add_argument('--boxes-csv',type=Path);ap.add_argument('--negatives-jsonl',type=Path);ap.add_argument('--video-root',type=Path)
    ap.add_argument('--per-class',type=int,default=96);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.recognizer:
        if not (args.boxes_csv and args.negatives_jsonl and args.video_root):ap.error('--recognizer needs --boxes-csv, --negatives-jsonl and --video-root')
        return build_v2(args)
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
