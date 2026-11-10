"""Run OCR exact-text scoring on human verified, source-disjoint TEST panels.

Use isolated .ocr-venv (RapidOCR CPU). Fail closed without adequate GT.
No model or TTS settings changed. No legal scope interpretation.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluation.ocr_metrics import choose_consensus,score_verified_text
from src.evaluation.field_metrics import field_gate

def load_verified(corpus):
    corpus=Path(corpus).resolve()
    meta=json.loads((corpus/'report.json').read_text(encoding='utf-8'))
    if meta.get('schema_version')!='itasmm-field-corpus-v1':
        raise ValueError('Not a reviewed field corpus')
    lines=(corpus/'frames.jsonl').read_text(encoding='utf-8').splitlines()
    rows=[json.loads(z) for z in lines if z.strip()]
    if len(rows)!=meta['frames']:raise ValueError('Corpus count mismatch')
    gt=[]
    for f in rows:
        if not f.get('reviewer') or not f.get('reviewed_at'):raise ValueError('Missing reviewer')
        if f['split']!='TEST':continue
        path=Path(f['image_path'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=f['image_sha256']:
            raise ValueError('Image hash mismatch')
        for s in f['signs']:
            if s.get('plate_text_verified') is True:
                if s.get('judgment')!='sign' or s.get('panel_role')!='supplementary' or not str(s.get('plate_text','')).strip():
                    raise ValueError('Invalid claimed plate GT')
                gt.append({'frame_id':f['frame_id'],'source_group':f['source_group'],
                    'image_path':str(path),'bbox_xyxy':s['bbox_xyxy'],
                    'true_text':s['plate_text'],'human_verified':True})
    return rows,gt

def extract_views(image,bbox):
    import cv2
    h,w=image.shape[:2]
    x1,y1,x2,y2=bbox
    a=max(0,int(x1));b=max(0,int(y1));c=min(w,int(x2));d=min(h,int(y2))
    patch=image[b:d,a:c]
    if patch.size==0 or min(patch.shape[:2])<8:return []
    large=cv2.resize(patch,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
    lab=cv2.cvtColor(large,cv2.COLOR_BGR2LAB)
    L,A,B=cv2.split(lab)
    L=cv2.createCLAHE(clipLimit=2.,tileGridSize=(5,5)).apply(L)
    enhanced=cv2.cvtColor(cv2.merge((L,A,B)),cv2.COLOR_LAB2BGR)
    return [('3x',large),('contrast_3x',enhanced)]

def run(gt):
    import cv2
    from rapidocr import RapidOCR
    engine=RapidOCR()
    observations=[]
    for index,row in enumerate(gt):
        image=cv2.imread(row['image_path'])
        if image is None:raise ValueError('Reviewed image unreadable')
        views=[]
        for name,im in extract_views(image,row['bbox_xyxy']):
            result=engine(im)
            views.append({'name':name,
                'text':list(getattr(result,'txts',[]) or []),
                'scores':[float(s) for s in (getattr(result,'scores',[]) or [])]})
        predicted=choose_consensus(views,threshold=.85,min_agreement=2)
        observations.append({**row,'predicted_text':predicted,'ocr_views':views})
        if (index+1)%10==0:print(f'OCR tested {index+1}/{len(gt)} plates',flush=True)
    return observations

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--corpus',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--evaluate',action='store_true')
    ap.add_argument('--minimum-verified-plates',type=int,default=10)
    args=ap.parse_args()
    if args.report.exists():raise FileExistsError('No overwrite of previous OCR audits')
    if args.minimum_verified_plates<1:raise ValueError('Invalid plate minimum')
    rows,gt=load_verified(args.corpus)
    # OCR needs source-disjoint verification, even if lane annotations are
    # missing; independent legal/speed/lane metrics have separate gates.
    test_groups={r['source_group'] for r in rows if r['split']=='TEST'}
    train_groups={r['source_group'] for r in rows if r['split']!='TEST'}
    hashes={r['source_video_sha256'] for r in rows if r['split']=='TEST' and r.get('source_video_sha256')}
    blocking=[]
    if len(test_groups)<2:blocking.append('need_two_independent_TEST_routes')
    if test_groups & train_groups:blocking.append('route_overlap_TRAIN_VAL_TEST')
    if len(hashes)<len(test_groups):blocking.append('missing_or_duplicate_original_video_hashes')
    if len(gt)<args.minimum_verified_plates:blocking.append('insufficient_human_verified_plate_transcriptions')
    report={'schema_version':'itasmm-field-ocr-evaluation-v1',
        'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_corpus':str(args.corpus.resolve()),
        'source_test_groups':len(test_groups),
        'human_verified_test_panels':len(gt),
        'blockers':blocking,
        'ocr_applicability':'transcribed text only, NOT automatic legal scope or TTS',
        'independent_test_runnable':not blocking,
        'run_status':'PRECHECK_ONLY'}
    if args.evaluate and blocking:
        report['run_status']='BLOCKED_NO_INDEPENDENT_OCR_GT'
    elif args.evaluate:
        records=run(gt)
        report['run_status']='RESEARCH_OCR_EVALUATION_COMPLETE'
        report['metrics']=score_verified_text(records)
        report['per_plate']=records
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='per_plate'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
