"""Human-reviewed, source-disjoint Vietnam road evaluation CLI.

Preflight by default. --evaluate runs selected models only when the TEST gate
passes, and always labels output as research evaluation, not ADAS certification.
"""
from __future__ import annotations
from pathlib import Path
import argparse,json,sys,hashlib,datetime
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluation.field_metrics import field_gate,score_frames

def load_corpus(folder):
    folder=Path(folder).resolve()
    meta=folder/'report.json'
    frames=folder/'frames.jsonl'
    if not(meta.is_file() and frames.is_file()):raise ValueError('Missing compiled human-reviewed field corpus')
    report=json.loads(meta.read_text(encoding='utf-8'))
    if report.get('schema_version')!='itasmm-field-corpus-v1':
        raise ValueError('Unknown human field corpus schema')
    records=[json.loads(line) for line in frames.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records:raise ValueError('No human reviewed frames')
    if len(records)!=report.get('frames'):raise ValueError('Corpus report/frame count discrepancy')
    for row in records:
        if row.get('split') not in ('TRAIN','VAL','TEST') or not row.get('reviewer') or not row.get('reviewed_at'):
            raise ValueError('Unreviewed or invalid record in corpus')
        path=Path(row['image_path']).resolve()
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['image_sha256']:
            raise ValueError('Image missing or SHA256 changed')
    return records,report

def evaluate(records,road_bottom=.65):
    import cv2
    from src.pipeline.orchestrator import Pipeline
    model=Pipeline()
    output=[]
    for i,r in enumerate(records):
        if r.get('split')!='TEST':continue
        im=cv2.imread(r['image_path'])
        if im is None:raise RuntimeError('Unreadable reviewed frame '+r['frame_id'])
        result=model.run(im,10.7769,106.7009,False,road_bottom=road_bottom,annotate=False)
        # This is the *end-to-end* output, after decoder and safety rules,
        # not raw YOLO proposed boxes; explicitly name it accordingly.
        predictions=[{'bbox_xyxy':list(s.bbox_xyxy),
                      'label':s.label,'panel_role':s.panel_role,
                      'applicability':s.applicability}
                     for s in result.perception.signs]
        output.append({'gt':r,'pred':{'signs':predictions,'lanes':result.perception.lane.polylines},
                       'image_shape':im.shape})
        if (i+1)%25==0:print('Processed '+str(i+1)+' reviewed TEST frames',flush=True)
    return score_frames(output),len(output)

def main():
    ap=argparse.ArgumentParser()
    sources=ap.add_mutually_exclusive_group(required=True)
    sources.add_argument('--corpus',type=Path)
    sources.add_argument('--review-pack',type=Path,
                        help='Check uncompiled labeling pack without scoring it as field GT')
    ap.add_argument('--report',type=Path,required=True)
    ap.add_argument('--evaluate',action='store_true',help='Explicitly run on locked TEST frames when preflight gate passes')
    ap.add_argument('--road-bottom',type=float,default=.65)
    args=ap.parse_args()
    if not .3<=args.road_bottom<=1:raise ValueError('Invalid road_bottom')
    if args.report.exists():raise FileExistsError('Refuse overwriting prior evaluation report')
    if args.review_pack:
        from src.evaluation.label_review_schema import validate_review
        pack=args.review_pack.resolve()
        source=json.loads((pack/'review_manifest.json').read_text(encoding='utf-8'))
        possible=pack/'ai_ocr_advertising_review_20261011.json'
        if not possible.is_file():possible=pack/'review_manifest.json'
        current=json.loads(possible.read_text(encoding='utf-8'))
        checked=validate_review(current,source)
        gate={'field_evidence_ready':False,
              'blockers':['review_pack_is_not_source_disjoint_TEST',
                          'missing_compiled_human_confirmed_field_corpus'],
              'source_video_groups':1,'labels':checked}
        info={'schema_version':'itasmm-independent-field-eval-v1',
              'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'mode':'review_pack_preflight','source_review_pack':str(pack),
              'gate':gate,'run_status':'BLOCKED_INSUFFICIENT_INDEPENDENT_GT',
              'metrics':{'status':'NOT_COMPUTED; development-review data is not independent TEST'}}
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'status':info['run_status'],'gate':gate,'report':str(args.report)},ensure_ascii=False))
        return
    records,corpus=load_corpus(args.corpus)
    gate=field_gate(records)
    info={'schema_version':'itasmm-independent-field-eval-v1',
          'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'corpus':str(args.corpus.resolve()),
          'source_corpus_report':str((args.corpus/'report.json').resolve()),
          'scope':'research metrics, not QCVN/legal/ADAS field certification',
          'mode':'evaluate' if args.evaluate else 'preflight',
          'corpus_provenance':corpus.get('review_provenance','unknown'),
          'gate':gate,'run_status':'PRECHECK_ONLY'}
    if args.evaluate and not gate['field_evidence_ready']:
        info['run_status']='BLOCKED_INSUFFICIENT_INDEPENDENT_GT'
        info['metrics']={'status':'BLOCKED, no metrics computed'}
    elif args.evaluate:
        metrics,count=evaluate(records,args.road_bottom)
        info['run_status']='RESEARCH_EVALUATION_COMPLETED'
        info['tested_frames']=count
        info['metrics']=metrics
        info['ocr_exact_text_accuracy']={'status':'not_evaluable_without_reviewer_verified_plate_transcriptions'}
        info['legal_applicability_accuracy']={'status':'not_evaluable_without_independent_spatial_legal_GT'}
        info['tts_false_alerts_per_hour']={'status':'not_evaluable_on_still_images_without_annotated_time_video'}
    else:
        info['metrics']={'status':'NOT_COMPUTED; pass --evaluate and all preflight gates'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'run_status':info['run_status'],'gate':gate,'report':str(args.report)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
