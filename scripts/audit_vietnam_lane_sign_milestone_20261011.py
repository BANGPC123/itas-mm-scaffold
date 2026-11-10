"""Milestone audit for Vietnam lane data and sign/supplementary reasoning.

Never promotes AI-only annotations into ground-truth or claims one route is
independent TEST. Produces machine-readable handoff for human review.
"""
from pathlib import Path
import json,hashlib,collections,datetime
from src.context.sign_applicability import SignApplicability
from src.utils.types import SignDetection,LaneInfo
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/lane_vietnam_review_20261011'
OUT=ROOT/'output/evaluation/VN_LANE_SIGN_MILESTONE_20261011.json'
src=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
rows=src['frames']
drafts=[json.loads(l) for l in (BASE/'ai_drafts.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
assert len(rows)==600 and len(drafts)==600 and len({r['id'] for r in rows})==600
assert len({r['source_video_id'] for r in rows})==1
assert not any(x.get('human_confirmed') for x in rows)
assert sum(1 for r in rows if not (BASE/r['image']).is_file())==0
split=collections.Counter(r['split'] for r in rows)
assert dict(split)=={'TRAIN':400,'VAL_INTERNAL':100,'TEST_HOLDOUT_SAME_VIDEO':100}
def sign(code,xy,**kwargs):return SignDetection(code,.98,xy,**kwargs)
def assessment(vehicle,with_plate):
    primary=sign('P.131a',(100,100,210,210),panel_role='primary',placement='roadside',
                 lane_scope='all',context_verified=True,label_verified=True)
    signs=[primary]
    if with_plate:
        signs.append(sign('S.505a',(115,219,193,251),panel_role='supplementary',
                 allowed_vehicles=['car'],context_verified=True,label_verified=True))
    return SignApplicability().apply(signs,LaneInfo([]),(720,1280,3),vehicle)[0].applicability
cases={k:{v:assessment(v,k=='primary_plus_car_plate') for v in ('car','truck','motorbike')}
       for k in ('primary_only','primary_plus_car_plate')}
assert cases['primary_only']=={'car':'applies','truck':'applies','motorbike':'applies'}
assert cases['primary_plus_car_plate']=={'car':'applies','truck':'ignored','motorbike':'ignored'}
segfile=ROOT/'output/retraining_20261011/segformer_culane_b0/training_report.json'
seg=json.loads(segfile.read_text(encoding='utf-8')) if segfile.is_file() else None
verified=ROOT/'data/processed/lane_human_verified_20261011/verified_manifest.jsonl'
report={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'status':'DATASET_AND_REVIEW_UI_READY_TRAINING_ON_VIETNAM_BLOCKED_BY_HUMAN_LABELS',
    'vehicle_scope_reviewed_examples':cases,
    'source_video':src['source_video'],'source_sha256':src['source_sha256'],
    'dataset':{'frames':len(rows),'splits':dict(split),'ai_drafts':len(drafts),
      'human_confirmed':sum(bool(x.get('human_confirmed')) for x in rows),
      'independent_routes':1,'heldout_is_independent_route':False,
      'all_frame_paths_exist':True},
    'review_ui':str(BASE/'review_lane_600.html'),
    'approved_mask_manifest_exists':verified.exists(),
    'segformer_culane_baseline':seg,
    'remaining':['Human must inspect/confirm real paint lines, road edges and no-lane cases before supervised Vietnam fine-tuning.',
                 'Split by truly different video routes for independent TEST.',
                 'CLRNet needs separate environment compatibility checks; cannot claim a model is trained merely from published scores.',
                 'Temporal tracking and false-positive rates require audited lane GT; no pseudo-GT promotion.']}
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('MILESTONE_READY',json.dumps({'status':report['status'],'splits':dict(split),
 'approved_human':report['dataset']['human_confirmed'],
 'vehicle_scope_examples':cases,'review_ui':report['review_ui']},ensure_ascii=False))
