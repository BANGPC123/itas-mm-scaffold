"""Re-score and integrity-audit existing source-disjoint evaluation evidence.

This checks provenance and arithmetic without presenting AI drafts as
independent human labels or claiming new model inference.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/evaluation/independent_holdout_audit_20261014.json'


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def audit_lane_review(path):
    root=ROOT/path
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    errors=[];count=0;reviewed=0;groups={}
    for row in manifest['frames']:
        image=root/row['image']
        if not image.is_file() or digest(image)!=row['sha256']:
            errors.append(row['id'])
        count+=1
        reviewed+=int(bool(row.get('human_confirmed')))
        grp=row.get('temporal_group')
        if grp in groups and groups[grp]!=row['split']:
            errors.append('group_leakage:'+str(grp))
        groups[grp]=row['split']
    return {'source':str(path),'frames':count,'images_sha256_verified':len(errors)==0,
        'bad_items':errors[:12], 'human_verified':reviewed,
        'independent_source_groups':manifest.get('video_sources_total'),
        'approved_for_training':manifest.get('approved_for_training',False)}


def audit_culane():
    root=ROOT/'output/evaluation/revalidation_20261008/culane_full'
    r=json.loads((root/'results.json').read_text(encoding='utf-8'))
    identity=json.loads((root/'run_identity.json').read_text(encoding='utf-8'))
    paths={'scripts/evaluate_culane.py':ROOT/'scripts/evaluate_culane.py',
           'src/perception/ufld_v2.py':ROOT/'src/perception/ufld_v2.py',
           'checkpoint':ROOT/'models/lane/culane_res18.pth'}
    matches={name:digest(path)==identity[name] for name,path in paths.items()}
    seen=set();totals={name:[0,0,0] for name in ('raw_adapter','deployed_filter')}
    count=0;dupes=0
    with (root/'frames.jsonl').open(encoding='utf-8') as file:
        for line in file:
            frame=json.loads(line);count+=1
            if frame['image'] in seen:dupes+=1
            seen.add(frame['image'])
            for mode in totals:
                v=frame[mode]
                for i in range(3):totals[mode][i]+=int(v[i])
    recomputed={}
    for mode,(tp,fp,fn) in totals.items():
        x=r['metrics'][mode]['all']
        recomputed[mode]={'tp':tp,'fp':fp,'fn':fn,
            'count_matches':(tp,fp,fn)==(x['tp'],x['fp'],x['fn']),
            'f1':2*tp/(2*tp+fp+fn)}
    return {'dataset':'CULane TEST human labels; not independent Vietnamese route',
       'frames':count,'unique_frames':len(seen),'duplicate_frames':dupes,
       'expected':r['expected_test_images'],'complete':r['complete_test'],
       'source_identity_matches_current':matches,'recomputed':recomputed,
       'historical_f1_is_current_code_measurement':all(matches.values()),
       'replay_not_new_inference':True}


def audit_sign():
    p=ROOT/'docs/evidence/video_quality_20261013/sign_test.json'
    record=json.loads(p.read_text(encoding='utf-8'))
    v=record['results']['rescue_12']
    return {'split':record['split'],'images':record['images'],
       'released_GT':v['released_gt'],'correct':v['correct'],
       'recognition_recall_recomputed':v['correct']/v['released_gt'],
       'precision_recomputed':v['correct']/v['evaluable_known'],
       'tiny_under16_recall_recomputed':v['lt16_correct']/v['lt16_gt'],
       'source':'Previously frozen labeled canonical TEST, not a new Vietnam field benchmark',
       'untouched_by_tuning_today':True}


def main():
    corpus=[audit_lane_review('data/lane_vietnam_review_20261011'),
            audit_lane_review('data/lane_external_source_review_20261011')]
    culane=audit_culane()
    sign=audit_sign()
    gt=sum(r['human_verified'] for r in corpus)
    ok=(all(r['images_sha256_verified'] for r in corpus) and
        culane['complete'] and culane['frames']==culane['expected']
        and culane['duplicate_frames']==0 and
        culane['source_identity_matches_current']['checkpoint'] and
        all(s['count_matches'] for s in culane['recomputed'].values()))
    report={'audit_integrity_passed':ok,'culane_test':culane,'sign_test':sign,
      'vietnam_lane_review':corpus,'source_disjoint_field_lane_GT_exists':gt>0,
      'independent_vietnam_field_accuracy_can_be_claimed':False,
      'full_product_field_validation_complete':False,
      'classification':'research software operational; road safety/legal certification not established'}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':ok,'culane_frames':culane['frames'],
         'lane_gt_human_verified':gt,'sign_test_images':sign['images'],
         'output':str(OUT)},ensure_ascii=False),flush=True)
    if not ok:raise SystemExit('Integrity gate failed')


if __name__=='__main__':main()
