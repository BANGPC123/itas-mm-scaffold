"""Audit existing Vietnam lane corpus without promoting AI draft annotations.

Verifies all 600 image SHA256 hashes, temporal split grouping and review state.
Creates reviewer priority index from disagreement, contrast and no-line frames.
"""
from pathlib import Path
import json,hashlib,collections,csv,datetime
ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'data/lane_vietnam_review_20261011'
OUT=ROOT/'output/evaluation/vietnam_lane_corpus_audit_20261013'
OUT.mkdir(parents=True,exist_ok=True)
original=json.loads((PACK/'manifest.json').read_text(encoding='utf-8'))
rows=original['frames']
assert len(rows)==600 and len({r['id'] for r in rows})==600
drafts={r['id']:r for r in (json.loads(line) for line in (PACK/'ai_drafts.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())}
assert len(drafts)==600
source_groups={}
errors=[]
stats=collections.Counter()
review=[]
for r in rows:
    p=PACK/r['image']
    if not p.is_file():errors.append((r['id'],'missing_image'));continue
    hashval=hashlib.sha256(p.read_bytes()).hexdigest()
    if hashval!=r['sha256'] or drafts[r['id']]['image_sha256']!=hashval:
        errors.append((r['id'],'hash_mismatch'));continue
    split=r['split'];grp=r['temporal_group']
    if grp in source_groups and source_groups[grp]!=split:errors.append((r['id'],'temporal_split_leak'))
    source_groups[grp]=split
    stats[split]+=1
    draft=drafts[r['id']]
    if not (PACK/draft['mask']).is_file():errors.append((r['id'],'mask_missing'))
    n1,n2=draft['ai_u_lines'],draft['ai_v2_lines']
    stats['human_confirmed']+=int(bool(r['human_confirmed']))
    stats['all_no_model_lane']+=int(n1==0 and n2==0)
    stats['disagreement']+=int((n1==0)!=(n2==0))
    stats['both_positive']+=int(n1>0 and n2>0)
    priority=0
    priority+=3 if (n1==0)!=(n2==0) else 0
    priority+=2 if n1==0 and n2==0 else 0
    priority+=2 if r['features'].get('contrast',100)<35 else 0
    priority+=2 if r['features'].get('lower_dark_ratio',0)>.45 else 0
    priority+=min(2,abs(n1-n2))
    review.append({'id':r['id'],'split':split,'timestamp_s':r['source_video_seconds'],
      'temporal_group':grp,'priority':priority,
      'ufld_ai_lines':n1,'yolopv2_ai_lines':n2,
      'human_confirmed':r['human_confirmed'],'image':r['image']})
if errors:raise RuntimeError('CORPUS_INTEGRITY_ERRORS: '+repr(errors[:12]))
review.sort(key=lambda x:(-x['priority'],x['timestamp_s']))
csvpath=OUT/'review_priority.csv'
with csvpath.open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(review[0]))
    writer.writeheader();writer.writerows(review)
manifest={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'integrity':'600_OF_600_SOURCE_HASHES_AND_AI_MASKS_VERIFIED',
 'frames':len(rows),'source_video_count':1,'group_count':len(source_groups),
 'counts':dict(stats),
 'human_confirmed':stats['human_confirmed'],
 'is_independent_video_test':False,
 'ai_drafts_are_gt':False,'can_train_supervised_on_vietnam':stats['human_confirmed']>=50,
 'can_report_vietnam_lane_f1':False,
 'holdout_source_same_as_train':True,
 'review_ui':str(PACK/'review_lane_600.html'),
 'review_priority_csv':str(csvpath),
 'next_required_action':'Review and confirm paint lane, road edge and true no-lane, then import masks through gate.'}
(OUT/'audit.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('LANE_CORPUS_AUDIT',json.dumps(manifest,ensure_ascii=False),flush=True)
