"""Validate human-reviewed lane annotations and create 3-class training masks.

0=background,1=painted_lane,2=road_edge,255=unknown/occluded.
AI source proposals NEVER qualify as ground truth; reviewer/exhaustive
completeness and source-image SHA are mandatory.
"""
from pathlib import Path
import argparse,json,hashlib,collections,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/lane_vietnam_review_20261011'
OUT=ROOT/'data/processed/lane_human_verified_20261011'
KINDS={'lane_paint':1,'road_edge':2,'occluded':255}
def validate(review_path,require_minimum=True):
    src=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
    exported=json.loads(Path(review_path).read_text(encoding='utf-8'))
    if exported.get('schema')!=src['schema'] or exported.get('source_sha256')!=src['source_sha256']:
        raise ValueError('Source video provenance mismatch')
    candidates=exported.get('frames')
    if not isinstance(candidates,list) or len(candidates)!=len(src['frames']):
        raise ValueError('Frame count mismatch')
    valid=[];issues=[];counts=collections.Counter()
    for row,original in zip(candidates,src['frames']):
        if any(row.get(k)!=original[k] for k in ('id','sha256','split','temporal_group')):
            raise ValueError('Modified source identity or split: '+original['id'])
        if not row.get('human_confirmed'):continue
        if not row.get('reviewer','').strip() or not row.get('reviewed_at') or not row.get('human_exhaustive'):
            issues.append([row['id'],'missing reviewer/time/exhaustive']);continue
        lanes=row.get('human_lanes',[])
        if not isinstance(lanes,list):
            issues.append([row['id'],'invalid human_lanes']);continue
        if row.get('human_no_visible_lane') and any(p.get('kind')=='lane_paint' for p in lanes):
            issues.append([row['id'],'no_visible_lane conflicts with painted_lane']);continue
        if not row.get('human_no_visible_lane') and not any(p.get('kind')=='lane_paint' for p in lanes):
            issues.append([row['id'],'no lane_paint and not explicitly no_visible']);continue
        if any(p.get('source')!='human_drawn' or p.get('kind') not in KINDS or
               not isinstance(p.get('points'),list) or len(p['points'])<2 or
               any(not isinstance(pt,(list,tuple)) or len(pt)!=2 or
                 not 0<=pt[0]<row['width'] or not 0<=pt[1]<row['height']
                 for pt in p['points']) for p in lanes):
            issues.append([row['id'],'malformed or AI-as-human annotation']);continue
        path=BASE/row['image']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=original['sha256']:
            issues.append([row['id'],'source jpeg SHA mismatch']);continue
        counts[row['split']]+=1;valid.append(row)
    if issues:raise ValueError('Rejected marked-as-confirmed annotation(s): '+str(issues[:10]))
    if require_minimum and (counts['TRAIN']<50 or counts['VAL_INTERNAL']<15):
        raise ValueError('Need >=50 human-reviewed TRAIN and >=15 human-reviewed VAL_INTERNAL frames; have '+str(dict(counts)))
    return src,valid,counts

def build(export_path):
    src,rows,counts=validate(export_path)
    if (OUT/'verified_manifest.jsonl').exists():raise FileExistsError('Refuse overwrite')
    (OUT/'masks').mkdir(parents=True,exist_ok=True)
    verified=[]
    for row in rows:
        h,w=row['height'],row['width']
        gt=np.zeros((h,w),np.uint8)
        # Draw lane and road edge with 5px line thickness at 960x540.
        for p in row['human_lanes']:
            if p['kind']=='occluded':continue
            points=np.asarray(p['points'],dtype=np.int32)
            cv2.polylines(gt,[points],False,KINDS[p['kind']],5,cv2.LINE_AA)
        # Ignore annotation scribbles for occlusions rather than hallucinating
        # invisible paint or treating occluders as confirmed background.
        for p in row['human_lanes']:
            if p['kind']=='occluded':
                cv2.polylines(gt,[np.asarray(p['points'],np.int32)],False,255,22)
        # cv2.LINE_AA may create invalid grayscale label IDs; enforce exact
        # integer class labels with nearest-value binning.
        allowed=np.array([0,1,2,255],np.int16)
        values=gt.astype(np.int16)
        index=np.abs(values[...,None]-allowed).argmin(axis=2)
        gt=allowed[index].astype(np.uint8)
        dst=OUT/'masks'/(row['id']+'.png')
        cv2.imwrite(str(dst),gt)
        verified.append({'id':row['id'],'split':row['split'],
            'temporal_group':row['temporal_group'],'image':str((BASE/row['image']).resolve()),
            'mask':str(dst.resolve()),
            'source_image_sha256':row['sha256'],
            'human_reviewer':row['reviewer'],'human_confirmed':True,
            'mask_colors':{'background':0,'lane_paint':1,'road_edge':2,'ignore':255},
            'source_video_sha256':src['source_sha256']})
    with (OUT/'verified_manifest.jsonl').open('w',encoding='utf-8') as fp:
        for e in verified:fp.write(json.dumps(e,ensure_ascii=False)+'\n')
    report={'status':'HUMAN_VERIFIED_LABELS_ONLY','counts':dict(counts),
            'source_groups':1,'independent_road_test':False,
            'same_video_holdout_not_fully_independent':True,
            'unverified_ai_promoted':0,'source_review_export':str(Path(export_path).resolve())}
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if args.validate_only:
        _,rows,counts=validate(args.review,require_minimum=False)
        print('VALIDATED',len(rows),dict(counts))
    else:build(args.review)
