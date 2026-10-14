"""Import self-declared human-reviewed pack exports with source-group split isolation.

Run only after reviewers actually submit JSON. This NEVER mutates canonical data.
No group can occur in two splits; no duplicated image SHA across source groups.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluation.label_review_schema import validate_review

def build(entries,output):
    if not isinstance(entries,list) or not entries:raise ValueError('Empty entries list')
    if output.exists():raise FileExistsError('Refuse overwriting a prior review corpus')
    split_of={};seen_hashes={};video_source_split={};records=[];report=[]
    for item in entries:
        pack=Path(item['pack']).resolve();submitted=Path(item['review']).resolve()
        split=item.get('split')
        if split not in ('TRAIN','VAL','TEST'):raise ValueError('Must explicitly assign TRAIN / VAL / TEST')
        manifest=json.loads((pack/'review_manifest.json').read_text(encoding='utf-8'))
        document=json.loads(submitted.read_text(encoding='utf-8'))
        score=validate_review(document,manifest)
        video_digest=manifest.get('source_video_sha256')
        if video_digest is not None:
            if (not isinstance(video_digest,str) or len(video_digest)!=64 or
                any(c not in '0123456789abcdefABCDEF' for c in video_digest)):
                raise ValueError('Invalid original video SHA256')
            previous=video_source_split.setdefault(video_digest.lower(),split)
            if previous!=split:
                raise ValueError('Identical source video appears across TRAIN / VAL / TEST')
        if not score['human_confirmed']:
            raise ValueError('No human self-attested frames: '+str(submitted))
        original={f['id']:f for f in manifest['frames']}
        for f in document['frames']:
            if not f.get('human_confirmed'):continue
            group=f['source_group']
            if group in split_of and split_of[group]!=split:
                raise ValueError('Same source session in multiple splits: '+group)
            split_of[group]=split
            img=(pack/f['image']).resolve()
            if not img.is_relative_to(pack) or not img.is_file():
                raise ValueError('Unsafe/missing source image')
            digest=hashlib.sha256(img.read_bytes()).hexdigest()
            if digest!=f['image_sha256']:raise ValueError('Source image file modified')
            if digest in seen_hashes:
                raise ValueError('Duplicate image across review packs/groups')
            seen_hashes[digest]=group
            records.append({
                'image_path':str(img),'image_sha256':digest,'frame_id':f['id'],
                'source_group':group,'source_video_sha256':video_digest,'split':split,
                'reviewer':f['reviewer'],'reviewed_at':f['reviewed_at'],
                'timestamp_seconds':f.get('timestamp_seconds'),
                'all_visible_signs_annotated':f['all_visible_signs_annotated'],
                'lane_status':f['lane_status'],'lanes':f['lanes'],
                'signs':[{'id':s['id'],'judgment':s['judgment'],'label':s['label'],
                          'bbox_xyxy':s['bbox_xyxy'],'panel_role':s['panel_role'],
                          'plate_text':s.get('plate_text',''),
                          'plate_text_verified':bool(s.get('plate_text_verified',False))}
                        for s in f['signs'] if s['judgment']!='unreviewed'],
                'notes':f.get('notes','')})
        report.append({'pack':str(pack),'review':str(submitted),'split':split,'summary':score})
    output.mkdir(parents=True)
    with (output/'frames.jsonl').open('w',encoding='utf-8') as out:
        for record in records:out.write(json.dumps(record,ensure_ascii=False)+'\n')
    by_split=Counter(x['split'] for x in records)
    docs={'schema_version':'itasmm-field-corpus-v1',
         'review_provenance':'Reviewer self-attested, not external auditor-certified',
         'frames':len(records),'split_counts':dict(by_split),'source_groups':split_of,
         'source_video_hashes_recorded':sum(bool(r.get('source_video_sha256')) for r in records),
         'exhaustive_frames':sum(r['all_visible_signs_annotated'] is True for r in records),
         'annotated_ego_frames':sum(any(p['role'] in ('ego_left','ego_right') for p in r['lanes']) for r in records),
         'confirmed_sign_instances':sum(s['judgment']=='sign' for r in records for s in r['signs']),
         'confirmed_negative_instances':sum(s['judgment']=='not_sign' for r in records for s in r['signs']),
         'packs':report,'separate_field_test_ready':bool(by_split['TEST'] and by_split['TRAIN'] and by_split['VAL']),
         'WARNING':'Only complete/exhaustive frames may contribute to detection recall; all unreviewed/uncertain pixels are not background labels.'}
    (output/'report.json').write_text(json.dumps(docs,ensure_ascii=False,indent=2),encoding='utf-8')
    return docs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--entries',type=Path,required=True,
                    help='JSON list [{"pack":"...","review":"...","split":"TRAIN"}, ...]')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    d=build(json.loads(args.entries.read_text(encoding='utf-8')),args.output.resolve())
    print(json.dumps({k:v for k,v in d.items() if k not in ('packs','WARNING')},ensure_ascii=False))
if __name__=='__main__':main()
