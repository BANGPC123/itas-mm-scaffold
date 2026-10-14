"""Check dataset additions provenance, duplicate hashes, and split isolation.

Development-only negative crops must never leak into canonical TEST.
The source video itself is not an independent held-out unit.
"""
from pathlib import Path
from collections import Counter
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    folder=ROOT/'data/audit/ai_removed_boxes_hardneg_20261011'
    data=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    rows=data['images']
    assert all(x['split']=='TRAIN_CANDIDATE_ONLY' for x in rows)
    assert all(x['review_type']=='AI_ONLY_NOT_HUMAN_CONFIRMED' for x in rows)
    assert all(x['source_group']=='test_video_development' for x in rows)
    sha=[];bad=[]
    for row in rows:
        f=folder/row['image']
        if not f.is_file():bad.append([row['image'],'missing']);continue
        digest=hashlib.sha256(f.read_bytes()).hexdigest()
        if digest!=row['source_sha256']:bad.append([row['image'],'hash mismatch'])
        sha.append(digest)
    duplicates={k:v for k,v in Counter(sha).items() if v>1}
    report={'count':len(rows),'distinct_file_sha256':len(set(sha)),
        'content_duplicates':duplicates,'bad':bad,
        'human_verified':0,'source_group_count':1,
        'train_candidate_only':True,'no_test_data_used_to_curate':False,
        'caveat':'These crops were mined from a previously evaluated DEVELOPMENT video; cannot serve as independent validation. Original canonical TEST is untouched.',
        'status':'pass' if not bad else 'fail'}
    out=ROOT/'output/evaluation/hardneg_provenance_audit_20261011.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
    if bad:raise SystemExit(1)
if __name__=='__main__':main()
