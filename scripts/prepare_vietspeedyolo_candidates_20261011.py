"""Prepare a vetted TRAIN_CANDIDATE list, but never deploy/train automatically.

Conservatively excludes exact duplicates, near-duplicates of source VAL and
immutable canonical TEST, including near-duplicates within TRAIN.
No original file or annotation modified. No pseudolabel converted to GT.
"""
from pathlib import Path
import json,cv2,hashlib,random,numpy as np
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw/external/vietspeedyolo_hf_20261011'
AUDIT=ROOT/'data/audit/external_vietspeedyolo_20261011'
entries=[json.loads(line) for line in (AUDIT/'external_manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
summary=json.loads((AUDIT/'audit.json').read_text(encoding='utf-8'))
if summary['near_duplicate_test_total'] or summary['exact_sha_overlap_canonical']['test']:
    raise ValueError('Cannot prepare source training until canonical TEST leakage is investigated')
val=[e for e in entries if e['source_split']=='val']
train=[e for e in entries if e['source_split']=='train']
val_hashes=[int(x['dhash64'],16) for x in val]
blocked=Counter();chosen=[];accepted_hashes=[];seen_sha=set()
for e in sorted(train,key=lambda x:x['file']):
    if e['label_issues']:
        blocked['bad_label']+=1;continue
    if e['sha256'] in seen_sha:
        blocked['exact_duplicate_train']+=1;continue
    seen_sha.add(e['sha256'])
    dh=int(e['dhash64'],16)
    if any((dh^v).bit_count()<=4 for v in val_hashes):
        blocked['near_source_val']+=1;continue
    if any((dh^v).bit_count()<=3 for v in accepted_hashes):
        blocked['near_duplicate_train']+=1;continue
    accepted_hashes.append(dh)
    chosen.append(e)
assert all(e['source_split']=='train' for e in chosen)
assert all(set(e['class_counts'])<=set(['0','1']) for e in chosen)
assert len(chosen)>0
DATASET_ID='vietspeedyolo_hf_20261011'
dest=ROOT/'data/processed/candidates'/DATASET_ID
if dest.exists():
    raise FileExistsError('Refuse overwrite of candidate dataset')
dest.mkdir(parents=True)
with (dest/'train_candidate_manifest.jsonl').open('w',encoding='utf-8') as f:
    for e in chosen:
        record={**e,
                'source_repo':'NghiMe/vietspeedyolo',
                'source_revision':json.loads((RAW/'PROVENANCE.json').read_text(encoding='utf-8'))['revision'],
                'label_class_mapping':{'0':'R.420','1':'R.421'},
                'split':'TRAIN_CANDIDATE_ONLY','source_group':'vietspeedyolo_HF_UNRESOLVED_ROUTES',
                'source_image_path':str((RAW/e['file']).resolve()),
                'source_labels_path':str((RAW/e['labels']).resolve()),
                'approved_by_human':False,
                'not_general_background':True}
        f.write(json.dumps(record,ensure_ascii=False)+'\n')
counts=Counter()
for e in chosen:
    for k,v in e['class_counts'].items():counts[k]+=v
report={'source':DATASET_ID,'source_repo':'https://huggingface.co/datasets/NghiMe/vietspeedyolo',
        'image_downloaded':summary['images_total'],
        'source_train_images':len(train),'source_val_images':len(val),
        'selected_train_candidates':len(chosen),
        'exclusion_counts':dict(blocked),
        'class_annotations_in_selected':dict(counts),
        'canonical_test_exact_or_near_duplicate_flags':0,
        'train_vs_source_val_max_dhash_similarity_threshold':4,
        'train_deduplication_dhash_distance_at_most':3,
        'source_split_VAL_is_NOT_independent_GROUND_TRUTH':True,
        'source_video_route_groups_unknown':True,
        'license_source_claim':'MIT, check actual image copyright before commercial reuse',
        'status':'TRAIN_CANDIDATE_ONLY_UNAPPROVED',
        'original_raw_files_modified':False,
        'reference':'All source data remains in data/raw/external/vietspeedyolo_hf_20261011'}
(dest/'selection_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
# Visual proof-of-annotation: save small 5x4 montage from deterministic sample.
random.Random(391).shuffle(chosen)
sample=chosen[:20]
tiles=[]
for e in sample:
    im=cv2.imread(str(RAW/e['file']))
    label=(RAW/e['labels']).read_text(encoding='utf-8-sig').splitlines()
    H,W=im.shape[:2]
    for line in label:
        if not line.strip():continue
        parts=line.split();c=int(parts[0]);x,y,w,h=map(float,parts[1:])
        x1=int((x-w/2)*W);y1=int((y-h/2)*H)
        x2=int((x+w/2)*W);y2=int((y+h/2)*H)
        cv2.rectangle(im,(x1,y1),(x2,y2),(50,230,40) if c==0 else (30,200,240),3)
        cv2.putText(im,('R420' if c==0 else 'R421'),(max(0,x1),max(19,y1-3)),
                    cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
    tile=cv2.resize(im,(330,205),interpolation=cv2.INTER_AREA)
    footer=np.full((28,330,3),24,np.uint8)
    cv2.putText(footer,e['file'][:35],(6,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
    tiles.append(np.vstack([tile,footer]))
while len(tiles)<20:tiles.append(np.full((233,330,3),25,np.uint8))
cv2.imwrite(str(dest/'source_annotation_preview_20.jpg'),
            np.vstack([np.hstack(tiles[i:i+5]) for i in (0,5,10,15)]),
            [cv2.IMWRITE_JPEG_QUALITY,92])
print('TRAIN_CANDIDATE '+json.dumps(report,ensure_ascii=False),flush=True)
