"""Audit externally downloaded VietSpeedYOLO dataset before any training.

Retains source splits and hashes. Canonical TEST never becomes training data.
Perceptual near-matches are *suspects*, not automatic verified duplicates.
The dataset only exhaustively labels its two R420/R421 classes; never use its
unannotated imagery as background for all Vietnamese traffic signs.
"""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,cv2,numpy as np,datetime

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/raw/external/vietspeedyolo_hf_20261011'
DEST=ROOT/'data/audit/external_vietspeedyolo_20261011'
DEST.mkdir(exist_ok=True,parents=True)
if not(SRC/'PROVENANCE.json').is_file():
    raise RuntimeError('Source download is incomplete / missing pinned provenance')
srcinfo=json.loads((SRC/'PROVENANCE.json').read_text(encoding='utf-8'))
canon=ROOT/'data/processed/canonical/itas_mm_v1/localization/images'

def hash_image(path):
    image=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if image is None or image.size==0:return None,None
    small=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(9,8),interpolation=cv2.INTER_AREA)
    bits=small[:,:-1]>small[:,1:]
    val=0
    for b in bits.ravel():val=(val<<1)|int(b)
    return image,val

def validate_label(path):
    problems=[];codes=Counter()
    if not path.is_file():
        return codes,['missing_label_file']
    for i,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines()):
        if not line.strip():continue
        chunks=line.strip().split()
        try:
            if len(chunks)!=5:raise ValueError('expected YOLO 5-tuples')
            c=int(chunks[0])
            if c not in (0,1):raise ValueError('unexpected class')
            values=[float(x) for x in chunks[1:]]
            if not all(np.isfinite(v) for v in values):raise ValueError('nonfinite')
            x,y,w,h=values
            if not (0<x<1 and 0<y<1 and 0<w<=1 and 0<h<=1):
                raise ValueError('nonpositive or out of range')
            if x-w/2 < -1e-4 or x+w/2>1.0001 or y-h/2<-1e-4 or y+h/2>1.0001:
                raise ValueError('bbox outside image')
            codes[c]+=1
        except Exception as exc:problems.append('line_'+str(i+1)+':'+str(exc))
    return codes,problems

entries=[];problems=[];codes=Counter();rawhash={};hashes_by_split=defaultdict(set)
for split in ('train','val'):
    folder=SRC/'images'/split
    imgs=sorted([p for p in folder.glob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png')])
    for j,p in enumerate(imgs):
        img,dh=hash_image(p)
        if img is None:
            problems.append((str(p),'bad_image'))
            continue
        code_count,issues=validate_label(SRC/'labels'/split/(p.stem+'.txt'))
        if issues:problems.append((str(p),issues))
        codes.update(code_count)
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in rawhash:
            problems.append((str(p),'same_image_sha_in_source:'+rawhash[digest]))
        rawhash[digest]=str(p.relative_to(SRC))
        hashes_by_split[split].add(digest)
        entries.append({'file':str(p.relative_to(SRC)).replace('\\','/'),
            'labels':f'labels/{split}/{p.stem}.txt','source_split':split,
            'itag':'EXTERNAL_UNREVIEWED_TRAIN_CANDIDATE',
            'sha256':digest,'dhash64':f'{dh:016x}',
            'dimensions':[img.shape[1],img.shape[0]],
            'class_counts':{str(k):v for k,v in code_count.items()},
            'label_issues':issues})
        if j%150==0:print('SCANNED',split,j,'/',len(imgs),flush=True)
# Compare to immutable canonical hashes encoded in their file stem.
canonical_splits={split:{p.stem for p in (canon/split).glob('*.jpg')}
                  for split in ('train','val','test')}
canonical_sha={k:[e['file'] for e in entries if e['sha256'] in hashes] for k,hashes in canonical_splits.items()}
raw_overlap=hashes_by_split['train']&hashes_by_split['val']
# Near-duplicate comparison within source and against untouched canonical TEST.
near_split=[];near_test=[]
candidates=[(int(e['dhash64'],16),e['file'],e['source_split']) for e in entries]
for i,(a,p,split) in enumerate(candidates):
    for b,q,other in candidates[:i]:
        dist=(a^b).bit_count()
        if split!=other and dist<=4:
            near_split.append({'a':p,'b':q,'dhash_distance':dist})
# Canonical TEST only: do not read its labels or train on its images.
existing=[]
test_folder=canon/'test'
for idx,p in enumerate(test_folder.glob('*.jpg')):
    _,h=hash_image(p)
    if h is not None:existing.append((h,p.stem))
    if idx%1000==0:print('CANONICAL_TEST_SCANNED',idx,flush=True)
for a,p,_ in candidates:
    for b,stem in existing:
        dist=(a^b).bit_count()
        if dist<=4:
            near_test.append({'external':p,'canonical_test_sha':stem,'dhash_distance':dist})
report={'source':srcinfo,'scan_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'images_total':len(entries),
        'source_splits':dict(Counter(e['source_split'] for e in entries)),
        'class_counts':{str(k):v for k,v in codes.items()},
        'decode_and_yolo_annotation_problems':problems[:100],
        'problems_count':len(problems),
        'exact_duplicate_source_images':len(entries)-len(rawhash),
        'train_val_same_sha':len(raw_overlap),
        'exact_sha_overlap_canonical':{k:len(v) for k,v in canonical_sha.items()},
        'canonical_test_count_scanned':len(existing),
        'near_duplicate_cross_source_split_candidates':near_split[:100],
        'near_duplicate_canonical_test_candidates':near_test[:100],
        'near_duplicate_cross_split_total':len(near_split),
        'near_duplicate_test_total':len(near_test),
        'record_count':len(entries),
        'approved_for_training':False,
        'independent_test':False,
        'risk':'R420/R421 only. Source labels may be incomplete for other traffic signs. Do not use unannotated frames as universal negative samples. Source may include augmentations; ensure route/session disjointness independently.'}
with (DEST/'external_manifest.jsonl').open('w',encoding='utf-8') as f:
    for e in entries:f.write(json.dumps(e,ensure_ascii=False)+'\n')
(DEST/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('AUDIT_RESULT '+json.dumps({k:v for k,v in report.items() if k not in ('source','decode_and_yolo_annotation_problems','near_duplicate_cross_source_split_candidates','near_duplicate_canonical_test_candidates')},ensure_ascii=False),flush=True)
