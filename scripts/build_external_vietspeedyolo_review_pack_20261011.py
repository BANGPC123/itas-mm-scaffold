"""Create a compact, offline annotation pack from NEW external Vietnam sign data.

Each label is a source proposal, not independent human GT. The image pool only
contains 2 labeled classes; do not assume other sign classes are absent.
We do NOT train on unreviewed image-crop pseudo-backgrounds.
"""
from pathlib import Path
import json,cv2,random,hashlib,shutil,collections,numpy as np
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'data/raw/external/vietspeedyolo_hf_20261011'
MAN=ROOT/'data/processed/candidates/vietspeedyolo_hf_20261011/train_candidate_manifest.jsonl'
OUT=ROOT/'output/review/external_vietspeedyolo_20261011'
if OUT.exists():raise FileExistsError('Refuse to overwrite prior source review pack')
rows=[json.loads(l) for l in MAN.read_text(encoding='utf-8').splitlines()]
quality=[];skipped=collections.Counter()
for e in rows:
    w,h=e['dimensions']
    label=[z.strip() for z in (SRC/e['labels']).read_text().splitlines() if z.strip()]
    max_area=max([float(z.split()[3])*float(z.split()[4]) for z in label] or [0])
    # Very large isolated icons/screenshots heavily outnumber real road scenes.
    # This is a photo-looking-candidate filter only, not trained semantics.
    if w<600 or h<320 or w/h<1.25:
        skipped['aspect_or_resolution']+=1;continue
    if not (0.0002<=max_area<=.065):
        skipped['sign_covers_too_much_or_too_small']+=1;continue
    quality.append((e,max_area))
print('PHOTO_LIKE_POOL',len(quality),'OF',len(rows),'SKIP',dict(skipped),flush=True)
rng=random.Random(20261011)
rng.shuffle(quality)
quality=sorted(quality,key=lambda p:(p[0]['class_counts'].get('0',0)==0,p[1]))
# Balanced 18 R420 and 18 R421 when possible. Future visual QA may reject cartoons.
buckets={'R.420':[],'R.421':[]}
for e,area in quality:
    label='R.420' if e['class_counts'].get('0',0) else 'R.421'
    buckets[label].append((e,area))
sample=buckets['R.420'][:18]+buckets['R.421'][:18]
rng.shuffle(sample)
(OUT/'frames').mkdir(parents=True)
pack=[]
for i,(e,area) in enumerate(sample):
    im=cv2.imread(str(SRC/e['file']))
    h,w=im.shape[:2]
    target=OUT/'frames'/f'{i:04d}.jpg'
    cv2.imwrite(str(target),im,[cv2.IMWRITE_JPEG_QUALITY,94])
    sha=hashlib.sha256(target.read_bytes()).hexdigest()
    signs=[]
    for j,line in enumerate((SRC/e['labels']).read_text().splitlines()):
        if not line.strip():continue
        chunks=line.split()
        code='R.420' if chunks[0]=='0' else 'R.421'
        x,y,bw,bh=map(float,chunks[1:])
        x1=max(0.,(x-bw/2)*w);y1=max(0.,(y-bh/2)*h)
        x2=min(float(w),(x+bw/2)*w);y2=min(float(h),(y+bh/2)*h)
        signs.append({'id':f'source_sign_{j}',
            'bbox_xyxy':[round(a,2) for a in (x1,y1,x2,y2)],
            'label':code,'ai_suggested_label':code,'judgment':'unreviewed',
            'panel_role':'primary','source':'external_source_YOLO_annotation_not_human_reviewed',
            'source_annotation_class':chunks[0]})
    pack.append({'id':f'ext_frame_{i:04d}',
        'source_group':'hf_vietspeedyolo_UNRESOLVED_ORIGINAL_ROUTES',
        'source_image_sha256':e['sha256'],
        'timestamp_seconds':None,'image':'frames/'+target.name,
        'image_sha256':sha,'width':w,'height':h,
        'reviewer':'','reviewed_at':None,'human_confirmed':False,
        'all_visible_signs_annotated':False,'lane_status':'unreviewed',
        'signs':signs,'lanes':[],
        'notes':'Source is R.420/R.421 specific. OTHER traffic signs may not be labeled, check whole frame and lane separately.',
        'source_filename':e['file'],'source_license_claim':'MIT'})
manifest={'schema_version':'itasmm-field-review-v1',
          'provenance':'PUBLIC EXTERNAL YOLO ANNOTATIONS; NOT HUMAN VERIFIED; source photographer/route unknown',
          'external_repository':'NghiMe/vietspeedyolo','source_commit':json.loads((SRC/'PROVENANCE.json').read_text())['revision'],
          'class_scope_only':['R.420','R.421'],'frames':pack}
(OUT/'review_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
template=(ROOT/'output/review/field_review_pack_20261011/index.html').read_text(encoding='utf-8')
import re
serialized=json.dumps(manifest,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
template,n=re.subn(r'const manifest=.*?;\nconst original=',lambda m:'const manifest='+serialized+';\nconst original=',template,count=1,flags=re.S)
if n!=1:raise RuntimeError('Review template not found')
template=template.replace('Nguồn VIDEO PHÁT TRIỂN — không phải independent TEST',
'Nguồn VietSpeedYOLO ngoài dự án, ảnh gốc chưa xác minh tác giả; KHÔNG phải TEST độc lập')
(OUT/'index.html').write_text(template,encoding='utf-8')
report={'source':'NghiMe/vietspeedyolo','class_scope_only':['R.420','R.421'],
    'source_train_clean_candidates':len(rows),'road_looking_heuristic_pool':len(quality),
    'sample_review_frames':len(pack),'source_annotation_boxes':sum(len(x['signs']) for x in pack),
    'breakdown':dict(collections.Counter(s['label'] for f in pack for s in f['signs'])),
    'human_confirmed':0,'source_labels_exhaustive_for_all_signs':False,
    'source_route_provenance_unknown':True,'independent_test':False,
    'status':'SOURCE_LABELED_VISUAL_REVIEW_REQUIRED',
    'rejected_by_photo_like_heuristic':dict(skipped)}
(OUT/'review_pack_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('EXTERNAL_REVIEW_PACK '+json.dumps(report,ensure_ascii=False),flush=True)
