"""Export four visually verified development-video false positives.

These crops are *training candidates*, NOT independently reviewed ground truth.
Source is the unannotated LEFT half of archived paired contact sheets.
"""
from pathlib import Path
import json,cv2,hashlib
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'output/video_tests/test_video_20261010_after_hardening'
out=ROOT/'data/audit/development_hard_negatives_20261011'
out.mkdir(parents=True,exist_ok=True)
rows=[json.loads(x) for x in (base/'predictions.jsonl').open(encoding='utf-8') if x.strip()]
cases=[
    (55.333,'P.127','extra_01.jpg',0,'dashboard_gauge_a'),
    (337.333,'P.127','extra_04.jpg',1,'dashboard_gauge_b'),
    (209.0,'I.409','pair_03.jpg',0,'commercial_logo'),
    (286.667,'W.210','extra_03.jpg',1,'vehicle_tail_lamp')
]
manifest=[]
for stamp,code,sheet,rownum,name in cases:
    item=min(rows,key=lambda x:abs(x['timestamp_seconds']-stamp))
    assert abs(item['timestamp_seconds']-stamp)<.02
    signs=[s for s in item['signs'] if s['label']==code]
    assert signs,(stamp,code)
    sheetpath=base/'visual_review_20261011'/sheet
    scene=cv2.imread(str(sheetpath))
    assert scene is not None and scene.shape[1]>=1560 and scene.shape[0]>=880
    source=scene[rownum*440:(rownum+1)*440,:780].copy()
    bbox=signs[0]['bbox_xyxy']
    x1,y1,x2,y2=[float(v) for v in bbox]
    x1*=780/1280;x2*=780/1280;y1*=440/720;y2*=440/720
    cx=(x1+x2)/2;cy=(y1+y2)/2
    halfw=max(9.,(x2-x1)*.68);halfh=max(9.,(y2-y1)*.68)
    a=max(0,int(cx-halfw));b=max(0,int(cy-halfh))
    c=min(source.shape[1],int(cx+halfw+1));d=min(source.shape[0],int(cy+halfh+1))
    crop=source[b:d,a:c]
    assert crop.shape[0]>=10 and crop.shape[1]>=10
    path=out/(name+'.png')
    assert cv2.imwrite(str(path),crop)
    manifest.append({'id':name,'label':'background','model_predicted':code,'time_seconds':stamp,
        'source_sheet':str(sheetpath.relative_to(ROOT)),'roi_xyxy_in_sheet':[a,b,c,d],
        'file':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'review':'assistant visual review from development video; not independent or blind QA',
        'split':'TRAIN_CANDIDATE_ONLY','width':crop.shape[1],'height':crop.shape[0]})
(out/'manifest.json').write_text(json.dumps({'samples':manifest,'count':len(manifest),
    'limitations':'All from the SAME development video; do NOT evaluate generalization on them or claim independent accuracy.'},
    ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'count':len(manifest),'items':[(x['id'],x['width'],x['height']) for x in manifest],
    'manifest':str(out/'manifest.json')},ensure_ascii=False),flush=True)
