"""Independently sourced Vietnam urban-video lane-test candidate review.

Uses 2 Wikimedia videos from different authors/routes, CC0 and CC BY-SA.
Only the source is independent; lane GT is NOT available until human review.
"""
from pathlib import Path
import cv2,hashlib,json,datetime,shutil,collections,numpy as np
ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'data/lane_external_source_review_20261011'
if (PACK/'manifest.json').exists():raise FileExistsError('Refuse overwrite external TEST candidates')
(PACK/'frames').mkdir(parents=True,exist_ok=True)
source_meta=json.loads((ROOT/'data/raw/external/vietnam_commons_street_20261011/SOURCES.json').read_text(encoding='utf-8'))['sources']
rows=[];sources=[]
for folder in ('commons_hanoi_20261011','commons_saigon_govap_20261011'):
    prefix='hanoi' if 'hanoi' in folder else 'govap'
    old=ROOT/'output/review'/folder
    doc=json.loads((old/'review_manifest.json').read_text(encoding='utf-8'))
    source=next(x for x in source_meta if x['sha256']==doc['source_video_sha256'])
    sources.append({'group':prefix,'video_sha256':source['sha256'],
        'license_id':source['license_id'],'author':source['author'],
        'commons_page':source['commons_page']})
    for j,frame in enumerate(doc['frames']):
        original=old/frame['image']
        path=PACK/'frames'/(prefix+'_'+str(j).zfill(3)+'.jpg')
        if hashlib.sha256(original.read_bytes()).hexdigest()!=frame['image_sha256']:
            raise RuntimeError('Source JPEG SHA mismatch')
        shutil.copyfile(original,path)
        im=cv2.imread(str(path))
        if im is None:raise RuntimeError('Bad copied frame')
        h,w=im.shape[:2]
        tiny=cv2.resize(im,(96,54))
        gray=cv2.cvtColor(tiny,cv2.COLOR_BGR2GRAY)
        rows.append({'id':'ext_lane_'+prefix+'_'+str(j).zfill(3),
          'image':'frames/'+path.name,'sha256':frame['image_sha256'],
          'source_video_seconds':frame['timestamp_seconds'],
          'source_video_id':prefix+'_'+source['sha256'][:12],
          'temporal_group':prefix,'split':'TEST_EXTERNAL_SOURCE_CANDIDATE',
          'width':w,'height':h,
          'features':{'brightness':round(float(gray.mean()),1),
             'contrast':round(float(gray.std()),1),
             'focus_laplacian_var':round(float(cv2.Laplacian(gray,cv2.CV_32F).var()),1),
             'lower_dark_ratio':float(np.mean(gray[round(gray.shape[0]*.68):]<50))},
          'human_lanes':[],'human_no_visible_lane':False,
          'human_exhaustive':False,'human_confirmed':False,
          'reviewer':'','reviewed_at':None,'review_note':''})
combined=hashlib.sha256('|'.join(sorted(s['video_sha256'] for s in sources)).encode()).hexdigest()
manifest={'schema':'itasmm-lane-human-review-v1',
 'source_sha256':combined,'source_video':'2 distinct Wikimedia Commons videos',
 'video_sources_total':2,'source_provenance':sources,
 'source_duration_seconds':None,
 'data_generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'sampling_method':'existing source-disjoint Wikimedia video frame sampling',
 'target_frames':len(rows),'sampled_frames':len(rows),
 'split_counts':{'TEST_EXTERNAL_SOURCE_CANDIDATE':len(rows)},
 'human_verified_frames':0,'independent_road_test':False,
 'independent_original_video_sources':True,
 'split_note':'External route independent from user MP4, but NO manually checked lane masks yet.',
 'approved_for_training':False,'frames':rows}
(PACK/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('EXTERNAL_LANE_CANDIDATES',len(rows),
      [(x['group'],x['license_id']) for x in sources],flush=True)
