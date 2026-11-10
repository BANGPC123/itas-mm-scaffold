"""CPU-only RapidOCR feasibility on 8 development supplementary sign regions.

Execute in the isolated .ocr-venv. Not integrated in runtime or legal TTS.
No claim of accuracy on independent Vietnam field GT.
"""
from pathlib import Path
import sys,json,time,cv2,numpy as np,statistics,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.perception.plate_text_parser import stable_ocr_candidate
from rapidocr import RapidOCR
import rapidocr
PACK=ROOT/'output/review/field_review_pack_20261011'
OUT=PACK/'ocr_experiments_20261011'
OUT.mkdir(exist_ok=True)
data=json.loads((PACK/'autobox_20261011/autobox_roles_review.json').read_text(encoding='utf-8'))
pairs=json.loads((PACK/'autobox_20261011/compound_pairs/pairs.json').read_text(encoding='utf-8'))
known={
('frame_013','added_time'):'06:00-22:00',
('frame_021','manual_1791687609125_1'):'06:00-22:00',
('frame_022','manual_1791687638009_1'):'06:00-22:00',
('frame_022','manual_1791687659050_3'):'06:00-08:00;16:00-18:00'
}
engine=RapidOCR()
items=[]
for pair in pairs:
    frame=next(f for f in data['frames'] if f['id']==pair['frame'])
    s=next(s for s in frame['signs'] if s['id']==pair['supplementary_id'])
    img=cv2.imread(str(PACK/frame['image']))
    h,w=img.shape[:2]
    x1,y1,x2,y2=s['bbox_xyxy']
    pad=max(2,int(.075*max(x2-x1,y2-y1)))
    a=max(0,int(x1-pad));b=max(0,int(y1-pad))
    c=min(w,int(x2+pad));d=min(h,int(y2+pad))
    crop=img[b:d,a:c]
    if crop.size==0:raise RuntimeError('Empty crop')
    upscaled=cv2.resize(crop,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
    lab=cv2.cvtColor(upscaled,cv2.COLOR_BGR2LAB)
    L,A,B=cv2.split(lab)
    L=cv2.createCLAHE(clipLimit=2,tileGridSize=(5,5)).apply(L)
    enhanced=cv2.cvtColor(cv2.merge((L,A,B)),cv2.COLOR_LAB2BGR)
    pairfile=PACK/'autobox_20261011/compound_pairs'/f"{frame['id']}_{s['id']}.jpg"
    views={'crop_3x':upscaled,'contrast_3x':enhanced}
    if pairfile.is_file():views['main_plus_plate']=cv2.imread(str(pairfile))
    observations=[]
    for name,view in views.items():
        start=time.perf_counter()
        result=engine(view)
        txts=list(getattr(result,'txts',[]) or [])
        scores=[float(v) for v in (getattr(result,'scores',[]) or [])]
        candidate=stable_ocr_candidate(txts,scores)
        observations.append({'view':name,'text':txts,
                             'scores':[round(s,3) for s in scores],
                             'time_ranges':candidate.windows,
                             'syntax_and_ocr_confident':candidate.complete,
                             'time_ms':round((time.perf_counter()-start)*1000,2)})
    plausible=[o for o in observations if o['syntax_and_ocr_confident']]
    votes={}
    for item in plausible:
        key=tuple(item['time_ranges'])
        votes[key]=votes.get(key,0)+1
    winning=max(votes,key=lambda k:votes[k]) if votes else ()
    agreement=bool(winning and votes[winning]>=2)
    row={'frame':frame['id'],'supplementary_id':s['id'],
         'parent_id':pair['primary_id'],
         'assistant_visual_hint_NOT_GT':known.get((frame['id'],s['id'])),
         'image_sha256':hashlib.sha256(crop.tobytes()).hexdigest(),
         'plate_crop_w':crop.shape[1], 'plate_crop_h':crop.shape[0],
         'variants':observations,
         'agreement_2_of_3':agreement,
         'ocr_agreed_time_candidate':list(winning) if agreement else [],
         'verified_by_human':False,'legal_effect_established':False}
    items.append(row)
    cv2.imwrite(str(OUT/f"{frame['id']}_{s['id']}.png"),crop)
    print('OCR '+json.dumps({'frame':row['frame'],'sub':row['supplementary_id'],
        'hint':row['assistant_visual_hint_NOT_GT'],'views':[(x['view'],x['text'],x['scores']) for x in observations],
        'agreed':row['ocr_agreed_time_candidate']},ensure_ascii=False),flush=True)
report={'purpose':'OCR research only, never direct TTS/legal scope',
        'source':'One previously used Vietnam development video',
        'independent_human_GT':False,'ocr_package_version':getattr(rapidocr,'__version__','unknown'),
        'ocr_backend':'ONNXRuntime_CPU',
        'pair_count':len(items),
        'paired_stable_candidate_count':sum(x['agreement_2_of_3'] for x in items),
        'assistant_visual_hint_count':sum(x['assistant_visual_hint_NOT_GT'] is not None for x in items),
        'rows':items,'deployed':False,
        'limitations':'Small panels, cropped blur, no human verified texts. Two preprocessing views are correlated, not independent. No legal temporal condition inference.'}
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('RESULT '+json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False),flush=True)
