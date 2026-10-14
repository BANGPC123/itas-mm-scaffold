"""Source-grounded assistant visual hard-negative candidates from NEW videos.

NOT human labels. These boxes are unreviewed suggestions; they are NOT imported
into a TRAIN background set. Commercial signboards are not traffic signs.
"""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]
PACKS={
'commons_hanoi_20261011':[
 ('frame_0000',(630,35,1045,216),'shop_sign','Pink commercial storefront banner, not road sign'),
],
'commons_saigon_govap_20261011':[
 ('frame_0010',(1130,155,1855,442),'commercial_billboard','Large BlueStone commercial appliances billboard'),
 ('frame_0013',(945,383,1345,598),'storefront_brand','VS food business facade branding, not road sign'),
 ('frame_0013',(1346,405,1495,613),'commercial_billboard','Product advertisement panel'),
 ('frame_0013',(1507,399,1774,615),'commercial_billboard','Printed delivery-phone advertising panel'),
],
}
results={}
for folder,items in PACKS.items():
    pack=ROOT/'output/review'/folder
    src=pack/'review_manifest.json'
    base=json.loads(src.read_text(encoding='utf-8'))
    index={f['id']:f for f in base['frames']}
    count=0
    for frame_id,box,kind,note in items:
        f=index[frame_id]
        w,h=f['width'],f['height']
        x1,y1,x2,y2=box
        if not(0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError(f'Box outside image: {frame_id}')
        row={'id':f'ai_background_candidate_{count}',
             'bbox_xyxy':[float(z) for z in box],
             'label':'unknown','ai_suggested_label':'background',
             'judgment':'unreviewed','panel_role':'unknown',
             'source':'AI_VISUAL_ADVERTISING_SUGGESTION_NOT_GT',
             'ai_visual_assessment':'not_sign',
             'ai_object_type':kind,'ai_visual_note':note,
             'human_verified':False}
        count+=1
        f['signs'].append(row)
    base['provenance']='AI visually proposed hard negatives, NOT human GT; source/license preserved'
    base['human_verified_ad_candidates']=0
    new_manifest=pack/'ai_hard_negative_candidates_20261011.json'
    if new_manifest.exists():raise FileExistsError('Refusing to overwrite prior AI candidates')
    new_manifest.write_text(json.dumps(base,ensure_ascii=False,indent=2),encoding='utf-8')
    gui=(pack/'index.html').read_text(encoding='utf-8')
    encoded=json.dumps(base,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    gui,n=re.subn(r'const manifest=.*?;\nconst original=',lambda m:'const manifest='+encoded+';\nconst original=',gui,count=1,flags=re.S)
    if n!=1:raise RuntimeError('Failed to insert review manifest')
    gui=gui.replace('</header>','<b class="warn">Khung quảng cáo AI gợi ý đang CHƯA XÁC NHẬN</b></header>',1)
    old='<small class="note"> Gợi ý AI: '
    assert old in gui
    gui=gui.replace(old,"<small class=\"note\"> '+(s.source==='AI_VISUAL_ADVERTISING_SUGGESTION_NOT_GT'?'AI NGHI QUẢNG CÁO / NOT_SIGN: '+esc(s.ai_visual_note||'')+' | ':'')+' Gợi ý AI: ",1)
    ui=pack/'index_AI_hard_negative_review.html'
    ui.write_text(gui,encoding='utf-8')
    results[folder]={'frames':len(base['frames']),'existing_model_box_proposals':
        sum(sum(s['source']=='yolo_candidate' for s in f['signs']) for f in base['frames']),
        'new_ai_background_proposals':count,
        'new_gt_labels':0,
        'independently_verified':False,
        'review_ui':str(ui),
        'source_video_sha256':base['source_video_sha256']}
dest=ROOT/'output/review/commons_urban_negative_intake_20261011.json'
dest.write_text(json.dumps({'sources':results,'status':'AI_PROPOSALS_ONLY_NO_GT',
                             'sign_or_no_sign_requires_human_review':True,
                             'warning':'All sampled frames from each video are one source group; no frame-level train/test split.'},
                            ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(results,ensure_ascii=False),flush=True)
