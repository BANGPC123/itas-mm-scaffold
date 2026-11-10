"""Non-destructive semi-automatic sign bounding boxes for ITAS-MM development video.

Source:
- selected YOLO11n over full+overlapping tiling passes;
- selected MobileNet recognizer supplies *suggested* code, never automatic GT;
- existing user review is the authoritative source for prior object judgments.
Everything new or changed is explicitly marked for HUMAN RECHECK.
The output is not legal/field validated and is not eligible for TRAIN without QA.
"""
from pathlib import Path
import copy, json, math, cv2, numpy as np, collections
from src.perception.sign_recognizer import SignRecognizer
from src.utils.config import load_config

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'output/review/field_review_pack_20261011'
INPUT=PACK/'itasmm_human_review_uploaded_20261011.json'
RAW=PACK/'ai_visual_review/multiscale_yolo_raw_20261011.json'
OUT=PACK/'autobox_20261011'
OUT.mkdir(exist_ok=True)
data=json.loads(INPUT.read_text(encoding='utf-8'))
raw={x['frame']:x['detections'] for x in json.loads(RAW.read_text(encoding='utf-8'))}
recognizer=SignRecognizer(load_config('perception'))


def area(b):return max(0.,b[2]-b[0])*max(0.,b[3]-b[1])

def iou(a,b):
    x=max(0.,min(a[2],b[2])-max(a[0],b[0]))
    y=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    i=x*y
    return i/(area(a)+area(b)-i) if i and area(a)+area(b)>i else 0.

def center_distance(a,b):
    ax=(a[0]+a[2])/2;ay=(a[1]+a[3])/2
    bx=(b[0]+b[2])/2;by=(b[1]+b[3])/2
    scale=max(10,(math.sqrt(area(a))+math.sqrt(area(b)))/2)
    return math.hypot(ax-bx,ay-by)/scale

def fusion(candidates,w,h):
    inp=sorted(candidates,key=lambda p:p['conf'],reverse=True)
    groups=[]
    for s in inp:
        box=s['box']
        if area(box)<90 or area(box)>.035*w*h:continue
        if box[2]>w+3 or box[3]>h+3:continue
        matched=None
        for g in groups:
            if iou(g['box'],box)>.32 and center_distance(g['box'],box)<.75:
                matched=g;break
        if matched is None:
            groups.append({'box':box[:], 'members':[s], 'conf':s['conf'], 'passes':{s['pass']}})
        else:
            matched['members'].append(s)
            matched['passes'].add(s['pass'])
            matched['conf']=max(matched['conf'],s['conf'])
            v=matched['members'];ws=np.array([p['conf']**2 for p in v])
            matched['box']=(np.average(np.array([p['box'] for p in v]),axis=0,weights=ws)).tolist()
    for g in groups:
        g['box']=[round(max(0,min(w if i%2==0 else h,v)),2) for i,v in enumerate(g['box'])]
        g['passes']=sorted(g['passes'])
        del g['members']
    return sorted(groups,key=lambda g:g['conf'],reverse=True)


def supplementary_candidates(im,main):
    """Conservative white-panel rectangle proposals BELOW a sign, no decoded text."""
    h,w=im.shape[:2]
    x1,y1,x2,y2=main
    ww=x2-x1;hh=y2-y1
    if ww<16 or hh<16 or hh>110 or ww>125:return []
    left=max(0,int(x1-.35*ww));right=min(w,int(x2+.35*ww))
    top=max(0,int(y2));bottom=min(h,int(y2+1.35*hh))
    if right-left<12 or bottom-top<10:return []
    roi=im[top:bottom,left:right]
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    # White / light gray opaque plates with dark text or border.
    light=cv2.inRange(hsv,(0,0,122),(179,94,255))
    light=cv2.morphologyEx(light,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    # Contour not bounding of entire brightly painted building: restrict size and alignment.
    cs,_=cv2.findContours(light,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ans=[]
    for c in cs:
        a,b,ow,oh=cv2.boundingRect(c)
        if ow<13 or oh<7 or not (.48*ww<=ow<=1.5*ww and .18*hh<=oh<=1.05*hh):continue
        if a+ow/2+left < x1-.17*ww or a+ow/2+left>x2+.17*ww:continue
        fill=cv2.contourArea(c)/(ow*oh)
        if fill<.40:continue
        box=[float(left+a),float(top+b),float(left+a+ow),float(top+b+oh)]
        if top+b-y2>.58*hh:continue
        ans.append({'box':box,'confidence':round(float(fill),2)})
    return sorted(ans,key=lambda z:z['confidence'],reverse=True)[:2]


def label_predictions(frame, proposals,img):
    from src.utils.types import SignDetection
    choices=[SignDetection('unknown',p['conf'],tuple(p['box'])) for p in proposals]
    choices=recognizer.recognize(img,choices,min_short_side=16)
    for p,s in zip(proposals,choices):
        p['suggested_code']=s.label
        p['suggested_code_confidence']=round(float(s.recognition_confidence or 0),3)


def target_match(box, other):
    return iou(box,other)>=.20 or (center_distance(box,other)<.63 and .32<area(box)/max(area(other),1)<3.1)


def annotate(im,frame,new):
    canvas=im.copy()
    # User boxes yellow, model snap green, auto proposals cyan, sub-panels purple.
    for s in frame['signs']:
        xy=tuple(map(int,map(round,s['bbox_xyxy'])))
        cv2.rectangle(canvas,xy[:2],xy[2:],(0,196,244),1)
    for s in new['signs']:
        xy=tuple(map(int,map(round,s['bbox_xyxy'])))
        ty=s.get('source','')
        if ty=='ai_autobox_20261011':color=(255,255,0)
        elif ty=='ai_autopanel_20261011':color=(255,50,255)
        elif s.get('ai_auto_bbox_changed'):color=(0,255,40)
        else:continue
        cv2.rectangle(canvas,xy[:2],xy[2:],color,3)
        label=s.get('ai_auto_suggested_code','unknown')[:12]
        cv2.putText(canvas,label,(max(0,xy[0]),max(14,xy[1]-3)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
    return canvas


results=[]
statistics=collections.Counter()
preview=[]
for frame in data['frames']:
    image=cv2.imread(str(PACK/frame['image']))
    assert image is not None
    h,w=image.shape[:2]
    candidates=fusion(raw[frame['id']],w,h)
    label_predictions(frame,candidates,image)
    statistics['merged_yolo_boxes']+=len(candidates)
    statistics['raw_yolo_boxes']+=len(raw[frame['id']])
    updated=copy.deepcopy(frame)
    oldsigns=updated['signs']
    used=set()
    for s in oldsigns:
        best=None
        box=s['bbox_xyxy']
        for i,p in enumerate(candidates):
            if i in used or not target_match(box,p['box']):continue
            score=iou(box,p['box'])+.18*p['conf']-.045*center_distance(box,p['box'])
            if best is None or score>best[0]:best=(score,i,p)
        if best is None:continue
        _,i,p=best; used.add(i)
        if s['judgment'] in ('not_sign','uncertain'):
            s['ai_auto_conflict_with_user']=bool(p['conf']>.68)
            if s['ai_auto_conflict_with_user']:statistics['review_conflicts']+=1
            continue
        # Keep the manual sign classification exactly; improve localization only.
        s['ai_auto_suggested_code']=p['suggested_code']
        s['ai_auto_recognition_confidence']=p['suggested_code_confidence']
        s['ai_auto_yolo_confidence']=p['conf']
        s['ai_auto_passes']=p['passes']
        if p['conf']>=.48 and (iou(box,p['box'])<.90 or s.get('ai_bbox_needs_correction')):
            s['ai_auto_original_bbox']=box[:]
            s['bbox_xyxy']=p['box']
            s['ai_auto_bbox_changed']=True
            s['ai_auto_requires_revalidation']=True
            statistics['snapped_existing_boxes']+=1
            if s.get('human_confirmed'):raise AssertionError('Sign objects do not store human_confirmed')
    # New detection candidates not matched to any existing user box, but
    # DO NOT automatically overwrite a user's prior "not_sign" judgments.
    added=[]
    for i,p in enumerate(candidates):
        if i in used:continue
        if p['conf']<.53 and not (len(p['passes'])>=2 and p['conf']>=.43):continue
        if any(iou(p['box'],s['bbox_xyxy'])>.07 or center_distance(p['box'],s['bbox_xyxy'])<.65
               for s in oldsigns):continue
        new={'id':'auto_sign_'+str(len(added)),'bbox_xyxy':p['box'],
             'label':p['suggested_code'] or 'unknown',
             'ai_suggested_label':p['suggested_code'] or 'unknown',
             'panel_role':'unknown','judgment':'unreviewed',
             'source':'ai_autobox_20261011',
             'ai_auto_yolo_confidence':p['conf'],
             'ai_auto_recognition_confidence':p['suggested_code_confidence'],
             'ai_auto_passes':p['passes'],
             'ai_auto_visual_not_confirmed':True}
        added.append(new)
        statistics['new_yolo_proposals']+=1

    # Supplementary white-board candidates. A user-labeled plate takes priority.
    plate_candidates=[]
    for s in oldsigns+added:
        if s['judgment']=='not_sign' or s['panel_role']=='supplementary':continue
        if s['source']=='ai_autobox_20261011' and s['ai_auto_yolo_confidence']<.7:continue
        for plate in supplementary_candidates(image,s['bbox_xyxy']):
            pb=plate['box']
            if any(iou(pb,z['bbox_xyxy'])>.10 or center_distance(pb,z['bbox_xyxy'])<.58
                   for z in oldsigns+added+plate_candidates):continue
            plate_candidates.append({'id':'auto_plate_'+str(len(plate_candidates)),
                'bbox_xyxy':pb,'label':'unknown','ai_suggested_label':'unknown',
                'panel_role':'supplementary','judgment':'unreviewed',
                'source':'ai_autopanel_20261011',
                'ai_auto_panel_potential_only':True,'ai_auto_requires_OCR':True})
            statistics['new_panel_proposals']+=1

    updated['signs']=oldsigns+added+plate_candidates
    mutated=bool(added or plate_candidates or any(s.get('ai_auto_bbox_changed') for s in oldsigns))
    if mutated and frame.get('human_confirmed'):
        updated['ai_auto_previous_human_confirmation']={'reviewer':frame.get('reviewer'),'reviewed_at':frame.get('reviewed_at')}
        updated['human_confirmed']=False
        updated['reviewed_at']=None
        updated['all_visible_signs_annotated']=False
        statistics['invalidated_previous_confirmation']+=1
    # In the candidate file, all previous manually assessed signs retain choices.
    updated['ai_auto_bbox_provenance']='multi_pass_yolo11n_candidate_unverified'
    result={'frame':frame['id'],'original':len(frame['signs']),
            'new_yolo':len(added),'new_supplementary':len(plate_candidates),
            'snapped':sum(bool(s.get('ai_auto_bbox_changed')) for s in oldsigns)}
    results.append(updated)
    statistics['frames']+=1
    if mutated:statistics['frames_with_proposals']+=1
    overlay=annotate(image,frame,updated)
    cv2.putText(overlay,f"{frame['id']} | YOLO new={len(added)} | panels={len(plate_candidates)} | snapped={result['snapped']}",
                (10,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
    cv2.imwrite(str(OUT/(frame['id']+'.jpg')),overlay,[cv2.IMWRITE_JPEG_QUALITY,90])
    preview.append(cv2.resize(overlay,(640,360)))
    print(json.dumps(result),flush=True)

for j in range(0,len(preview),4):
    contact=np.vstack([np.hstack(preview[j:j+2]),np.hstack(preview[j+2:j+4])])
    cv2.imwrite(str(OUT/f'contact_{j//4+1:02d}.jpg'),contact,[cv2.IMWRITE_JPEG_QUALITY,92])
document={'schema_version':'itasmm-field-review-v1',
          'provenance':'AI AUTOMATIC BOX CANDIDATES FROM YOLO11n+MobileNet, NOT HUMAN GT',
          'frames':results}
dest=OUT/'autobox_review_draft.json'
dest.write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding='utf-8')
statistics['old_boxes']=sum(len(f['signs']) for f in data['frames'])
statistics['new_total_boxes']=sum(len(f['signs']) for f in results)
statistics['human_confirmed_frames_in_output']=sum(f['human_confirmed'] for f in results)
statistics['eligible_independent_test']=False
(OUT/'summary.json').write_text(json.dumps(dict(statistics),ensure_ascii=False,indent=2),encoding='utf-8')
print('SUMMARY '+json.dumps(dict(statistics),ensure_ascii=False),flush=True)
