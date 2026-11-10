"""Nine source-grounded main+supplementary grouping previews, no OCR/legal inference."""
from pathlib import Path
import json,cv2,numpy as np
root=Path(__file__).resolve().parents[1]
pack=root/'output/review/field_review_pack_20261011'
data=json.loads((pack/'autobox_20261011/autobox_roles_review.json').read_text(encoding='utf-8'))
out=pack/'autobox_20261011/compound_pairs'
out.mkdir(exist_ok=True)
tiles=[]
report=[]
for f in data['frames']:
    img=cv2.imread(str(pack/f['image']))
    for s in f['signs']:
        parent_id=s.get('ai_panel_parent_candidate_id')
        if not parent_id:continue
        parent=next(x for x in f['signs'] if x['id']==parent_id)
        a=parent['bbox_xyxy'];b=s['bbox_xyxy']
        h,w=img.shape[:2]
        x1=max(0,int(min(a[0],b[0])-max(a[2]-a[0],b[2]-b[0])*.42))
        y1=max(0,int(min(a[1],b[1])-max(a[3]-a[1],b[3]-b[1])*.3))
        x2=min(w,int(max(a[2],b[2])+max(a[2]-a[0],b[2]-b[0])*.42))
        y2=min(h,int(max(a[3],b[3])+max(a[3]-a[1],b[3]-b[1])*.3))
        if x2-x1<16 or y2-y1<20:continue
        crop=img[y1:y2,x1:x2].copy()
        for box,color in [(a,(50,225,80)),(b,(255,80,235))]:
            xx1,yy1,xx2,yy2=map(int,[box[0]-x1,box[1]-y1,box[2]-x1,box[3]-y1])
            cv2.rectangle(crop,(xx1,yy1),(xx2,yy2),color,2)
        resized=cv2.resize(crop,(320,260))
        footer=np.full((52,320,3),24,np.uint8)
        text=f'{f["id"]}: main={parent["label"][:12]}'
        cv2.putText(footer,text,(6,21),cv2.FONT_HERSHEY_SIMPLEX,.52,(236,236,236),1)
        cv2.putText(footer,'white plate: text UNKNOWN / review',(6,41),cv2.FONT_HERSHEY_SIMPLEX,.42,(170,200,250),1)
        tiles.append(np.vstack([resized,footer]))
        key=f'{f["id"]}_{s["id"]}'.replace('/','_').replace('\\','_')
        cv2.imwrite(str(out/(key+'.jpg')),crop,[cv2.IMWRITE_JPEG_QUALITY,94])
        report.append({'frame':f['id'],'primary_id':parent_id,'primary_code':parent['label'],
                       'supplementary_id':s['id'],'supplementary_code':s['label'],
                       'text_verified':False})
for j in range(0,len(tiles),6):
    sub=tiles[j:j+6]
    while len(sub)<6:sub.append(np.full((312,320,3),16,np.uint8))
    composite=np.vstack([np.hstack(sub[:3]),np.hstack(sub[3:])])
    cv2.imwrite(str(out/f'sheet_{j//6+1:02d}.jpg'),composite,[cv2.IMWRITE_JPEG_QUALITY,92])
(out/'pairs.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'paired_plates':len(report),'sheets':(len(report)+5)//6,'directory':str(out)},ensure_ascii=False))
