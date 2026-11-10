"""Visual evidence crops for independently identifiable false positives."""
from pathlib import Path
import cv2,numpy as np,json
ROOT=Path(r'D:\itas-mm')
video=ROOT/'output/video_tests/test_video_20261010_after_hardening/test_video_annotated.mp4'
out=ROOT/'output/video_tests/test_video_20261010_after_hardening/visual_review_20261011'
data=ROOT/'output/video_tests/test_video_20261010_after_hardening/predictions.jsonl'
targets=[(55.333,'P.127','Dashboard gauge'),(337.333,'P.127','Dashboard gauge again'),
         (209.0,'I.409','Commercial logo'),(286.667,'W.210','Car rear lamp')]
rows=[json.loads(x) for x in data.open(encoding='utf-8')]
cap=cv2.VideoCapture(str(video))
cards=[]
for t,code,comment in targets:
    pred=rows[round(t*3)]
    found=[s for s in pred['signs'] if s['label']==code]
    if not found:print('not found',t,code);continue
    b=found[0]['bbox_xyxy']
    cap.set(cv2.CAP_PROP_POS_FRAMES,round(pred['timestamp_seconds']*3))
    ok,img=cap.read()
    if not ok:continue
    sy,sx=img.shape[0]/720,img.shape[1]/1280
    x1,y1,x2,y2=(b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy)
    cx=(x1+x2)/2;cy=(y1+y2)/2
    cw=max(410,(x2-x1)*7);ch=max(270,(y2-y1)*7)
    left=max(0,int(cx-cw/2));right=min(img.shape[1],int(cx+cw/2))
    top=max(0,int(cy-ch/2));bottom=min(img.shape[0],int(cy+ch/2))
    crop=img[top:bottom,left:right].copy()
    if crop.size==0:continue
    cv2.rectangle(crop,(int(x1-left),int(y1-top)),(int(x2-left),int(y2-top)),(0,210,255),3)
    crop=cv2.resize(crop,(780,480))
    cv2.rectangle(crop,(0,0),(780,40),(0,0,0),-1)
    cv2.putText(crop,f'{t:.1f}s predicted {code} | {comment}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.76,(255,255,255),2,cv2.LINE_AA)
    cards.append(crop)
cap.release()
if len(cards)==4:
    final=np.vstack([np.hstack(cards[:2]),np.hstack(cards[2:])])
    cv2.imwrite(str(out/'false_positive_closeups.jpg'),final,[cv2.IMWRITE_JPEG_QUALITY,92])
print(json.dumps(dict(cropped=len(cards),file=str(out/'false_positive_closeups.jpg'))))
