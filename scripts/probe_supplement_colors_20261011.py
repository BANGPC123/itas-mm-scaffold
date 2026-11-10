import cv2,json,numpy as np
from pathlib import Path
root=Path('D:/itas-mm/output/review/field_review_pack_20261011')
d=json.loads((root/'autobox_20261011/autobox_review_draft.json').read_text(encoding='utf-8'))
for n in [0,5,10,13,14,22]:
 f=d['frames'][n];img=cv2.imread(str(root/f['image']));h,w=img.shape[:2]
 print('FRAME',n)
 for s in f['signs']:
  if s.get('judgment')!='sign' or s.get('panel_role')=='supplementary':continue
  x1,y1,x2,y2=map(int,s['bbox_xyxy']);ww=x2-x1;hh=y2-y1
  if min(ww,hh)<18:continue
  if y2+round(.8*hh)>=h:continue
  patch=img[y2:y2+round(.85*hh),max(0,x1-3):min(w,x2+3)]
  hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV)
  mask=(hsv[:,:,1]<130)&(hsv[:,:,2]>100)
  rows=mask.mean(1)
  print(s['id'],s['label'],[round(x,2) for x in rows[::max(1,len(rows)//10)]])
