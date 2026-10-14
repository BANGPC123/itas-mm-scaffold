"""Visual sheets for careful assistant-only sign/scene review."""
from pathlib import Path
import cv2,numpy as np,json
root=Path(r'D:\itas-mm')
pack=root/'output/review/field_review_pack_20261011'
data=json.loads((pack/'review_manifest.json').read_text(encoding='utf-8'))
out=pack/'ai_visual_review'
out.mkdir(exist_ok=True)
summary=[]
thumbs=[]
allpatch=[]
for ix,f in enumerate(data['frames']):
    im=cv2.imread(str(pack/f['image']))
    full=im.copy()
    captions=[]
    for j,s in enumerate(f['signs']):
        x1,y1,x2,y2=[int(round(x)) for x in s['bbox_xyxy']]
        typ=s.get('source','ai')
        color=(50,120,255) if typ=='ai_removed_candidate' else (50,200,255)
        cv2.rectangle(full,(x1,y1),(x2,y2),color,2)
        lab=f"{j}:{s['label'][:10]}"+('!' if typ=='ai_removed_candidate' else '')
        py=max(18,y1-3)
        cv2.rectangle(full,(x1,py-17),(min(im.shape[1]-1,x1+max(55,11*len(lab))),py+2),(10,10,10),-1)
        cv2.putText(full,lab,(x1,py-3),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
        captions.append({'j':j,'id':s['id'],'label':s['label'],'bbox':s['bbox_xyxy'],'source':typ})
        cx=(x1+x2)/2;cy=(y1+y2)/2
        side=max(80,int(max(x2-x1,y2-y1)*3.0))
        xx1=max(0,int(cx-side/2));yy1=max(0,int(cy-side/2))
        xx2=min(im.shape[1],int(cx+side/2));yy2=min(im.shape[0],int(cy+side/2))
        patch=im[yy1:yy2,xx1:xx2].copy()
        if patch.size==0:continue
        cv2.rectangle(patch,(x1-xx1,y1-yy1),(x2-xx1,y2-yy1),color,2)
        patch=cv2.resize(patch,(250,230))
        foot=np.full((45,250,3),24,np.uint8)
        cv2.putText(foot,f"{ix:02d} box {j}: {s['label'][:13]}",(5,18),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1)
        cv2.putText(foot,typ.replace('_candidate','')[:22],(5,36),cv2.FONT_HERSHEY_SIMPLEX,.39,(170,190,230),1)
        allpatch.append(np.vstack([patch,foot]))
    top=np.full((36,1280,3),16,np.uint8)
    cv2.putText(top,f"FRAME {ix:02d}  t={f['timestamp_seconds']:.1f}s  boxes={len(f['signs'])}",(12,25),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
    cv2.imwrite(str(out/f'frame_{ix:02d}.jpg'),np.vstack([top,full]),[cv2.IMWRITE_JPEG_QUALITY,93])
    summary.append({'frame':ix,'time':f['timestamp_seconds'],'boxes':captions})
    tile=cv2.resize(np.vstack([top,full]),(800,472))
    thumbs.append(tile)
for i in range(0,len(thumbs),4):
    sheet=np.vstack([np.hstack(thumbs[i:i+2]),np.hstack(thumbs[i+2:i+4])])
    cv2.imwrite(str(out/f'scene_sheet_{i//4+1:02d}.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,93])
for i in range(0,len(allpatch),12):
    set12=allpatch[i:i+12]
    while len(set12)<12:set12.append(np.full((275,250,3),20,np.uint8))
    grid=np.vstack([np.hstack(set12[j:j+4]) for j in (0,4,8)])
    cv2.imwrite(str(out/f'boxes_sheet_{i//12+1:02d}.jpg'),grid,[cv2.IMWRITE_JPEG_QUALITY,94])
(out/'reference.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'scenes':len(thumbs),'boxes':len(allpatch),'scene_sheets':(len(thumbs)+3)//4,'box_sheets':(len(allpatch)+11)//12,'dir':str(out)},ensure_ascii=False))
