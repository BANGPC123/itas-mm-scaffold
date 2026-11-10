"""Create scene previews for independently sourced, unverified road videos."""
from pathlib import Path
import cv2,json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
for folder in ('commons_hanoi_20261011','commons_saigon_govap_20261011'):
    pack=ROOT/'output/review'/folder
    if not(pack/'review_manifest.json').exists():
        print('PENDING',folder);continue
    data=json.loads((pack/'review_manifest.json').read_text(encoding='utf-8'))
    thumbs=[]
    for i,f in enumerate(data['frames']):
        im=cv2.imread(str(pack/f['image']))
        if im is None:raise RuntimeError('missing image')
        vis=im.copy()
        for s in f['signs']:
            x1,y1,x2,y2=map(int,s['bbox_xyxy'])
            cv2.rectangle(vis,(x1,y1),(x2,y2),(40,170,245),3)
        tile=cv2.resize(vis,(480,270))
        cv2.putText(tile,f"{f['id']} t={f['timestamp_seconds']:.1f}s boxes={len(f['signs'])}",
                    (10,24),cv2.FONT_HERSHEY_SIMPLEX,.7,(5,5,5),4)
        cv2.putText(tile,f"{f['id']} t={f['timestamp_seconds']:.1f}s boxes={len(f['signs'])}",
                    (10,24),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1)
        thumbs.append(tile)
    for j in range(0,len(thumbs),16):
        sub=thumbs[j:j+16]
        while len(sub)<16:sub.append(np.full((270,480,3),20,np.uint8))
        cv2.imwrite(str(pack/f'contact_{j//16+1:02d}.jpg'),
            np.vstack([np.hstack(sub[k:k+4]) for k in (0,4,8,12)]),
            [cv2.IMWRITE_JPEG_QUALITY,90])
    print('PREVIEW',folder,'frames',len(thumbs),
        'proposals',sum(len(f['signs']) for f in data['frames']))
