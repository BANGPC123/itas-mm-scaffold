"""Before/after screenshots from BOTH actually rendered MP4s on original user video."""
from pathlib import Path
import cv2,numpy as np,json
ROOT=Path(__file__).resolve().parents[1]
old=ROOT/'output/video_tests/user_test_video_yolopv2_hybrid_20261011/hybrid_lane_75s.mp4'
new=ROOT/'output/video_tests/user_test_video_lane_dedup_checked_20261011/hybrid_lane_75s.mp4'
out=ROOT/'output/evaluation/user_video_visual_before_after_20261011.jpg'
a=cv2.VideoCapture(str(old));b=cv2.VideoCapture(str(new))
assert a.isOpened() and b.isOpened()
pairs=[]
for sec in [5,21.5,69]:
    frames=[]
    for cap,title in [(a,'OLD: false lines / no QA'),(b,'NEW: screened road markings')]:
        cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        ok,img=cap.read()
        if not ok:raise RuntimeError(sec)
        img=cv2.resize(img,(640,360))
        cv2.rectangle(img,(0,0),(640,30),(14,16,20),-1)
        cv2.putText(img,f'{sec:.1f}s {title}',(9,23),
            cv2.FONT_HERSHEY_SIMPLEX,.60,(255,255,255),2)
        frames.append(img)
    pairs.append(np.hstack(frames))
a.release();b.release()
cv2.imwrite(str(out),np.vstack(pairs),[cv2.IMWRITE_JPEG_QUALITY,94])
print('BEFORE_AFTER_IMAGE_READY',out,flush=True)
