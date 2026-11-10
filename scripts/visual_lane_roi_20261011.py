from pathlib import Path
import cv2,numpy as np
from src.perception.lane_detector import LaneDetector
from src.utils.config import load_config

root=Path(__file__).resolve().parents[1]
c=cv2.VideoCapture(str(root/'data/demo/ufld_example.mp4'))
net=LaneDetector(load_config('perception'))
samples=[]
for i in [0,250,500]:
    c.set(cv2.CAP_PROP_POS_FRAMES,i)
    ok,img=c.read()
    if not ok:continue
    output=net.detect(img)
    overlay=img.copy()
    for p in output.polylines:
        cv2.polylines(overlay,[np.asarray(p,np.int32)],False,(50,190,0),3)
    cv2.rectangle(overlay,(0,0),(610,42),(0,0,0),-1)
    cv2.putText(overlay,f'Frame {i} | UFLD full frame | {len(output.polylines)} lanes',(8,28),
        cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2)
    samples.append(cv2.resize(overlay,(960,540)))
c.release()
path=root/'output/evaluation/lane_roi_full_visual_20261011.jpg'
cv2.imwrite(str(path),np.vstack(samples),[cv2.IMWRITE_JPEG_QUALITY,90])
print(path)
