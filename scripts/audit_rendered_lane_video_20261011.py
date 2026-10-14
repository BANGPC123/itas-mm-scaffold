"""Audit the actual rendered video, not counts of predicted lane polylines."""
from pathlib import Path
import cv2,numpy as np,json
root=Path(r'D:\itas-mm')
base=root/'output/video_tests/user_test_video_yolopv2_hybrid_20261011'
video=base/'hybrid_lane_75s.mp4'
assert video.is_file()
events=[json.loads(s) for s in (base/'model_evidence.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
v=cv2.VideoCapture(str(video))
assert v.isOpened()
fps=v.get(cv2.CAP_PROP_FPS);frames=v.get(cv2.CAP_PROP_FRAME_COUNT)
print('OUTPUT_VIDEO',fps,frames,flush=True)
output=base/'visual_audit_samples'
output.mkdir(exist_ok=True)
samples=[]
for t in np.arange(1.5,74,2.5):
    v.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000)
    ok,img=v.read()
    if not ok:continue
    nearest=min(events,key=lambda e:abs(e['t']-t))
    img=cv2.resize(img,(560,315))
    samples.append(img)
    print('SAMPLE',round(float(t),1),nearest['backend'],nearest['lanes'],flush=True)
v.release()
for batch in range(0,len(samples),12):
    current=samples[batch:batch+12]
    tile=np.zeros((4*315,3*560,3),dtype='uint8')
    for n,img in enumerate(current):
        a,b=divmod(n,3)
        tile[a*315:(a+1)*315,b*560:(b+1)*560]=img
    name=output/f'contact_{batch//12+1:02d}.jpg'
    cv2.imwrite(str(name),tile,[cv2.IMWRITE_JPEG_QUALITY,93])
    print('SAVED',name,flush=True)
