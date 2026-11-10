"""Pixel-for-pixel same 50 labeled CULane VAL images; 3 architectures.

Metrics are GT raster masks at original image size. Not legal lane scope.
UFLDv2 and YOLOPv2 weights frozen. SegFormer is 2-epoch CULane candidate.
No human Vietnam ground truth used in this comparison.
"""
from pathlib import Path
import cv2,numpy as np,torch,json,time,collections
from transformers import SegformerForSemanticSegmentation
from src.perception.ufld_v2 import UFLDv2
from src.perception.yolopv2_lane import YOLOPv2Lane
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/evaluation/lane_three_way_culane_20261011'
if (OUT/'report.json').exists():raise FileExistsError('No overwrite')
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);cv2.setNumThreads(1)
device='cuda:0' if torch.cuda.is_available() else 'cpu'
ufld=UFLDv2(ROOT/'models/lane/selected_20261009.pth',device,half=True)
yolopv2=YOLOPv2Lane(ROOT/'models/lane/candidates/yolopv2_official_20261011/yolopv2.pt',
                      device=device,half=True,roi=1.,threshold=.56)
seg=SegformerForSemanticSegmentation.from_pretrained(str(ROOT/'output/retraining_20261011/segformer_culane_b0/best')).to(device).eval()
name=[line.split()[0].lstrip('/') for line in (ROOT/'data/raw/culane/test/list/val.txt').read_text().splitlines() if line.strip()][:50]
base=ROOT/'data/raw/culane/val'
summary={kind:{'tp':0,'fp':0,'fn':0,'latency_ms':[]} for kind in ['UFLD-v2','YOLOPv2 segmentation','SegFormer-B0']}
scenes=[]
for i,file in enumerate(name):
    image=cv2.imread(str(base/file))
    if image is None:continue
    h,w=image.shape[:2]
    g=np.zeros((h,w),np.uint8)
    lines=(base/Path(file).with_suffix('.lines.txt')).read_text().splitlines()
    for line in lines:
        numbers=np.fromstring(line,sep=' ')
        if len(numbers)>=4:
            cv2.polylines(g,[numbers.reshape(-1,2).round().astype(np.int32)],False,1,16)
    start=time.perf_counter()
    ufld_output=ufld.detect(image,filter_geometry=False).polylines
    u=np.zeros_like(g)
    for ln in ufld_output:
        if len(ln)>1:
            cv2.polylines(u,[np.asarray(ln,np.int32)],False,1,16)
    ut=(time.perf_counter()-start)*1000
    start=time.perf_counter()
    v=yolopv2.segment(image)
    vt=(time.perf_counter()-start)*1000
    start=time.perf_counter()
    img=cv2.resize(image,(512,288),interpolation=cv2.INTER_AREA)
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    rgb=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    x=torch.from_numpy(rgb.transpose(2,0,1).copy()).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast('cuda',enabled=device=='cuda:0'):
        logits=seg(pixel_values=x).logits
    pred=torch.nn.functional.interpolate(logits.float(),size=(h,w),
                    mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy().astype('uint8')
    s=(pred==1).astype('uint8')
    st=(time.perf_counter()-start)*1000
    for kind,mask,latency in [('UFLD-v2',u,ut),('YOLOPv2 segmentation',v,vt),('SegFormer-B0',s,st)]:
        row=summary[kind]
        positive=mask>0;gt=g>0
        row['tp']+=int(np.logical_and(positive,gt).sum())
        row['fp']+=int(np.logical_and(positive,~gt).sum())
        row['fn']+=int(np.logical_and(~positive,gt).sum())
        row['latency_ms'].append(latency)
    if i in (0,5,13,25,40):
        vis=[]
        for name_mask,m in [('GT',g),('UFLD-v2',u),('YOLOPv2',v),('SegFormer',s)]:
            overlay=image.copy()
            overlay[m>0]=(0,245,230)
            tiny=cv2.resize(overlay,(520,190))
            cv2.rectangle(tiny,(0,0),(520,27),(15,20,30),-1)
            cv2.putText(tiny,name_mask,(6,20),cv2.FONT_HERSHEY_SIMPLEX,.63,(255,255,255),2)
            vis.append(tiny)
        cv2.imwrite(str(OUT/f'comparison_{i:03d}.jpg'),np.hstack(vis))
    if (i+1)%10==0:print('BENCH_FRAME',i+1,'/',len(name),flush=True)
report={'dataset':'CULane VAL 50 original ground-truth frames',
    'not_independent_Vietnam_gt':True,'segformer_training':'640 CULane train images, 2 epochs',
    'prediction_types':{'UFLD-v2':'polyline to 16-pixel raster',
       'YOLOPv2 segmentation':'BDD100K lane segmentation probability mask thresholded',
       'SegFormer-B0':'argmax pixel class 1 from supervised CULane candidate'},
    'models':{},'source_frames_compared':len(name),
    'same_pixel_GT_for_all_models':True,
    'selected_weights_changed':False}
for kind,row in summary.items():
    tp,fp,fn=[row[k] for k in ('tp','fp','fn')]
    report['models'][kind]={'pixel_precision':tp/max(1,tp+fp),
        'pixel_recall':tp/max(1,tp+fn),
        'pixel_f1':2*tp/max(1,2*tp+fp+fn),
        'pixel_iou':tp/max(1,tp+fp+fn),
        'mean_inference_ms':float(np.mean(row['latency_ms'])),
        'p95_inference_ms':float(np.percentile(row['latency_ms'],95)),
        'tp':tp,'fp':fp,'fn':fn}
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('THREE_WAY_VAL_DONE',json.dumps(report,ensure_ascii=False),flush=True)
