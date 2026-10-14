"""Evaluate 3 lane architectures against HUMAN-reviewed Vietnam video masks ONLY.

No AI proposals allowed as evaluation truth; TEST_HOLDOUT is protected by
--final-holdout and a write-once result gate. Frames from the same user video
are explicitly labelled as SAME_VIDEO holdout, not independent source TEST.
"""
from pathlib import Path
import argparse,json,cv2,torch,numpy as np,time,collections
from transformers import SegformerForSemanticSegmentation
from src.perception.lane_detector import LaneDetector
from src.perception.yolopv2_lane import YOLOPv2Lane
from src.utils.config import load_config
ROOT=Path(__file__).resolve().parents[1]
VER=ROOT/'data/processed/lane_human_verified_20261011/verified_manifest.jsonl'
class Stats:
    def __init__(self):self.tp=self.fp=self.fn=self.ms=[];self.neg_frames=0;self.neg_false=0
    def add(self,mask,gt,dt,negative):
        valid=gt!=255
        truth=(gt==1)&valid;detected=(mask>0)&valid
        self.tp+=int((truth&detected).sum())
        self.fp+=int((~truth&detected).sum())
        self.fn+=int((truth&~detected).sum())
        self.ms.append(dt)
        self.neg_frames+=int(negative)
        self.neg_false+=int(negative and detected.sum()>5)
    def report(self):
        tp,fp,fn=self.tp,self.fp,self.fn
        return {'pixel_precision':tp/max(1,tp+fp),
            'pixel_recall':tp/max(1,tp+fn),
            'pixel_f1':2*tp/max(1,2*tp+fp+fn),
            'paint_pixel_iou':tp/max(1,tp+fp+fn),
            'false_positive_frames_on_no_lane':self.neg_false,
            'no_lane_frames':self.neg_frames,
            'mean_inference_ms':round(float(np.mean(self.ms)),2),
            'p95_inference_ms':round(float(np.percentile(self.ms,95)),2)}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--final-holdout',action='store_true')
    args=parser.parse_args()
    if not VER.exists():
        raise SystemExit('BLOCKED: missing human-verified source masks. AI proposals may NOT be used as TEST GT.')
    data=[json.loads(s) for s in VER.read_text(encoding='utf-8').splitlines() if s.strip()]
    split='TEST_HOLDOUT_SAME_VIDEO' if args.final_holdout else 'VAL_INTERNAL'
    test=[r for r in data if r['split']==split and r['human_confirmed']]
    if len(test)<15:raise SystemExit('BLOCKED: Need at least 15 reviewer-verified '+split+' frames, have '+str(len(test)))
    output=ROOT/'output/evaluation'/('lane_human_samevideo_FINAL_HOLDOUT_20261011.json' if args.final_holdout else 'lane_human_internal_VAL_20261011.json')
    if output.exists():raise FileExistsError('No automatic overwrite or repeated holdout tuning')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    u=LaneDetector(load_config('perception'))
    y=YOLOPv2Lane(ROOT/'models/lane/candidates/yolopv2_official_20261011/yolopv2.pt',
                  device='cuda:0' if torch.cuda.is_available() else 'cpu',roi=.8)
    model_path=ROOT/'output/retraining_20261011/segformer_vietnam_human_b0/best'
    source='Vietnam_human_GT'
    if not model_path.is_dir():
        model_path=ROOT/'output/retraining_20261011/segformer_culane_b0/best'
        source='CULane_pretrain_only'
    if not model_path.is_dir():raise RuntimeError('SegFormer checkpoint missing')
    s=SegformerForSemanticSegmentation.from_pretrained(str(model_path)).to('cuda:0' if torch.cuda.is_available() else 'cpu').eval()
    scores={name:Stats() for name in ['UFLD','YOLOPv2_seg','SegFormer_B0']}
    scene=[]
    for r in test:
        img=cv2.imread(r['image']);gt=cv2.imread(r['mask'],cv2.IMREAD_UNCHANGED)
        assert img is not None and gt is not None
        H,W=img.shape[:2];negative=not np.any(gt==1)
        start=time.perf_counter();lanes=u.detect(img).polylines
        um=np.zeros((H,W),np.uint8)
        for line in lanes:
            if len(line)>1:cv2.polylines(um,[np.asarray(line,np.int32)],False,1,5)
        scores['UFLD'].add(um,gt,(time.perf_counter()-start)*1000,negative)
        start=time.perf_counter();ym=y.segment(img)
        scores['YOLOPv2_seg'].add(ym,gt,(time.perf_counter()-start)*1000,negative)
        start=time.perf_counter()
        x=cv2.resize(img,(512,288),interpolation=cv2.INTER_AREA)
        x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        x=(x-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        t=torch.from_numpy(x.transpose(2,0,1).copy())[None].to(next(s.parameters()).device)
        with torch.inference_mode():
            logits=s(pixel_values=t).logits
        pred=torch.nn.functional.interpolate(logits.float(),size=(H,W),mode='bilinear',align_corners=False).argmax(1)[0].cpu().numpy()
        scores['SegFormer_B0'].add((pred==1).astype('uint8'),gt,(time.perf_counter()-start)*1000,negative)
        scene.append({'frame':r['id'],'group':r['temporal_group'],'GT_negative':negative})
    output.write_text(json.dumps({'split':split,'source_routes':1,
        'source_disjoint_independent_test':False,'human_GT_count':len(test),
        'segformer_weight_source':source,'metrics':{n:st.report() for n,st in scores.items()},
        'reviewed_frames':scene,'training_and_holdout_same_video':True,
        'weight_promotion_automatic':False},
        ensure_ascii=False,indent=2),encoding='utf-8')
    print('HUMAN_GT_EVALUATION_READY',output,flush=True)
if __name__=='__main__':main()
