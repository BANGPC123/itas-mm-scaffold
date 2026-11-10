"""Diagnostic comparison against AI-reviewed development annotations, never human GT."""
import argparse,copy,gc,json,sys
from pathlib import Path
import cv2,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.perception.sign_recognizer import SignRecognizer
from src.perception.sign_detector import SignDetector
from src.utils.config import load_config
from src.utils.types import SignDetection

def iou(a,b):
    x=max(a[0],b[0]);y=max(a[1],b[1]);u=min(a[2],b[2]);v=min(a[3],b[3]);area=max(0,u-x)*max(0,v-y)
    return area/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area+1e-9)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='auto');args=parser.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    data=ROOT/'output/review/test_video_ai_review';packet=json.loads((data/'ai_reviewed.json').read_text(encoding='utf-8'))
    changes=json.loads((data/'changes.json').read_text(encoding='utf-8'))
    output={}
    for version,detweight,clsweight in [('baseline','models/sign_detector/best.pt','models/sign_recognizer/robust_val.pt'),
            ('candidate','output/retraining_20261009/detector/selected.pt','output/retraining_20261009/recognizer/candidate.pt')]:
        cfg=copy.deepcopy(load_config('perception'));cfg['sign_detector']['weights_path']=str(ROOT/detweight);cfg['sign_recognizer']['weights_path']=str(ROOT/clsweight)
        cfg['sign_detector']['device']=args.device;cfg['sign_recognizer']['device']=args.device
        det=SignDetector(cfg);rec=SignRecognizer(cfg);known=[];background=[];detections=[]
        for frame in packet['samples']:
            im=cv2.imread(str(data/frame['images'][1]));road=im[:468]
            boxes=[SignDetection(s['label'],1.,tuple(s['bbox_xyxy'])) for s in frame['signs'] if s['label']!='unknown']
            targets=[s.label for s in boxes];results=rec.recognize(im,boxes,min_short_side=16)
            known.extend(dict(frame=frame['id'],expected=y,predicted=s.label,confidence=s.recognition_confidence) for y,s in zip(targets,results))
            negatives=[c['before'] for c in changes if c['frame']==frame['id'] and c['action']=='removed' and 'trùng' not in c['reason']]
            boxes=[SignDetection('unknown',1.,tuple(s['bbox_xyxy'])) for s in negatives]
            results=rec.recognize(im,boxes,min_short_side=16)
            background.extend(dict(frame=frame['id'],predicted=s.label,confidence=s.recognition_confidence) for s in results)
            found=rec.recognize(road,det.detect(road),min_short_side=16)
            detections.append(dict(frame=frame['id'],boxes=len(found),recognized=sum(s.label!='unknown' for s in found),
                false_positive_boxes_overlapping_ai_removed=sum(any(iou(s.bbox_xyxy,n['bbox_xyxy'])>=.3 for n in negatives) for s in found),
                known_codes_on_ai_removed=sum(s.label!='unknown' and any(iou(s.bbox_xyxy,n['bbox_xyxy'])>=.3 for n in negatives) for s in found)))
        output[version]=dict(known_crop_agreement=sum(r['expected']==r['predicted'] for r in known),known_crop_n=len(known),
            negative_crop_false_accepts=sum(r['predicted']!='unknown' for r in background),negative_crop_n=len(background),
            known=known,background=background,frames=detections)
        del det,rec;gc.collect();torch.cuda.empty_cache()
    output['scope']='24 AI-reviewed development frames; diagnostic agreement, not human-verified accuracy. No video used for training.'
    output['device']=args.device
    path=ROOT/'output/retraining_20261009/video_review_comparison.json';path.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a not in ['known','background','frames']} for k,v in output.items() if isinstance(v,dict)},indent=2))

if __name__=='__main__':main()
