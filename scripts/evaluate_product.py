from pathlib import Path
import sys,json,time,base64,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from src.utils.config import load_config
from src.perception.sign_detector import SignDetector
from src.perception.sign_recognizer import SignRecognizer
from src.pipeline.orchestrator import Pipeline

ROOT=Path(__file__).resolve().parents[1]
LOC=ROOT/'data/processed/canonical/itas_mm_v1/localization'
SEM=ROOT/'data/audit/semantic/final_v1_1'

def write_manifest():
    paths=[ROOT/'models/sign_detector/best.pt',ROOT/'models/sign_recognizer/best.pt',ROOT/'models/lane/culane_res18.pth']
    manifest={}
    for path in paths:
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
        manifest[str(path.relative_to(ROOT))]={'bytes':path.stat().st_size,'sha256':digest.hexdigest()}
    (ROOT/'models/model_manifest.json').write_text(json.dumps(manifest,indent=2))

def iou_matrix(a,b):
    if not len(a) or not len(b):return np.zeros((len(a),len(b)))
    a=np.array(a);b=np.array(b)
    tl=np.maximum(a[:,None,:2],b[None,:,:2]);br=np.minimum(a[:,None,2:],b[None,:,2:])
    wh=np.maximum(0,br-tl);inter=wh[:,:,0]*wh[:,:,1]
    aa=np.prod(a[:,2:]-a[:,:2],axis=1);bb=np.prod(b[:,2:]-b[:,:2],axis=1)
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-9)

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'output/evaluation/product')
    parser.add_argument('--skip-benchmark',action='store_true')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(4)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=load_config('perception');cfg['sign_detector']['device']='0' if torch.cuda.is_available() else 'cpu'
    detector=SignDetector(cfg);recognizer=SignRecognizer(cfg)
    if detector._model is None or recognizer.model is None:raise RuntimeError('Finish model training first')
    samples=pd.read_csv(LOC/'samples.csv');samples=samples[samples.split=='TEST'].sort_values('sample_id')
    boxes=pd.read_csv(SEM/'semantic_labels_v1_1.csv',keep_default_na=False)
    boxes=boxes[boxes.split=='TEST'];groups={key:g for key,g in boxes.groupby('sample_id')}
    totals={'released_gt':0,'matched_released_gt':0,'correct_codes':0,'released_ge32':0,'matched_ge32':0,'correct_ge32':0,'predictions':0,'matched_all_gt':0}
    rows=[];start=time.perf_counter()
    for number,sample in enumerate(samples.itertuples(),1):
        image=cv2.imread(str(LOC/sample.image_path))
        signs=recognizer.recognize(image,detector.detect(image))
        gt=groups[sample.sample_id].reset_index(drop=True)
        released=gt.semantic_final_disposition.str.startswith('RELEASED')
        totals['released_gt']+=int(released.sum());totals['released_ge32']+=int((released&(gt.short_side_px>=32)).sum())
        totals['predictions']+=len(signs)
        mat=iou_matrix([s.bbox_xyxy for s in signs],[json.loads(v) for v in gt.bbox_xyxy])
        # Maximize valid-pair cardinality before total IoU, mirroring class-agnostic matching.
        cost=np.where(mat>=.5,1+mat/1000,0)
        pi,gi=linear_sum_assignment(-cost)
        for p,g in zip(pi,gi):
            if mat[p,g]<.5:continue
            totals['matched_all_gt']+=1
            row=gt.iloc[g]
            if not released.iloc[g]:continue
            correct=signs[p].label==row.canonical_sign_code
            totals['matched_released_gt']+=1;totals['correct_codes']+=int(correct)
            if row.short_side_px>=32:
                totals['matched_ge32']+=1;totals['correct_ge32']+=int(correct)
            rows.append({'box_id':row.canonical_box_id,'truth':row.canonical_sign_code,'prediction':signs[p].label,'iou':float(mat[p,g]),'short_side_px':row.short_side_px})
        if number%100==0:print('Evaluated TEST images',number,'/',len(samples),flush=True)
    totals['elapsed_seconds']=time.perf_counter()-start
    totals['correct_code_recall_all_released']=totals['correct_codes']/totals['released_gt']
    totals['correct_code_recall_ge32']=totals['correct_ge32']/totals['released_ge32']
    totals['matched_released_code_accuracy']=totals['correct_codes']/max(1,totals['matched_released_gt'])
    totals['operating_precision_iou50']=totals['matched_all_gt']/max(1,totals['predictions'])
    totals['operating_recall_iou50']=totals['matched_all_gt']/len(boxes)
    totals['operating_confidence']=cfg['sign_detector']['confidence_threshold']
    totals['notes']='TEST released ground truth only; masked semantic targets excluded. IoU >=0.5, class-agnostic one-to-one matching. Pipeline crop-size and confidence rejection enabled. This does not evaluate lane applicability or legal guidance.'
    pd.DataFrame(rows).to_csv(out/'matched_sign_predictions.csv',index=False)
    (out/'sign_pipeline_results.json').write_text(json.dumps(totals,indent=2))
    if args.skip_benchmark:
        print(json.dumps(totals,indent=2),flush=True)
        return
    del detector,recognizer
    if torch.cuda.is_available():torch.cuda.empty_cache()
    pipeline=Pipeline();timings=[]
    for i,sample in enumerate(samples.head(11).itertuples()):
        image=cv2.imread(str(LOC/sample.image_path))
        result=pipeline.run(image,10.7769,106.7009,synthesize_audio=False)
        if i:timings.append(result.timings_ms)
        if i in [1,2,3]:
            (out/f'example_{i}.jpg').write_bytes(base64.b64decode(result.annotated_image.split(',',1)[1]))
            from dataclasses import asdict
            record=asdict(result);record.pop('annotated_image',None)
            (out/f'example_{i}.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    benchmark={'frames':10,'warmup_frames':1,'tts_enabled':False,'input_split':'first 11 sorted TEST images; descriptive benchmark only',
               'devices':{'detector':str(pipeline.sign_detector.device),'lane':pipeline.lane_detector.learned.device,'recognizer':pipeline.sign_recognizer.device},
               'mean_ms':pd.DataFrame(timings).mean().to_dict(),'median_ms':pd.DataFrame(timings).median().to_dict(),
               'p95_ms':pd.DataFrame(timings).quantile(.95).to_dict(),
               'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
               'torch':torch.__version__,'lane_accuracy':'NOT_MEASURED_NO_LOCAL_GROUND_TRUTH'}
    (out/'benchmark.json').write_text(json.dumps(benchmark,indent=2))
    video=ROOT/'data/demo/ufld_example.mp4'
    if video.is_file():
        cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,30);ok,frame=cap.read();cap.release()
        if ok:
            demo=pipeline.run(frame,10.7769,106.7009,synthesize_audio=False)
            (out/'lane_demo.jpg').write_bytes(base64.b64decode(demo.annotated_image.split(',',1)[1]))
    write_manifest()
    print(json.dumps(totals,indent=2));print(json.dumps(benchmark,indent=2))

if __name__=='__main__':main()
