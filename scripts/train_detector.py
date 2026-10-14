from pathlib import Path
import argparse, json, shutil, time
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,help='Completed baseline best.pt; skip baseline training')
    parser.add_argument('--resume-baseline',type=Path,help='Resume baseline last.pt with two data workers')
    parser.add_argument('--amp-on-resume',action='store_true',help='Enable mixed precision at a saved epoch boundary')
    parser.add_argument('--fine-epochs',type=int,default=10)
    parser.add_argument('--resume-finetuned',type=Path,help='Resume refinement from a saved last.pt; supply --baseline too')
    args=parser.parse_args()
    if args.resume_finetuned and not args.baseline:
        parser.error('--resume-finetuned requires --baseline')
    evidence = json.loads((ROOT/'data/local-import-report.json').read_text())
    if not evidence['training_ready']:
        raise RuntimeError('Dataset import verification must pass before training')
    started = time.time()
    import torch
    torch.set_num_threads(4)
    common = dict(data=str(ROOT/'configs/sign_dataset.yaml'), imgsz=640, batch=16,
                  device=0, workers=2, seed=391, deterministic=True, amp=True,
                  project=str(ROOT/'output/training'), exist_ok=False,
                  patience=8, plots=True, cache=False)
    if args.resume_baseline:
        resume_path=args.resume_baseline.resolve()
        if args.amp_on_resume:
            checkpoint=torch.load(resume_path,map_location='cpu',weights_only=False)
            checkpoint['train_args']['amp']=True
            checkpoint['scaler']=None
            resume_path=resume_path.with_name('resume_amp.pt')
            torch.save(checkpoint,resume_path)
            del checkpoint
        model=YOLO(str(resume_path))
        model.train(resume=True,workers=2)
        baseline=Path(model.trainer.save_dir)/'weights/best.pt'
    elif args.baseline:
        baseline=args.baseline.resolve()
    else:
        model = YOLO(str(ROOT/'yolo11n.pt'))
        model.train(**common, name='detector_baseline', epochs=20,
                    hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, translate=0, scale=0,
                    shear=0, perspective=0, flipud=0, fliplr=0, mosaic=0,
                    mixup=0, copy_paste=0)
        baseline = Path(model.trainer.save_dir)/'weights/best.pt'
    if args.resume_finetuned:
        model=YOLO(str(args.resume_finetuned.resolve()))
        model.train(resume=True,workers=2)
    else:
        model = YOLO(str(baseline))
        model.train(**common, name='detector_finetuned', epochs=args.fine_epochs, optimizer='AdamW', lr0=.001,
                    hsv_h=.005, hsv_s=.15, hsv_v=.15, degrees=2, translate=.03,
                    scale=.15, shear=0, perspective=0, flipud=0, fliplr=0,
                    mosaic=0, mixup=0, copy_paste=0)
    refined=Path(model.trainer.save_dir)/'weights/best.pt'
    candidates=[]
    for name,weight in [('detector_baseline',baseline),('detector_finetuned',refined)]:
        m=YOLO(str(weight))
        val=m.val(data=common['data'],split='val',device=0,batch=16,
                  project=str(ROOT/'output/evaluation'),name=name+'_val',plots=True)
        candidates.append({'name':name,'weights':str(weight),'val':val.results_dict})
    chosen=max(candidates,key=lambda x:x['val']['metrics/mAP50-95(B)'])
    dest=ROOT/'models/sign_detector/best.pt';dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(chosen['weights'],dest)
    test=YOLO(str(dest)).val(data=common['data'],split='test',device=0,batch=16,
                           project=str(ROOT/'output/evaluation'),name='detector_test',plots=True)
    report={'candidates':candidates,'selection':'maximum VAL mAP50-95',
            'selected':chosen['name'],'test':test.results_dict,'speed_ms':test.speed,
            'elapsed_seconds':time.time()-started,
            'budget_note':'Baseline stopped at last completed checkpoint; refinement epoch budget recorded in run args.' if args.baseline else 'See phase histories for actual completed epochs.'}
    (ROOT/'output/evaluation/detector_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
