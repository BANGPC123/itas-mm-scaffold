"""Sample a new driving video into the same offline field-review UI.

Source group should identify one route/session. Images only, no source video copy.
Nothing is marked as ground truth until a human confirms each frame.
"""
from pathlib import Path
import argparse,cv2,hashlib,json,re
from src.utils.config import load_config

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',required=True,type=Path)
    ap.add_argument('--source-group',required=True,help='Unique route/session id, e.g. vn_hcm_route1_20261011')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--interval-seconds',type=float,default=5.)
    ap.add_argument('--max-frames',type=int,default=120)
    ap.add_argument('--proposals',action='store_true',help='Run selected YOLO to add UNVERIFIED sign boxes')
    args=ap.parse_args()
    if not args.video.is_file():raise FileNotFoundError(args.video)
    if not re.fullmatch('[A-Za-z0-9_-]{8,90}',args.source_group):raise ValueError('Bad source group')
    if args.source_group=='test_video_development':raise ValueError('Use distinct source group for NEW recordings')
    if not(1<=args.max_frames<=500 and .5<=args.interval_seconds<=60):raise ValueError('Sampling bounds')
    out=args.output.resolve()
    if ROOT.resolve()==out or ROOT.resolve() in out.parents and 'data' in out.relative_to(ROOT).parts:
        raise ValueError('Output must be outside protected dataset source tree')
    if (out/'review_manifest.json').exists():raise FileExistsError('Refuse overwrite '+str(out))
    cap=cv2.VideoCapture(str(args.video))
    fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or not fps>0 or not count>0:raise RuntimeError('Invalid video')
    detector=None
    if args.proposals:
        from src.perception.sign_detector import SignDetector
        detector=SignDetector(load_config('perception'))
        if detector._model is None:raise RuntimeError('Selected detector not loaded')
    (out/'frames').mkdir(parents=True,exist_ok=True)
    frames=[]
    # Start after a half-second to avoid unrepresentative intro/title frame.
    positions=range(int(.5*fps),count,max(1,int(round(args.interval_seconds*fps))))
    for number,pos in enumerate(positions):
        if number>=args.max_frames:break
        cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
        ok,img=cap.read()
        if not ok:continue
        h,w=img.shape[:2]
        filename=f'{number:04d}.jpg';path=out/'frames'/filename
        if not cv2.imwrite(str(path),img,[cv2.IMWRITE_JPEG_QUALITY,92]):raise RuntimeError('Image encoding failed')
        predictions=[]
        if detector is not None:
            for i,s in enumerate(detector.detect(img[:int(.7*h)])):
                box=list(map(float,s.bbox_xyxy))
                predictions.append({'id':f'ai_{i}', 'bbox_xyxy':[round(x,2) for x in box],
                    'label':'unknown','ai_suggested_label':'YOLO traffic_sign',
                    'panel_role':'unknown','judgment':'unreviewed','source':'yolo_candidate'})
        frames.append({'id':f'frame_{number:04d}','source_group':args.source_group,
            'timestamp_seconds':round(pos/fps,3),'image':'frames/'+filename,
            'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'width':w,'height':h,'reviewer':'','reviewed_at':None,'human_confirmed':False,
            'all_visible_signs_annotated':False,'lane_status':'unreviewed',
            'signs':predictions,'lanes':[],'notes':''})
    cap.release()
    if not frames:raise RuntimeError('No frames sampled')
    with args.video.open('rb') as source_stream:
        video_digest=hashlib.file_digest(source_stream,'sha256').hexdigest()
    manifest={'schema_version':'itasmm-field-review-v1',
        'provenance':'New video sampling, source session user supplied. AI boxes never ground truth.',
        'source_video_sha256':video_digest,
        'sample_interval_seconds':args.interval_seconds,'frames':frames}
    (out/'review_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    html=(ROOT/'output/review/field_review_pack_20261011/index.html').read_text(encoding='utf-8')
    replacement='const manifest='+json.dumps(manifest,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')+';'
    html,n=re.subn(r'const manifest=.*?;\nconst original=',lambda m:replacement+'\nconst original=',html,count=1,flags=re.S)
    if n!=1:raise RuntimeError('Unable to clone review UI')
    (out/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps({'frames':len(frames),'source_group':args.source_group,
        'review_html':str(out/'index.html'),'model_proposals':bool(detector)},ensure_ascii=False))
if __name__=='__main__':main()
