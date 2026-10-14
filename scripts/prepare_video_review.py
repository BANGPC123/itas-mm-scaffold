"""Create a local review packet; predictions remain unverified proposals."""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.perception.sign_catalog import NAMES_VI

def main():
    parser=argparse.ArgumentParser();parser.add_argument('video',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'output/review/test_video_development')
    parser.add_argument('--holdout',action='store_true',help='Freeze current code/models before preparing an unseen video; requires a distinct source hash')
    args=parser.parse_args();out=args.output.resolve()
    if args.holdout and out==ROOT/'output/review/test_video_development':
        raise ValueError('Use a separate --output directory for holdout')
    out.mkdir(parents=True,exist_ok=True)
    (out/'images').mkdir(exist_ok=True)
    with args.video.open('rb') as source:digest=hashlib.file_digest(source,'sha256').hexdigest()
    development=ROOT/'output/review/test_video_development/draft.json'
    known=json.loads(development.read_text(encoding='utf-8')) if development.exists() else None
    if args.holdout:
        if known and digest==known['source_sha256']:raise ValueError('Development video cannot be independent holdout')
        paths=list((ROOT/'src').rglob('*.py'))+list((ROOT/'configs').glob('*.yaml'))+list((ROOT/'models').rglob('*.pt'))+list((ROOT/'models').rglob('*.pth'))
        hashes={}
        for path in paths:
            with path.open('rb') as file:hashes[str(path.relative_to(ROOT))]=hashlib.file_digest(file,'sha256').hexdigest()
        (out/'frozen_system.json').write_text(json.dumps({'source_sha256':digest,'files':hashes,'status':'awaiting_human_labels','different_trip_requires_human_confirmation':True},indent=2))
    cap=cv2.VideoCapture(str(args.video));fps=cap.get(cv2.CAP_PROP_FPS);duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    prior=ROOT/'output/video_tests/test_video_final/predictions.jsonl'
    use_prior=not args.holdout and known is not None and digest==known['source_sha256']
    rows=[json.loads(line) for line in prior.read_text(encoding='utf-8').splitlines()] if prior.is_file() and use_prior else []
    samples=[]
    # Uniform time blocks include negatives; select the most detections within each block for review.
    for block in range(24):
        candidates=[r for r in rows if block*duration/24<=r['timestamp_seconds']<(block+1)*duration/24]
        row=max(candidates,key=lambda r:(r['detections'],r.get('known_predictions',0))) if candidates else None
        ts=row['timestamp_seconds'] if row else (block+.5)*duration/24
        pictures=[]
        for offset in [-.2,0,.2]:
            cap.set(cv2.CAP_PROP_POS_MSEC,max(0,ts+offset)*1000);ok,image=cap.read()
            if not ok:raise RuntimeError(f'Cannot decode {ts+offset}')
            image=cv2.resize(image,(1280,720));filename=f'images/{block:03d}_{len(pictures)}.jpg'
            cv2.imwrite(str(out/filename),image,[cv2.IMWRITE_JPEG_QUALITY,94]);pictures.append(filename)
        proposals=[]
        for i,s in enumerate(row['signs'] if row else []):
            proposals.append({'id':f's{i+1}','label':s['label'],'bbox_xyxy':s['bbox_xyxy'],'panel_role':'unknown','allowed_vehicles':[],
                              'overhead':None,'parent_id':None,'expected_applicability':'uncertain'})
        samples.append({'id':f'frame_{block:03d}','timestamp_seconds':ts,'images':pictures,'width':1280,'height':720,
            'vehicle':'car','visibility':'uncertain','ego_boundaries':[], 'signs':proposals,
            'human_confirmed':False,'reviewer':'','reviewed_at':None,'notes':'AI box/code proposals only; correct omissions and false positives.'})
    cap.release()
    packet={'schema_version':1,'sign_names':NAMES_VI,'source_sha256':digest,'source_video':str(args.video.resolve()),'purpose':'holdout_pending_human_verification' if args.holdout else 'development_review_not_independent_holdout',
            'sampling':'24 time blocks; maximum detector count per block; biased review set, not prevalence estimate' if rows else '24 uniform temporal samples; no model-based selection','samples':samples}
    (out/'draft.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2),encoding='utf-8')
    template=(ROOT/'scripts/video_review_template.html').read_text(encoding='utf-8')
    (out/'index.html').write_text(template.replace('__PACKET__',json.dumps(packet,ensure_ascii=False).replace('<','\\u003c')),encoding='utf-8')
    print(json.dumps({'packet':str(out),'frames':len(samples),'human_verified':0},indent=2))

if __name__=='__main__':main()
