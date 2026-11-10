"""Conservative, auditable cleanup; whitelist generated artifacts only.

Never traverse data/, models/, third_party/, logs or historical evaluation
checkpoint directories. Default dry-run. --apply removes only known obsolete
intermediates and project Python caches after final output verification.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, time
import cv2

root=Path(__file__).resolve().parents[1]
out=root/'output'
product=out/'product_20261010'
manifest=out/'CLEANUP_20261011.json'
parser=argparse.ArgumentParser()
parser.add_argument('--apply',action='store_true')
args=parser.parse_args()
summary=json.loads((product/'verification.json').read_text())
final=product/'test_video_annotated.mp4'
if not final.is_file() or final.stat().st_size<200_000_000:
    raise RuntimeError('Final product MP4 is missing or suspiciously short')
video=cv2.VideoCapture(str(final))
frames=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
ok,first=video.read()
video.release()
if not ok or frames!=summary['output_frames'] or summary['audio_peak']<=0 or summary['speech_events']<=0:
    raise RuntimeError('Final verified 30FPS output is not intact; refuse cleanup')
targets=[product/'silent_before_motion_fix.mp4',product/'speech.wav']
caches=[root/'.pytest_cache']
for folder in ('src','backend','tests','scripts'):
    caches.extend(p for p in (root/folder).rglob('__pycache__') if p.is_dir() and not p.is_symlink())
entries=[]
for path in targets:
    if path.exists():
        size=path.stat().st_size
        digest=hashlib.sha256()
        with path.open('rb') as fh:
            for block in iter(lambda:fh.read(1024*1024),b''):
                digest.update(block)
        entries.append({'path':str(path.relative_to(root)),'bytes':size,
                        'sha256':digest.hexdigest(),'action':'delete obsolete intermediate'})
for cache in caches:
    if cache.is_dir():
        size=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
        entries.append({'path':str(cache.relative_to(root)),'bytes':size,
                        'action':'delete automatically generated cache'})
if args.apply:
    for item in entries:
        path=root/item['path']
        if path.is_dir():shutil.rmtree(path)
        elif path.is_file():path.unlink()
result={'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'), 'apply':args.apply,
        'verified_final_mp4':str(final),'frames':frames,'file_count':len(entries),
        'bytes_recovered_if_apply':sum(x['bytes'] for x in entries),
        'items':entries,'preserved':['data/','models/','output/retraining_20261009/','output/video_tests/','output/product_20261010/silent.mp4','third_party/']}
manifest.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'apply':args.apply,'files':len(entries),'MB':round(result['bytes_recovered_if_apply']/1048576,2),'manifest':str(manifest)},ensure_ascii=False))
