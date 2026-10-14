from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def main():
    import gdown
    from ultralytics import YOLO
    destination=ROOT/'models/lane/culane_res18.pth'
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.is_file():
        partial=destination.with_suffix('.pth.partial')
        result=gdown.download(id='1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W',output=str(partial),quiet=False)
        if not result or not partial.is_file():raise RuntimeError('Author checkpoint download failed')
        partial.replace(destination)
    YOLO(str(ROOT/'yolo11n.pt'))
    record={'lane':{'source':'https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2',
                    'checkpoint_url':'https://drive.google.com/file/d/1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W/view',
                    'dataset':'CULane','backbone':'ResNet18','sha256':sha256(destination)}}
    (ROOT/'models/pretrained_sources.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
