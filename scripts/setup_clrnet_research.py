"""Install pinned upstream research CLRNet in an isolated source directory."""
from pathlib import Path
import hashlib,json,zipfile,io,requests
ROOT=Path(__file__).resolve().parents[1]
REV='7269e9d1c1c650343b6c7febb8e764be538b1aed'
def download(url):
 r=requests.get(url,timeout=(30,180));r.raise_for_status();return r.content

def main():
 out=ROOT/'third_party/CLRNet_research'
 if not out.exists():
  data=download(f'https://codeload.github.com/Turoad/CLRNet/zip/{REV}')
  with zipfile.ZipFile(io.BytesIO(data)) as z:
   for name in z.namelist():
    rel=Path(*Path(name).parts[1:])
    if not rel.parts or name.endswith('/'):continue
    if '..' in rel.parts:raise ValueError('Unsafe archive')
    p=out/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
  # Explicit portability substitutions; original source SHA recorded below.
  p=out/'clrnet/ops/nms.py'
  p.write_text(p.read_text().replace('from . import nms_impl','from src.perception.clrnet_portable_nms import nms as portable_nms').replace('return nms_impl.nms_forward(boxes, scores, overlap, top_k)','return portable_nms(boxes, scores, overlap, top_k)'),encoding='utf-8')
  p=out/'clrnet/models/heads/clr_head.py'
  p.write_text(p.read_text().replace('np.bool)','np.bool_)'),encoding='utf-8')
  (out/'ITAS_PROVENANCE.json').write_text(json.dumps({'repository':'https://github.com/Turoad/CLRNet','commit':REV,'archive_sha256':hashlib.sha256(data).hexdigest(),'changes':['CPU reference lane NMS','NumPy np.bool compatibility'],'license':'Apache-2.0; preserve LICENSE and ops notices'},indent=2))
 dest=ROOT/'models/lane/candidates/clrnet_official/culane_r18.pth'
 dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():
  url='https://github.com/Turoad/CLRNet/releases/download/models/culane_r18.pth.zip'
  data=download(url)
  with zipfile.ZipFile(io.BytesIO(data)) as z:
   name=next(n for n in z.namelist() if n.endswith('.pth'))
   dest.write_bytes(z.read(name))
  (dest.parent/'provenance.json').write_text(json.dumps({'url':url,'zip_sha256':hashlib.sha256(data).hexdigest(),'checkpoint_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'pretraining':'author CULane ResNet18','research_only':True},indent=2))
 print('CLRNET_READY',str(dest),flush=True)
if __name__=='__main__':main()
