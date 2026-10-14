"""Acquire immutable research candidate of official YOLOPv2, never overwrite selected UFLD."""
from pathlib import Path
import requests,hashlib,json,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'models/lane/candidates/yolopv2_official_20261011'
OUT.mkdir(parents=True,exist_ok=True)
URL='https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt'
FILE=OUT/'yolopv2.pt'
if not FILE.exists():
    response=requests.get(URL,stream=True,timeout=(20,130),headers={'User-Agent':'ITASMM-Research/1.0'})
    response.raise_for_status()
    partial=FILE.with_suffix('.partial')
    sha=hashlib.sha256();n=0
    with partial.open('wb') as f:
        for b in response.iter_content(1024*1024):
            if not b:continue
            f.write(b);sha.update(b);n+=len(b)
            if n%(30*1024*1024)<len(b):print('DOWNLOADING',n,'bytes',flush=True)
    if n<1_000_000:raise RuntimeError('Likely not a real weight file')
    partial.rename(FILE)
else:
    sha=hashlib.sha256(FILE.read_bytes());n=FILE.stat().st_size
record={'download_url':URL,'size_bytes':n,'sha256':sha.hexdigest(),
  'upstream':'CAIC-AD/YOLOPv2','training_domain':'BDD100K',
  'license':'upstream repository MIT; check model weights license separately',
  'status':'EXPERIMENT_ONLY_NO_PRODUCTION_PROMOTION'}
(OUT/'provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('CANDIDATE_WEIGHT_ACQUIRED',json.dumps(record),flush=True)
