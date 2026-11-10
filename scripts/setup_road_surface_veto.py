"""Download the pinned semantic road/sidewalk veto without changing runtime packages."""
from pathlib import Path
import hashlib,json,requests
ROOT=Path(__file__).resolve().parents[1]
REPO='nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
REVISION='21b3847fae21ddee674abd31129307b6a1235bd9'
EXPECTED={'config.json':'649ba4afbaa11ee730b14ad5e31bd50f2e2d5be661a291e31a91512510dba8d9',
 'preprocessor_config.json':'d96ab2cb112985c19d5ea32f1d1f2369104fc78b1379f5ce26e33ca8646202a7',
 'pytorch_model.bin':'027ed78d8ff9c535df5a66361c92b9673cca3cb923cbeb8c5802385b6b93194c'}
def main():
 target=ROOT/'models/lane/candidates/segformer_cityscapes_20261011';target.mkdir(parents=True,exist_ok=True);files={}
 for name,expected in EXPECTED.items():
  path=target/name
  if path.is_file():
   data=path.read_bytes()
   if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Existing file hash mismatch')
  else:
   response=requests.get('https://huggingface.co/'+REPO+'/resolve/'+REVISION+'/'+name,timeout=(15,60));response.raise_for_status();data=response.content
   if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Downloaded file hash mismatch')
   path.write_bytes(data)
  files[name]={'sha256':expected,'bytes':len(data)}
 (target/'PROVENANCE.json').write_text(json.dumps({'repo':REPO,'revision':REVISION,'files':files},indent=2),encoding='utf-8')
 print('ROAD_SURFACE_MODEL_VERIFIED',flush=True)
if __name__=='__main__':main()
