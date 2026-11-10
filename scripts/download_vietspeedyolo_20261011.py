"""Download source-pinned VietSpeedYOLO images and labels as TRAIN CANDIDATE.

Do not automatically incorporate in canonical splits. Preserve readme/license,
all original file names and immutable HF commit to permit source auditing.
"""
from pathlib import Path
import json,datetime
from huggingface_hub import snapshot_download,HfApi
root=Path(__file__).resolve().parents[1]
repo='NghiMe/vietspeedyolo'
metadata=HfApi().dataset_info(repo)
commit=metadata.sha
dest=root/'data/raw/external/vietspeedyolo_hf_20261011'
if (dest/'PROVENANCE.json').exists():
    raise FileExistsError('Refusing overwrite of existing external import')
dest.mkdir(parents=True,exist_ok=True)
print('DOWNLOADING',repo,commit,'to',dest,flush=True)
local=snapshot_download(repo_id=repo,repo_type='dataset',revision=commit,
                        local_dir=dest, max_workers=8,
                        allow_patterns=['images/**','labels/**','README.md','README_TRAIN.md','data.yaml','.gitattributes'])
info={'id':'vietspeedyolo_hf_20261011','repository':repo,'revision':commit,
      'source_url':'https://huggingface.co/datasets/'+repo,
      'claimed_license_in_hf_card':(metadata.card_data or {}).get('license') if isinstance(metadata.card_data,dict) else getattr(metadata.card_data,'license',None),
      'ingested_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'status':'RAW_EXTERNAL_SOURCE_TRAIN_CANDIDATE',
      'not_independent_test':True,
      'not_human_audited_by_ITAS_MM':True,
      'expected_class_mapping':{'0':'R.420','1':'R.421'},
      'statement':'Source claims MIT license. Verify image ownership, independent provenance and duplicates before downstream use. No automatic deployment.'}
(dest/'PROVENANCE.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
print('DOWNLOAD_COMPLETE',json.dumps(info,ensure_ascii=False),flush=True)
