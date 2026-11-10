"""Validate exported human review JSON, preserving review provenance.

Usage:
python scripts/validate_human_review_20261011.py path\\to\\itasmm_human_review_*.json
Optional 2nd argument: path to the pack's review_manifest.json (default: the 24-frame starter pack).
Defaults to the still-unreviewed starter manifest (0 human labels).
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluation.label_review_schema import validate_review
manifest=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/'output/review/field_review_pack_20261011/review_manifest.json'
source=json.loads(manifest.read_text(encoding='utf-8'))
name=Path(sys.argv[1]) if len(sys.argv)>1 else manifest
data=json.loads(name.read_text(encoding='utf-8'))
summary=validate_review(data,source)
dest=manifest.parent/'validation_latest.json'
dest.write_text(json.dumps({'review_file':str(name),'summary':summary},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
if summary['human_confirmed']==0:
    print('No independently human-confirmed labels yet. No accuracy or field GT claims.')
