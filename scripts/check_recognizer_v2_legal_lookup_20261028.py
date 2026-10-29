"""Check which recognizer v2 codes the offline legal corpus can resolve to a QCVN appendix locator."""
from __future__ import annotations
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.reasoning.offline_legal_retrieval import OfflineLegalRetriever
OUT=ROOT/'output/kiet_wp_20261028/recognizer_v2'


def classify(ref):
    if ref is None:return 'not_in_corpus'
    if ref.get('locator_source')=='appendix_item_in_excerpt':return 'appendix_locator'
    if 'locator' in ref and ref.get('locator'):return 'corpus_heading_only'
    return 'excerpt_only'


def main():
    report=json.loads((OUT/'class_list.json').read_text(encoding='utf-8'))
    retriever=OfflineLegalRetriever();rows=[]
    for category in ('main','low_support'):
        for code in report[category]:
            ref=retriever.lookup_sign(code);kind=classify(ref)
            rows.append(dict(code=code,category=category,result=kind,locator=(ref or {}).get('locator'),corpus_locator=(ref or {}).get('corpus_locator'),
                page=(ref or {}).get('page'),locator_source=(ref or {}).get('locator_source'),document=(ref or {}).get('document') or (ref or {}).get('edition')))
    summary={k:sum(r['result']==k for r in rows) for k in ('appendix_locator','corpus_heading_only','excerpt_only','not_in_corpus')}
    (OUT/'legal_lookup.json').write_text(json.dumps(dict(summary=summary,rows=rows,retriever='OfflineLegalRetriever().lookup_sign'),ensure_ascii=False,indent=1),encoding='utf-8')
    print(json.dumps(summary));print('without appendix locator:',[(r['code'],r['result']) for r in rows if r['result']!='appendix_locator'])


if __name__=='__main__':main()
