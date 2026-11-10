"""Evaluate offline retrieval on the unchanged 24-case Task 6 benchmark."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.reasoning.lexical_retrieval import LexicalLegalIndex

ROOT = Path(__file__).resolve().parents[1]


def evaluate(index, cases):
    scores = []
    available = {(c['document_id'], c['locator']) for c in index.chunks}
    for case in cases:
        expected = {(r['document_id'], r['locator']) for r in case['expected']}
        if not expected <= available:
            raise ValueError('Benchmark contains targets missing from corpus')
        hits = index.search(case['query'], 4)
        rank = next((i for i, hit in enumerate(hits, 1) if (hit['document_id'], hit['locator']) in expected), None)
        scores.append({'id':case['id'], 'category':case['category'], 'rank':rank,
                       'retrieved':[{'document_id':h['document_id'],'locator':h['locator'],'chunk_id':h['chunk_id']} for h in hits]})
    n = len(scores)
    return {'case_count':n, **{f'recall_at_{k}':sum(s['rank'] is not None and s['rank']<=k for s in scores)/max(1,n) for k in (1,3,4)},
            'mrr':sum(1/s['rank'] if s['rank'] else 0 for s in scores)/max(1,n), 'cases':scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path, default=ROOT/'data/regulations/corpus.json')
    parser.add_argument('--benchmark', type=Path, default=ROOT/'data/evaluation/legal_retrieval.json')
    parser.add_argument('--output', type=Path, default=ROOT/'output/evaluation/offline_retrieval_20261011.json')
    args = parser.parse_args()
    index = LexicalLegalIndex(args.corpus)
    result = evaluate(index, json.loads(args.benchmark.read_text(encoding='utf-8'))['cases'])
    result.update({'method':'BM25 with positional bigram/trigram evidence; distinct legal locators; no generation', 'corpus_sha256':index.fingerprint,
                   'generated_at_utc':datetime.now(timezone.utc).isoformat(),
                   'independent_field_validation':False, 'legal_currency_verified':False,
                   'benchmark_role':'existing development regression suite, not unseen legal QA'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
