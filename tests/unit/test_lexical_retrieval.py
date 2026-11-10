import json
import pytest
from src.reasoning.lexical_retrieval import LexicalLegalIndex, tokens
from tests.unit.test_document_loader import _corpus


def index(tmp_path):
    corpus=_corpus()
    corpus['chunks'][0]['text']='Biển cấm đỗ xe P.131a có hiệu lực với xe cơ giới.'
    corpus['chunks'][1]['text']='Biển cấm đi ngược chiều P.102.'
    path=tmp_path/'corpus.json'
    path.write_text(json.dumps(corpus),encoding='utf-8')
    return LexicalLegalIndex(path)


def test_lexical_hits_retain_exact_source_identity_and_full_excerpt(tmp_path):
    idx=index(tmp_path)
    result=idx.search('cấm đỗ xe P.131a',1)[0]
    assert result['excerpt']==idx.chunks[0]['text']
    assert result['document_id']=='law-36-2024-qh15'
    assert result['locator']=='Điều 7'
    assert result['source_sha256']=='a'*64
    assert result['legal_currency_verified'] is False
    assert idx.search('cấm đỗ xe P.131a',1)==idx.search('cấm đỗ xe P.131a',1)


def test_unknown_query_and_document_do_not_fabricate_matches(tmp_path):
    idx=index(tmp_path)
    assert idx.search('xyzabracadabra',4)==[]
    assert idx.search('cấm xe',4,document='missing')==[]


@pytest.mark.parametrize('top_k',[0,21,True,1.5])
def test_invalid_top_k_rejected(tmp_path,top_k):
    with pytest.raises(ValueError): index(tmp_path).search('xe',top_k)


def test_vietnamese_normalization_handles_d_stroke():
    assert tokens('Đỗ XE')==tokens('do xe')
