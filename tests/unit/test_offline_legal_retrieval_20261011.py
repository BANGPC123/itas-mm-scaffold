"""Capstone law retriever: official source provenance and no invented fine."""
from pathlib import Path
import json,pytest
from fastapi.testclient import TestClient
from backend.main import app
from src.reasoning.offline_legal_retrieval import OfflineLegalRetriever
from src.utils.types import SignDetection

ROOT=Path(__file__).resolve().parents[2]
def test_official_corpus_index_has_three_document_families_and_real_sources():
    retriever=OfflineLegalRetriever()
    assert retriever.manifest['chunks']>=650
    assert len(retriever.chunks)==retriever.manifest['chunks']
    assert {x['document'] for x in retriever.chunks}=={
        '36/2024/QH15','QCVN 41:2024/BGTVT'}
    # Nghị định 168 is present in original PDFs but is scanned.
    assert any(x['document']=='168/2024/NĐ-CP' and x['pages_with_extracted_text']==0
               for x in retriever.manifest['source_stats'])
    assert not retriever.manifest['all_current_amendments_checked']

def test_full_text_search_is_grounded_in_official_source_pdf_page():
    r=OfflineLegalRetriever()
    results=r.search('quy tắc chấp hành báo hiệu đường bộ',top_k=3)
    assert len(results)==3
    assert all(x['excerpt'] and x['source_url'].startswith('https://datafiles.chinhphu.vn/')
               and x['page']>0 and x['research_only'] for x in results)
    assert all(not x['legal_currency_verified'] for x in results)

def test_code_lookup_prefers_project_reference_and_abstains_from_fines():
    search=OfflineLegalRetriever()
    item=search.lookup_sign('P.130')
    assert item and item['code']=='P.130'
    assert 'source_url' in item and not item['legal_currency_verified']
    predicted=SignDetection('P.130',.8,(1,1,40,40),panel_role='primary')
    response=search.for_signs([predicted],'Urban')
    assert response.references and len(response.retrieved_chunks)>0
    assert 'mức phạt' in response.guidance_text
    assert not any(str(x).strip().startswith('Phạt') for x in response.retrieved_chunks)

def test_unknown_sign_returns_no_fabricated_regulation():
    search=OfflineLegalRetriever()
    r=search.for_signs([SignDetection('unknown',.5,(1,1,40,40))],'Urban')
    assert r.references==[] and r.retrieved_chunks==[]

def test_legal_search_api_returns_traceable_urls_without_ollama():
    client=TestClient(app)
    response=client.post('/pipeline/legal-search',json={'query':'báo hiệu đường bộ','top_k':2})
    assert response.status_code==200
    d=response.json()
    assert d['retrieved_count']==2 and d['penalty_amount_verified'] is False
    assert d['retrieval_backend']=='task6_bm25'
    assert all('#' not in x['source_url'] and x['locator'] and x['source_sha256'] for x in d['hits'])

@pytest.mark.parametrize('bad',[{'query':''},{'query':'x','top_k':100}])
def test_illegal_search_payload_rejected(bad):
    assert TestClient(app).post('/pipeline/legal-search',json=bad).status_code==422
