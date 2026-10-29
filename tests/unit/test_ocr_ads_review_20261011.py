"""OCR and background QA must remain non-authoritative and source-preserving."""
import pytest
import json
from pathlib import Path
from src.evaluation.label_review_schema import validate_review

ROOT=Path(__file__).resolve().parents[2]
PACK=ROOT/'output/review/field_review_pack_20261011'

def read(relative):
    return json.loads((PACK/relative).read_text(encoding='utf-8'))

@pytest.mark.legacy_artifact
def test_ocr_draft_source_provenance_no_human_certification():
    source=read('review_manifest.json')
    data=read('ai_ocr_advertising_review_20261011.json')
    result=validate_review(data,source)
    assert result['frames']==24 and result['human_confirmed']==0
    assert not result['detector_recall_evaluable']
    assert sum(len(f['signs']) for f in data['frames'])==87

@pytest.mark.legacy_artifact
def test_ocr_two_stable_suggestions_never_apply_law():
    doc=read('ai_ocr_advertising_review_20261011.json')
    ocr=[s for f in doc['frames'] for s in f['signs'] if 'ai_ocr_status' in s]
    assert len(ocr)==8
    stable=[s for s in ocr if s['ai_ocr_status']=='stable_two_views']
    assert len(stable)==2
    assert all(s['panel_role']=='supplementary' for s in ocr)
    assert all(s['ai_ocr_verified'] is False for s in ocr)
    assert all(s['ai_ocr_human_corroboration_required'] for s in ocr)

@pytest.mark.legacy_artifact
def test_ad_review_conflicts_preserve_non_sign_judgments():
    doc=read('ai_ocr_advertising_review_20261011.json')
    flagged=[s for f in doc['frames'] for s in f['signs'] if s.get('ai_advertising_false_accept_alert')]
    assert len(flagged)==2
    assert all(s['judgment']=='not_sign' for s in flagged)
    assert all(s['ai_background_confusing_confidence']>.9 for s in flagged)

@pytest.mark.legacy_artifact
def test_original_user_submission_untouched():
    a=read('itasmm_human_review_uploaded_20261011.json')
    b=read('ai_ocr_advertising_review_20261011.json')
    for frame_a,frame_b in zip(a['frames'],b['frames']):
        for s in frame_a['signs']:
            m=next(t for t in frame_b['signs'] if t['id']==s['id'])
            assert s['judgment']==m['judgment']
            assert s['label']==m['label'] or (s['label']=='' and m['label']=='unknown')
    assert sum(f['human_confirmed'] for f in a['frames'])==1

@pytest.mark.legacy_artifact
def test_local_review_ui_and_images_exist():
    path=PACK/'index_AI_OCR_va_loc_quang_cao.html'
    assert path.is_file()
    html=path.read_text(encoding='utf-8')
    assert 'ai_ocr_time_candidate' in html
    assert 'ai_advertising_false_accept_alert' in html
    assert 'chưa xác minh' in html
    d=read('ai_ocr_advertising_review_20261011.json')
    assert all((PACK/f['image']).is_file() for f in d['frames'])
