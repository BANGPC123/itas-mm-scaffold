"""The compound sign AI suggestions must not be mistaken for human ground truth."""
import pytest
from pathlib import Path
import json
from src.evaluation.label_review_schema import validate_review

PACK=Path(__file__).resolve().parents[2]/'output/review/field_review_pack_20261011'
AUTO=PACK/'autobox_20261011'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

@pytest.mark.legacy_artifact
def test_role_audit_full_provenance_and_total_boxes():
    a=read(AUTO/'autobox_curated_review.json')
    b=read(AUTO/'autobox_roles_review.json')
    source=read(PACK/'review_manifest.json')
    assert validate_review(b,source)['human_confirmed']==0
    assert len(a['frames'])==len(b['frames'])==24
    assert sum(len(f['signs']) for f in b['frames'])==87
    assert sum(len(f['signs']) for f in a['frames'])==87
    assert all(f['image_sha256']==g['image_sha256'] for f,g in zip(a['frames'],b['frames']))

@pytest.mark.legacy_artifact
def test_role_suggestions_are_source_grounded_and_only_five_change():
    a=read(AUTO/'autobox_curated_review.json')
    b=read(AUTO/'autobox_roles_review.json')
    changes=[];changed_labels=[]
    for x,y in zip(a['frames'],b['frames']):
        assert [s['id'] for s in x['signs']]==[s['id'] for s in y['signs']]
        for before,after in zip(x['signs'],y['signs']):
            assert before['bbox_xyxy']==after['bbox_xyxy']
            assert before['judgment']==after['judgment']
            if before['panel_role']!=after['panel_role']:
                changes.append((y['id'],after['id']))
                assert after['ai_original_panel_role']=='supplementary'
                assert after['panel_role']=='primary'
                assert after['ai_role_needs_revalidation']
            if before['label']!=after['label']:
                changed_labels.append((y['id'],before['label'],after['label']))
    assert len(changes)==5
    assert changed_labels==[('frame_000','','unknown')]

@pytest.mark.legacy_artifact
def test_8_proposed_plate_relations_have_unique_existing_parents():
    b=read(AUTO/'autobox_roles_review.json')
    total=0
    for f in b['frames']:
        mains={x['id'] for x in f['signs'] if x['panel_role']=='primary' and x['judgment']=='sign'}
        for s in f['signs']:
            if s['judgment']!='sign' or s['panel_role']!='supplementary':continue
            total+=1
            assert s['ai_panel_parent_candidate_id'] in mains
            assert s['ai_plate_text_verified'] is False
            assert s['ai_panel_semantics']=='unknown'
    assert total==8

@pytest.mark.legacy_artifact
def test_four_visible_time_texts_are_unverified_hints():
    b=read(AUTO/'autobox_roles_review.json')
    candidates=[s for f in b['frames'] for s in f['signs'] if s.get('ai_suggested_time_text')]
    assert len(candidates)==4
    assert all(s['ai_time_text_verified'] is False for s in candidates)
    assert all(s['ai_time_text_provenance']=='assistant_visual_transcription_unreviewed' for s in candidates)

@pytest.mark.legacy_artifact
def test_interface_exists_and_imagery_accessible():
    p=PACK/'index_AI_bien_chinh_bien_phu.html'
    assert p.is_file()
    html=p.read_text(encoding='utf-8')
    assert "next_ai_review" in html
    assert "ai_panel_parent_candidate_id" in html
    doc=read(AUTO/'autobox_roles_review.json')
    assert all((PACK/f['image']).is_file() for f in doc['frames'])
