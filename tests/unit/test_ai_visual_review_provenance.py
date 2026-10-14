"""Human/AI provenance invariants for complete assistant visual review pack."""
import json
from pathlib import Path
from src.evaluation.label_review_schema import validate_review

ROOT=Path(__file__).resolve().parents[2]
PACK=ROOT/'output/review/field_review_pack_20261011'

def test_ai_review_never_marks_human_ground_truth():
    manifest=json.loads((PACK/'review_manifest.json').read_text(encoding='utf-8'))
    audit=json.loads((PACK/'ai_visual_review/ai_assessment.json').read_text(encoding='utf-8'))
    result=validate_review(audit,manifest)
    assert result['frames']==24
    assert result['human_confirmed']==0
    assert result['exhaustive']==0
    assert all(not f['human_confirmed'] and not f['all_visible_signs_annotated'] for f in audit['frames'])
    assert all(s['judgment']=='unreviewed' for f in audit['frames'] for s in f['signs'])

def test_assessed_82_candidates_source_coordinates_unchanged():
    manifest=json.loads((PACK/'review_manifest.json').read_text(encoding='utf-8'))
    audit=json.loads((PACK/'ai_visual_review/ai_assessment.json').read_text(encoding='utf-8'))
    decisions=[]
    for a,b in zip(manifest['frames'],audit['frames']):
        assert len(a['signs'])==len(b['signs'])
        for x,y in zip(a['signs'],b['signs']):
            assert (x['id'],x['bbox_xyxy'],x['label'])==(y['id'],y['bbox_xyxy'],y['label'])
            decisions.append(y['ai_visual_assessment'])
    assert len(decisions)==82
    assert set(decisions)=={'sign','not_sign','uncertain'}

def test_review_ui_images_resolve_and_no_preset_decision():
    review=PACK/'index_ai_assisted.html'
    assert review.is_file()
    txt=review.read_text(encoding='utf-8')
    assert 'Nhận xét AI (CHƯA xác minh)' in txt
    manifest=json.loads((PACK/'ai_visual_review/ai_assessment.json').read_text(encoding='utf-8'))
    assert all((PACK/f['image']).is_file() for f in manifest['frames'])
    assert all(s['judgment']=='unreviewed' for f in manifest['frames'] for s in f['signs'])
