"""Automated YOLO box proposals must never silently become human ground truth."""
import copy,json
from pathlib import Path
from src.evaluation.label_review_schema import validate_review

ROOT=Path(__file__).resolve().parents[2]
PACK=ROOT/'output/review/field_review_pack_20261011'
AUTO=PACK/'autobox_20261011'

def load(name):
    return json.loads((name).read_text(encoding='utf-8'))

def test_autobox_keeps_user_review_immutable():
    initial=load(PACK/'itasmm_human_review_uploaded_20261011.json')
    new=load(AUTO/'autobox_curated_review.json')
    source=load(PACK/'review_manifest.json')
    assert validate_review(initial,source)['human_confirmed']==1
    assert validate_review(new,source)['human_confirmed']==0
    assert len(initial['frames'])==len(new['frames'])==24
    assert sum(len(f['signs']) for f in initial['frames'])==83
    assert sum(len(f['signs']) for f in new['frames'])==87
    assert all(a['image_sha256']==b['image_sha256'] and a['source_group']==b['source_group']
               for a,b in zip(initial['frames'],new['frames']))

def test_autobox_does_not_override_user_judgments_or_codes():
    old=load(PACK/'itasmm_human_review_uploaded_20261011.json')
    new=load(AUTO/'autobox_curated_review.json')
    changes=0
    for before,after in zip(old['frames'],new['frames']):
        index={s['id']:s for s in after['signs']}
        for s in before['signs']:
            a=index[s['id']]
            assert a['judgment']==s['judgment']
            assert a['label']==s['label']
            assert a['panel_role']==s['panel_role']
            changes+=bool(a.get('ai_auto_bbox_changed'))
    assert changes==32
    assert len([s for f in new['frames'] for s in f['signs']
                if s.get('source')=='ai_autobox_20261011'])==4

def test_autobox_extra_signs_require_human_review():
    data=load(AUTO/'autobox_curated_review.json')
    extra=[s for f in data['frames'] for s in f['signs'] if s.get('source')=='ai_autobox_20261011']
    assert len(extra)==4
    assert all(s['judgment']=='unreviewed' and s['ai_auto_visual_assessment']!='not_sign' for s in extra)
    assert len(load(AUTO/'rejected_model_candidates.json')['items'])==6
    assert not load(AUTO/'curated_summary.json')['ground_truth']

def test_autobox_ui_can_access_source_images():
    ui=PACK/'index_AI_tu_bat_box.html'
    assert ui.is_file()
    source=load(AUTO/'autobox_curated_review.json')
    assert all((PACK/f['image']).is_file() for f in source['frames'])
    assert 'AI' in ui.read_text(encoding='utf-8')
