"""OCR field scoring and signed supplementary text regression checks."""
import copy
import pytest
from src.evaluation.ocr_metrics import normalize_plate_text,choose_consensus,score_verified_text
from src.evaluation.label_review_schema import validate_review

def reviewed_plate():
    row={'id':'f','width':640,'height':480,'image_sha256':'sha',
        'source_group':'route_a','image':'frames/0.jpg','timestamp_seconds':1.,
        'reviewer':'reviewer1','reviewed_at':'2026-10-11T11:00:00+07:00',
        'human_confirmed':True,'all_visible_signs_annotated':False,
        'lane_status':'uncertain','lanes':[],
        'signs':[{'id':'plate','bbox_xyxy':[15,10,90,50],
            'label':'unknown','judgment':'sign','panel_role':'supplementary',
            'plate_text':'6:00 - 22:00','plate_text_verified':True}]}
    return {'schema_version':'itasmm-field-review-v1','frames':[row]}

def test_verified_plate_text_accepted_with_real_reviewer():
    src=reviewed_plate()
    assert validate_review(src)['human_confirmed']==1

@pytest.mark.parametrize('bad',[
    ('plate_text',''),('plate_text_verified','not_bool'),('panel_role','primary'),
    ('judgment','not_sign'),('plate_text',5),
])
def test_reject_unreliable_verified_ocr(bad):
    src=reviewed_plate();src['frames'][0]['signs'][0][bad[0]]=bad[1]
    with pytest.raises(ValueError):
        validate_review(src)

def test_ocr_normalization_preserves_vietnamese_accents():
    assert normalize_plate_text(' 6：00 – 22:00 ')=='6:00-22:00'
    assert normalize_plate_text('cấm xe tải')!=normalize_plate_text('cấm xe tái')

def test_consensus_require_two_views_and_confidence():
    view=[
        {'text':['6:00-22:00'],'scores':[.96]},
        {'text':['6：00–22:00'],'scores':[.97]},
        {'text':['6:00-24:00'],'scores':[.99]}]
    assert choose_consensus(view)=='6:00-22:00'
    assert choose_consensus(view[:1]) is None
    assert choose_consensus([{'text':['6:00-22:00'],'scores':[.66]},
                             {'text':['6:00-22:00'],'scores':[.99]}]) is None

def test_ocr_field_exact_scoring_counts_abstentions():
    rows=[{'human_verified':True,'true_text':'6：00 – 22:00',
           'predicted_text':'6:00-22:00'},
          {'human_verified':True,'true_text':'6:00-8:00',
           'predicted_text':None},
          {'human_verified':True,'true_text':'16:00–18:00',
           'predicted_text':'16:00-19:00'}]
    m=score_verified_text(rows)
    assert m['verified_panels']==3
    assert m['accepted_panels']==2
    assert m['coverage']==pytest.approx(2/3)
    assert m['exact_accuracy_on_all']==pytest.approx(1/3)
    assert m['exact_accuracy_on_accepted']==pytest.approx(.5)

def test_ocr_metrics_block_unverified_truth():
    with pytest.raises(ValueError):
        score_verified_text([{'human_verified':False,'true_text':'6:00-8:00','predicted_text':'6:00-8:00'}])
