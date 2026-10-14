import copy
import pytest
from src.evaluation.label_review_schema import validate_review

def base():
    frame={'id':'frame0','image_sha256':'sha','source_group':'sample',
        'width':640,'height':480,'signs':[{'id':'s1','bbox_xyxy':[20,40,60,80],
        'judgment':'unreviewed','panel_role':'unknown','label':'P.130'}],
        'lanes':[],'lane_status':'unreviewed','human_confirmed':False,
        'reviewer':'','reviewed_at':None,'all_visible_signs_annotated':False}
    return {'schema_version':'itasmm-field-review-v1','frames':[frame]}

def test_starter_manifest_is_not_fake_gt():
    a=validate_review(base())
    assert a['human_confirmed']==0 and not a['detector_recall_evaluable']

def test_human_review_cannot_leave_ai_sign_unreviewed():
    data=base();f=data['frames'][0]
    f.update(human_confirmed=True,reviewer='reviewer01',reviewed_at='2026-10-11',lane_status='uncertain')
    with pytest.raises(ValueError):validate_review(data)

def test_human_review_classification_only_no_detector_recall():
    data=base();f=data['frames'][0]
    f.update(human_confirmed=True,reviewer='reviewer01',reviewed_at='2026-10-11',
        lane_status='uncertain')
    f['signs'][0]['judgment']='sign'
    assert validate_review(data)['confirmed_signs']==1
    assert not validate_review(data)['detector_recall_evaluable']

def test_exhaustive_requires_no_uncertain_candidates():
    data=base();f=data['frames'][0]
    f.update(human_confirmed=True,reviewer='reviewer01',reviewed_at='now',
             lane_status='no_visible_lane',all_visible_signs_annotated=True)
    f['signs'][0]['judgment']='uncertain'
    with pytest.raises(ValueError):validate_review(data)
    f['signs'][0]['judgment']='not_sign'
    score=validate_review(data)
    assert score['confirmed_negatives']==1 and score['exhaustive']==1

@pytest.mark.parametrize('bbox',[
    [-1,0,20,30],[20,20,20,30],[20,20,641,40],
    [20,20,float('nan'),40],[20,20,40,float('inf')]])
def test_invalid_boxes_rejected(bbox):
    data=base();data['frames'][0]['signs'][0]['bbox_xyxy']=bbox
    with pytest.raises(ValueError):validate_review(data)

def test_hash_provenance_cannot_be_changed():
    a=base()
    import copy
    source=copy.deepcopy(a)
    a['frames'][0]['image_sha256']='tamper'
    with pytest.raises(ValueError):validate_review(a,source)

def test_lane_sanity_checks():
    data=base();f=data['frames'][0]
    f['lane_status']='annotated'
    with pytest.raises(ValueError):validate_review(data)
    f['lanes']=[{'role':'ego_left','points':[[10,20],[30,470]]}]
    assert validate_review(data)['frames']==1
    f['lanes'][0]['points'][1]=[800,470]
    with pytest.raises(ValueError):validate_review(data)
