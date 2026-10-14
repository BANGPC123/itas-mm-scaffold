"""Unit/regression tests for independent-field evaluation and safety gates."""
import json,hashlib
from pathlib import Path
import numpy as np
import pytest
from src.evaluation.field_metrics import associate,field_gate,iou,lane_pixel_counts,score_frames

def frame(*,all_signs=True,gt=None,neg=None,unknown=None,lanes=None,lane_status='unreviewed',
          split='TEST',source='route1',name='f0'):
    return {'image_sha256':'sha','image_path':'test.jpg','frame_id':name,'source_group':source,
            'source_video_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'split':split,'reviewer':'human_a','reviewed_at':'date',
            'all_visible_signs_annotated':all_signs,'lane_status':lane_status,
            'signs':(gt or [])+(neg or [])+(unknown or []),'lanes':lanes or []}

def box(x,y,x2,y2,label='P.130',judgment='sign'):
    return {'id':'item','bbox_xyxy':[x,y,x2,y2],'judgment':judgment,
            'label':label,'panel_role':'primary'}

def entry(gt,pred):
    return {'gt':gt,'pred':{'signs':pred,'lanes':[]},'image_shape':(100,100,3)}

def prediction(coords,label='P.130'):
    return {'bbox_xyxy':list(coords),'label':label,'panel_role':'primary'}

def test_one_to_one_matching_duplicate_detection_is_fp():
    a=box(10,10,30,30)
    result=score_frames([entry(frame(gt=[a]),[prediction(a['bbox_xyxy']),
                       prediction(a['bbox_xyxy'])])])
    assert result['detection']['true_positive']==1
    assert result['detection']['false_positive']==1
    assert result['detection']['false_negative']==0
    assert result['detection']['precision']==pytest.approx(.5)
    assert result['detection']['recall']==pytest.approx(1.)

def test_missed_real_sign_recall_is_zero():
    d=score_frames([entry(frame(gt=[box(10,10,30,30)]),[])])['detection']
    assert d['false_negative']==1
    assert d['recall']==0.

def test_partially_reviewed_frame_does_not_fake_detection_precision():
    d=score_frames([entry(frame(all_signs=False,gt=[box(10,10,30,30)]),[])])
    assert d['detection']['status']=='blocked_no_exhaustive_review_frames'
    assert d['reviewed_sign_boxes']==1

def test_explicit_background_box_has_independent_false_accept_metric():
    neg=box(40,40,60,60,label='background',judgment='not_sign')
    gt=frame(neg=[neg],gt=[box(10,10,30,30)])
    det=[prediction((40,40,60,60),label='P.127'),prediction((10,10,30,30))]
    score=score_frames([entry(gt,det)])
    assert score['non_sign_regions']['false_known_sign_accepts']==1
    assert score['non_sign_regions']['false_accept_rate']==1.
    assert score['detection']['false_positive']==1

def test_unknown_class_on_background_does_not_count_as_accepted_sign():
    neg=box(40,40,60,60,label='background',judgment='not_sign')
    d=score_frames([entry(frame(neg=[neg],all_signs=False),[prediction((40,40,60,60),'unknown')])])
    assert d['non_sign_regions']['false_known_sign_accepts']==0

def test_untouched_uncertain_is_not_false_positive():
    uncertain=box(40,40,60,60,label='unknown',judgment='uncertain')
    gt=frame(all_signs=True,unknown=[uncertain])
    d=score_frames([entry(gt,[prediction((40,40,60,60))])])
    assert d['detection']['status']=='blocked_no_exhaustive_review_frames'

def test_exact_code_accuracy_is_not_detection_accuracy():
    gt=box(10,10,30,30,label='P.130')
    d=score_frames([entry(frame(gt=[gt]),[prediction(gt['bbox_xyxy'],'P.127')])])
    assert d['detection']['true_positive']==1
    assert d['sign_code']['evaluated']==1
    assert d['sign_code']['accuracy']==0.

def test_lane_tolerance_exact_match_and_miss():
    a={'role':'ego_left','points':[[10,20],[20,95]]}
    assert lane_pixel_counts((100,100,3),[a],[a])['matched_pred_pixels']>0
    rec=frame(gt=[],lane_status='annotated',lanes=[a])
    d=score_frames([{'gt':rec,'pred':{'signs':[],'lanes':[a['points']]},'image_shape':(100,100,3)}])
    assert d['lane']['pixel_tolerance_f1']==pytest.approx(1.)
    far=[(80,20),(90,95)]
    d=score_frames([{'gt':rec,'pred':{'signs':[],'lanes':[far]},'image_shape':(100,100,3)}])
    assert d['lane']['pixel_tolerance_f1']==0.

def test_lane_metric_blocked_without_annotation():
    d=score_frames([entry(frame(all_signs=False),[])])
    assert d['lane']['status']=='blocked_no_reviewed_lane_annotations'

def test_gate_blocks_dev_leakage_small_test_and_missing_lanes():
    rows=[frame(split='TRAIN',source='same'),
          frame(split='TEST',source='same')]
    ready=field_gate(rows)
    assert not ready['field_evidence_ready']
    assert 'test_route_leakage' in ready['blockers']
    assert 'need_more_human_exhaustive_test_frames' in ready['blockers']

def test_gate_can_pass_evidence_minimum_not_certification():
    records=[frame(split='TRAIN',source='train')]
    lane={'role':'ego_left','points':[[1,1],[2,3]]}
    records += [frame(split='TEST',source='test1',name=f'a{i}',
                       lane_status='annotated',lanes=[lane]) for i in range(15)]
    records += [frame(split='TEST',source='test2',name=f'b{i}') for i in range(15)]
    d=field_gate(records)
    assert d['field_evidence_ready']
    assert d['test_route_groups']==2
    assert d['exhaustive_test_frames']==30
    assert 'NOT safety validation' in d['note']

def test_unattributed_ground_truth_rejected():
    f=frame();f['reviewer']=''
    with pytest.raises(ValueError):
        score_frames([entry(f,[])])

@pytest.mark.parametrize('invalid',[0,-5,21,100])
def test_bad_lane_tolerance_rejected(invalid):
    with pytest.raises(ValueError):
        lane_pixel_counts((100,100,3),[],[],invalid)
