"""Imported YOLO11 is opt-in, advisory only, and cannot influence car warnings."""
from pathlib import Path
import json
import numpy as np
from src.perception.auxiliary_sign_review import (
    box_iou, review_candidates, AuxiliarySignReviewer,
)


def test_auxiliary_proposals_are_strictly_non_regulatory():
    out=review_candidates(
        [([20,20,60,60],.92,10),([100,50,120,70],.84,2)],
        {'10':'Cấm dừng và đỗ xe','2':'Cấm đi ngược chiều'},
        (300,400,3))
    assert len(out)==2
    assert out[0]['applicability']=='uncertain'
    assert out[0]['car_applicability_verified'] is False
    assert out[0]['human_verified'] is False
    assert out[0]['legal_warning_allowed'] is False
    assert out[0]['model']=='yolo11_optimized_user_import'


def test_duplicate_primary_and_tiny_and_invalid_candidates_are_removed():
    raw=[([20,20,60,60],.95,10),
         ([100,100,106,106],.99,2),
         ([-10,0,30,40],.99,2),
         ([120,90,170,160],.60,2),
         ([220,150,290,230],.83,2),
         ([220,150,290,230],.82,2)]
    result=review_candidates(raw,{'10':'Sign','2':'Other'},(300,400,3),
                             primary_boxes=[(18,18,62,62)])
    assert [r['class_id'] for r in result]==[2]
    assert len(result)==1


def test_model_absence_fails_closed_without_importing_ultralytics(tmp_path):
    reviewer=AuxiliarySignReviewer({'enabled':True,
        'weights_path':str(tmp_path/'missing.pt'),
        'labels_path':str(tmp_path/'missing.json')})
    assert reviewer.status=='missing_model'
    assert reviewer.detect(np.zeros((200,200,3),np.uint8))==[]


def test_wrong_weights_hash_is_never_loaded(tmp_path):
    model=tmp_path/'model.pt';model.write_bytes(b'not a checkpoint')
    labels=tmp_path/'classes.json'
    labels.write_text(json.dumps({'num_classes':59,
        'class_names':{str(i):str(i) for i in range(59)},
        'source_sha256':'wrong'}),encoding='utf8')
    reviewer=AuxiliarySignReviewer({'enabled':True,
        'weights_path':str(model),'labels_path':str(labels)})
    assert reviewer.status=='weights_hash_mismatch'
    assert reviewer.model is None


def test_duplicate_iou_is_symmetric():
    a=(20,10,70,60);b=(40,25,90,75)
    assert box_iou(a,b)==box_iou(b,a)
    assert box_iou(a,a)==1
