import importlib.util
from pathlib import Path
import cv2
import numpy as np


def test_targets_preserve_segmentation_ids_and_missing_lanes():
    path=Path(__file__).resolve().parents[2]/'scripts/retrain_lane_candidate.py'
    spec=importlib.util.spec_from_file_location('lane_training',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    mask=np.zeros((590,1640),np.uint8)
    left=np.array([[700,250],[500,589]],np.int32)
    right=np.array([[900,250],[1200,589]],np.int32)
    cv2.polylines(mask,[left],False,2,16)
    cv2.polylines(mask,[right],False,3,16)
    row,col=module.targets([right.tolist(),left.tolist()],mask)
    assert row.shape==(72,4) and col.shape==(81,4)
    assert (row[:,[0,3]]==-1).all()
    assert (col[:,[0,3]]==-1).all()
    assert 58<=row[-2,1]<=63
    assert 141<=row[-2,2]<=148
    assert (col[:,1]>=0).any() and (col[:,2]>=0).any()


def test_empty_annotation_has_only_missing_targets():
    path=Path(__file__).resolve().parents[2]/'scripts/retrain_lane_candidate.py'
    spec=importlib.util.spec_from_file_location('lane_training_empty',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    row,col=module.targets([],np.zeros((590,1640),np.uint8))
    assert (row==-1).all() and (col==-1).all()
