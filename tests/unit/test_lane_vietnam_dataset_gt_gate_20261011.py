"""No pseudo-label leakage, verified-only training and temporal no-ghost checks."""
import json
from pathlib import Path
import numpy as np
import pytest
from src.perception.lane_temporal import LaneStabilizer
from src.perception.lane_vehicle_occlusion import suppress_over_vehicle_boxes
import importlib.util

ROOT=Path(__file__).resolve().parents[2]
def load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
validate=load_script('import_human_lane_masks_20261011').validate
PACK=ROOT/'data/lane_vietnam_review_20261011'

@pytest.mark.legacy_artifact
def test_temporal_blocks_are_source_tagged_and_not_independent_routes():
    d=json.loads((PACK/'manifest.json').read_text(encoding='utf-8'))
    assert d['sampled_frames']==600 and d['video_sources_total']==1
    assert d['split_counts']=={'TRAIN':400,'VAL_INTERNAL':100,'TEST_HOLDOUT_SAME_VIDEO':100}
    groups={}
    for f in d['frames']:
        assert f['human_confirmed'] is False
        assert f['source_video_id']=='USER_original_test_video_20261011'
        groups.setdefault(f['temporal_group'],set()).add(f['split'])
    assert len(groups)==30 and all(len(s)==1 for s in groups.values())
    assert not d['independent_road_test'] and not d['approved_for_training']

@pytest.mark.legacy_artifact
def test_ai_source_proposals_not_human_ground_truth():
    a=[json.loads(x) for x in (PACK/'ai_drafts.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(a)==600
    assert all(r['ai_only'] and not r['human_verified'] for r in a)
    assert sum(r['ai_u_lines'] for r in a)==399
    assert sum(r['ai_v2_lines'] for r in a)==365
    assert len(list((PACK/'ai_masks').glob('*.png')))==600

@pytest.mark.legacy_artifact
def test_train_import_cannot_promote_unreviewed_export_or_original_manifest():
    with pytest.raises(ValueError,match='human-reviewed'):
        validate(PACK/'manifest.json',require_minimum=True)
    assert not (ROOT/'data/processed/lane_human_verified_20261011/verified_manifest.jsonl').is_file()

def test_second_stage_vietnam_training_requires_real_reviewed_labels():
    trainer=load_script('train_segformer_verified_lanes_20261011')
    with pytest.raises(RuntimeError,match='Human-verified'):
        trainer.load_human('TRAIN')

@pytest.mark.legacy_artifact
def test_culane_segformer_candidate_is_not_promoted_as_vietnam_model():
    d=json.loads((ROOT/'output/retraining_20261011/segformer_culane_b0/training_report.json').read_text(encoding='utf-8'))
    assert d['status']=='EXPERIMENTAL_SEGFORMER_CULANE_GT'
    assert d['train_samples']>=500 and d['val_samples']>=50
    assert not d['trained_on_AI_pseudolabels']
    assert d['vietnam_field_test_accuracy'] is None
    assert not d['selected_production_model_changed']

def test_scene_change_resets_smoother_instead_of_ghost_shift():
    st=LaneStabilizer(alpha=.65)
    old=[(150,100+i*25) for i in range(12)]
    updated=[(162,100+i*25) for i in range(12)]
    black=np.zeros((720,1280,3),np.uint8)
    white=np.full((720,1280,3),255,np.uint8)
    st.update([old],1.0,black.shape,image=black)
    after=st.update([updated],1.1,white.shape,image=white)
    assert after==[updated]  # hard cut, no blending across unrelated scenes
    st.update([],1.2,white.shape,image=white)
    assert st.previous==[]
    assert st.update([old],1.3,white.shape,image=white)==[old]

def test_object_suppression_never_adds_fake_lane():
    line=[(300,200+i*16) for i in range(10)]
    assert suppress_over_vehicle_boxes([line],[(280,180,340,400)],(720,1280,3))==[]
    assert suppress_over_vehicle_boxes([],[(280,180,340,400)],(720,1280,3))==[]
