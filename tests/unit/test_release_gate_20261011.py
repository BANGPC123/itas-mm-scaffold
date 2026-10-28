"""Final no-promotion contract for 2026-10-11 ITAS-MM research stage."""
import pytest
import json
from pathlib import Path
from backend.routers.pipeline import status
from src.utils.config import load_config
ROOT=Path(__file__).resolve().parents[2]

def read(relative):
    return json.loads((ROOT/relative).read_text(encoding='utf-8'))

@pytest.mark.legacy_artifact
def test_release_gate_blocks_unapproved_retraining():
    result=read('output/evaluation/ITAS_MM_RELEASE_GATE_20261011.json')
    assert result['model_release_approved'] is False
    assert result['status'].startswith('BLOCKED_')
    assert len(result['workstreams'])==4
    assert all(entry['blocked_by'] for entry in result['workstreams'])

def test_api_status_exposes_gate_and_does_not_claim_field_validation():
    actual=status()
    assert actual['field_validated'] is False
    assert actual['legal_corpus_validated'] is False
    assert actual['release_readiness']['model_release_approved'] is False
    assert actual['mode']=='research_preview'

def test_model_configuration_still_points_to_known_baselines():
    """Detector default moved to the 2026-10-26 fine-tune; other models unchanged.

    On the immutable TEST split the fine-tune improves recall 0.8509->0.8754 and
    mAP50 0.9194->0.9357, with a bootstrap CI that excludes 0
    (docs/KIET_REVALIDATION_20261026.md). The previous retrained_20261009.pt
    detector remains available for rollback.
    """
    cfg=load_config('perception')
    assert cfg['sign_detector']['weights_path']=='models/sign_detector/finetuned_20261026.pt'
    assert cfg['sign_recognizer']['weights_path']=='models/sign_recognizer/retrained_20261009.pt'
    assert cfg['lane_detector']['weights_path']=='models/lane/selected_20261009.pth'
    assert cfg['lane_detector']['filter_geometry'] is False

@pytest.mark.legacy_artifact
def test_33_class_checkpoint_is_experimental_only():
    r=read('output/retraining_20261011/recognizer_33class_EXPERIMENT/report.json')
    assert len(r['classes'])==33
    assert r['classes'][-3:]==['R.420','R.421','unknown']
    assert r['candidate_saved'] is True
    assert r['selected_checkpoint_unchanged']
    assert (ROOT/'output/retraining_20261011/recognizer_33class_EXPERIMENT/candidate_33classes.pt').exists()

@pytest.mark.legacy_artifact
def test_e2e_latency_is_separate_from_safety_accuracy():
    r=read('output/evaluation/vietnam_e2e_benchmark_20261011/benchmark.json')
    assert r['frames']==20 and r['gps_is_mocked']
    assert r['p95_wall_ms']>0 and r['effective_serial_throughput_fps']>0
    assert not r['ground_truth_metrics_available']
    assert r['audio_calls']==0
