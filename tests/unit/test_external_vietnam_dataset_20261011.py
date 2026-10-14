"""External Vietnam dataset source evidence and no-fake-GT regression checks."""
from pathlib import Path
import json,hashlib
from src.evaluation.label_review_schema import validate_review

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data/raw/external'
REV=ROOT/'output/review'
PROC=ROOT/'data/processed/candidates/vietspeedyolo_hf_20261011'
AUDIT=ROOT/'data/audit/external_vietspeedyolo_20261011'

def load(p):
    return json.loads(p.read_text(encoding='utf-8'))

def test_external_dataset_source_is_pinned_and_all_candidate_images_have_hashes():
    provenance=load(RAW/'vietspeedyolo_hf_20261011/PROVENANCE.json')
    assert len(provenance['revision'])==40
    assert provenance['not_independent_test'] is True
    selection=load(PROC/'selection_report.json')
    assert selection['selected_train_candidates']==253
    assert selection['image_downloaded']==357
    assert selection['source_train_images']==330
    assert selection['source_val_images']==27
    rows=[json.loads(x) for x in (PROC/'train_candidate_manifest.jsonl').read_text().splitlines()]
    assert len(rows)==253
    assert len({x['sha256'] for x in rows})==len(rows)
    assert all(x['split']=='TRAIN_CANDIDATE_ONLY' and x['not_general_background']
               and not x['approved_by_human'] for x in rows)
    assert all(x['source_revision']==provenance['revision'] for x in rows)
    audit=load(AUDIT/'audit.json')
    assert audit['near_duplicate_test_total']==0
    assert audit['train_val_same_sha']==2

def test_new_independently_sourced_vietnam_videos_have_distinct_copyright_records():
    info=load(RAW/'vietnam_commons_street_20261011/SOURCES.json')['sources']
    assert len(info)==2
    assert {x['license_id'] for x in info}=={'CC0','CC BY-SA 4.0'}
    assert len({x['sha256'] for x in info})==2
    for row in info:
        file=RAW/'vietnam_commons_street_20261011'/row['filename']
        assert file.is_file() and file.stat().st_size==row['size_bytes']
        assert hashlib.sha256(file.read_bytes()).hexdigest()==row['sha256']
        assert row['status'].startswith('UNREVIEWED_')

def test_59_new_video_frames_and_five_ai_ads_are_all_unconfirmed():
    frames=ads=0
    hashes=set()
    for src in ['commons_hanoi_20261011','commons_saigon_govap_20261011']:
        pack=REV/src
        orig=load(pack/'review_manifest.json')
        annotated=load(pack/'ai_hard_negative_candidates_20261011.json')
        val=validate_review(annotated,orig)
        assert val['human_confirmed']==0
        assert val['exhaustive']==0
        assert annotated['source_video_sha256']==orig['source_video_sha256']
        hashes.add(orig['source_video_sha256'])
        frames+=len(annotated['frames'])
        ads+=sum(s['source']=='AI_VISUAL_ADVERTISING_SUGGESTION_NOT_GT'
                 for f in annotated['frames'] for s in f['signs'])
        assert all((pack/f['image']).is_file() and not f['human_confirmed']
                   for f in annotated['frames'])
        assert all(s['judgment']=='unreviewed'
                   for f in annotated['frames'] for s in f['signs'])
    assert len(hashes)==2
    assert frames==59
    assert ads==5

def test_external_pictorial_sign_images_remain_ai_quality_review_only():
    pack=REV/'external_vietspeedyolo_20261011'
    source=load(pack/'review_manifest.json')
    audit=load(pack/'ai_visual_quality_triage.json')
    assert len(source['frames'])==15
    assert audit['human_verified']==0
    assert audit['ai_rejected_obvious_augmentation']==5
    assert validate_review(source)['human_confirmed']==0

def test_lane_spotcheck_has_no_unearned_field_F1():
    summary=load(ROOT/'output/evaluation/commons_lane_spotcheck_20261011/summary.json')
    assert summary['frames_evaluated']==12
    assert summary['human_verified_lane_lines']==0
    assert summary['status']=='VISUAL_PREDICTIONS_ONLY_NO_METRICS'
    assert sum(x['polylines_predicted'] for x in summary['results'])==0
