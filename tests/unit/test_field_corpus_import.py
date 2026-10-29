import json,importlib.util,hashlib
from pathlib import Path
import pytest
from src.evaluation.label_review_schema import validate_review

SPEC=importlib.util.spec_from_file_location('field_import',Path(__file__).resolve().parents[2]/'scripts/import_human_field_corpus_20261011.py')
module=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def pack(tmp_path,split,group='route_a',reviewed=True,exhaustive=True):
    root=tmp_path/('pack_'+split)
    (root/'frames').mkdir(parents=True)
    image=root/'frames'/'0.jpg'
    image.write_bytes(b'fake_image_bytes_test_only')
    sha=hashlib.sha256(image.read_bytes()).hexdigest()
    obj={'id':'x','image':'frames/0.jpg','image_sha256':sha,
         'source_group':group,'width':640,'height':480,
         'human_confirmed':reviewed,'reviewer':'test_reviewer' if reviewed else '',
         'reviewed_at':'2026-10-11T00:00:00Z' if reviewed else None,
         'all_visible_signs_annotated':exhaustive,'lane_status':'no_visible_lane',
         'lanes':[],'signs':[{'id':'s1','label':'unknown','bbox_xyxy':[50,50,70,70],
                             'panel_role':'unknown','judgment':'not_sign'}]}
    data={'schema_version':'itasmm-field-review-v1','frames':[obj]}
    (root/'review_manifest.json').write_text(json.dumps(data))
    export=root/'reviewed.json';export.write_text(json.dumps(data))
    return {'pack':str(root),'review':str(export),'split':split}


def test_field_corpus_requires_self_attested_review(tmp_path):
    a=pack(tmp_path,'TRAIN',reviewed=False)
    with pytest.raises(ValueError,match='No human'):
        module.build([a],tmp_path/'dataset')


def test_field_corpus_tracks_negatives_but_not_unsupported_accuracy(tmp_path):
    a=pack(tmp_path,'TRAIN')
    d=module.build([a],tmp_path/'dataset')
    assert d['confirmed_negative_instances']==1
    assert d['source_groups']=={'route_a':'TRAIN'}
    assert not d['separate_field_test_ready']
    assert (tmp_path/'dataset'/'frames.jsonl').is_file()
    with pytest.raises(FileExistsError):
        module.build([a],tmp_path/'dataset')


def test_field_corpus_cross_split_source_rejected(tmp_path):
    a=pack(tmp_path,'TRAIN',group='same_session')
    b=pack(tmp_path,'VAL',group='same_session')
    with pytest.raises(ValueError,match='multiple splits'):
        module.build([a,b],tmp_path/'dataset')


def test_field_corpus_duplicate_image_rejected(tmp_path):
    a=pack(tmp_path,'TRAIN',group='route_a')
    b=pack(tmp_path,'VAL',group='route_b')
    with pytest.raises(ValueError,match='Duplicate image'):
        module.build([a,b],tmp_path/'dataset')


def test_review_source_metadata_tamper_rejected():
    sample={'schema_version':'itasmm-field-review-v1','frames':[{
        'id':'x','width':100,'height':100,'image':'frames/a.jpg','image_sha256':'abcdef',
        'source_group':'session','timestamp_seconds':10,'human_confirmed':False,
        'signs':[],'lanes':[],'lane_status':'unreviewed'}]}
    src=json.loads(json.dumps(sample))
    sample['frames'][0]['source_group']='different_session'
    with pytest.raises(ValueError):validate_review(sample,src)


def test_same_original_video_hash_not_allowed_across_splits_even_when_route_names_differ(tmp_path):
    a=pack(tmp_path,'TRAIN',group='vn_routeA')
    b=pack(tmp_path,'VAL',group='vn_routeB')
    for p in (a,b):
        path=Path(p['pack'])/'review_manifest.json'
        data=json.loads(path.read_text(encoding='utf-8'))
        data['source_video_sha256']='f'*64
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Identical source video'):
        module.build([a,b],tmp_path/'new_corpus')


def test_video_digest_included_in_imported_corpus(tmp_path):
    a=pack(tmp_path,'TEST',group='vn_routeA')
    source=Path(a['pack'])/'review_manifest.json'
    data=json.loads(source.read_text(encoding='utf-8'))
    data['source_video_sha256']='a'*64
    source.write_text(json.dumps(data))
    module.build([a],tmp_path/'corpus')
    record=json.loads((tmp_path/'corpus'/'frames.jsonl').read_text().splitlines()[0])
    assert record['source_video_sha256']=='a'*64
    assert 'plate_text_verified' in record['signs'][0]


def test_field_corpus_keeps_parent_link(tmp_path):
    a=pack(tmp_path,'TRAIN')
    root=Path(a['pack'])
    data=json.loads((root/'review_manifest.json').read_text())
    signs=data['frames'][0]['signs']
    signs[0].update(judgment='sign',label='P.130',panel_role='primary')
    signs.append({'id':'s2','label':'unknown','bbox_xyxy':[50,90,70,110],
                  'panel_role':'supplementary','judgment':'sign','parent_id':'s1'})
    signs[0]['id']='s1'
    (root/'review_manifest.json').write_text(json.dumps(data))
    (root/'reviewed.json').write_text(json.dumps(data))
    module.build([a],tmp_path/'dataset')
    row=json.loads((tmp_path/'dataset'/'frames.jsonl').read_text().splitlines()[0])
    assert {s['id']:s['parent_id'] for s in row['signs']}=={'s1':None,'s2':'s1'}
