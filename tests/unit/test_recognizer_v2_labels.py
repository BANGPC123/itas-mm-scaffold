import json
import pandas as pd
import pytest
from src.perception.recognizer_v2_labels import (BACKGROUND, build_boxes, build_class_list,
                                                 count_table, natural_key, read_extension_labels)


def extension_row(box_id, code, split='TRAIN', **overrides):
    row = dict(canonical_box_id=box_id, split=split, code=code, name_vi='x', pass_a=code, pass_b=code,
               agreed=code is not None, reviewer='ai-review (Claude Code)', provenance='ai_reviewed', qcvn_ref=None)
    row.update(overrides)
    return row


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def label_frame(rows):
    return pd.DataFrame([dict(canonical_box_id=f'box:{i}', image_path=f'images/{i}.jpg', bbox_xyxy='[0, 0, 40, 40]',
                              split=split, canonical_sign_code=code, short_side_px=side, semantic_final_disposition=disp)
                         for i, (split, code, side, disp) in enumerate(rows)])


def test_extension_reader_keeps_only_agreed_ai_reviewed_rows(tmp_path):
    path = tmp_path / 'ext.jsonl'
    disagreed = extension_row('box:b', None, pass_a='P.105', pass_b='uncertain')
    write_rows(path, [extension_row('box:a', 'P.105'), disagreed])
    frame = read_extension_labels(path)
    assert list(frame.canonical_box_id) == ['box:a']


@pytest.mark.parametrize('overrides', [dict(provenance='human_reviewed'), dict(reviewer='Kiet (owner)'),
                                       dict(pass_b='P.106a'), dict(pass_a='P105', pass_b='P105', code_override='P105')])
def test_extension_reader_rejects_unsupported_provenance_or_codes(tmp_path, overrides):
    path = tmp_path / 'ext.jsonl'
    overrides = dict(overrides)
    code = overrides.pop('code_override', 'P.105')
    write_rows(path, [extension_row('box:a', code, **overrides)])
    with pytest.raises(ValueError):
        read_extension_labels(path)


def test_build_boxes_adds_extension_to_released_but_never_unreadable():
    labels = label_frame([('TRAIN', 'P.127', 50, 'RELEASED_AI_VERIFIED_HIGH'), ('TRAIN', '', 40, 'EXCLUDED_OUT_OF_ONTOLOGY_V1_2'),
                          ('VAL', '', 30, 'EXCLUDED_UNREADABLE_V1_2'), ('TEST', 'W.203c', 20, 'RELEASED_AI_LABELED_V1_2')])
    extension = pd.DataFrame([extension_row('box:1', 'P.105')])
    boxes = build_boxes(labels, extension)
    assert sorted(boxes.canonical_box_id) == ['box:0', 'box:1', 'box:3']
    assert dict(zip(boxes.canonical_box_id, boxes.source))['box:1'] == 'ai_ontology_extension'


def test_build_boxes_refuses_extension_on_released_or_wrong_split():
    labels = label_frame([('TRAIN', 'P.127', 50, 'RELEASED_AI_VERIFIED_HIGH'), ('TRAIN', '', 40, 'EXCLUDED_OUT_OF_ONTOLOGY_V1_2')])
    with pytest.raises(ValueError):
        build_boxes(labels, pd.DataFrame([extension_row('box:0', 'P.105')]))
    with pytest.raises(ValueError):
        build_boxes(labels, pd.DataFrame([extension_row('box:1', 'P.105', split='VAL')]))


def boxes_for(code, train, val, test=0, side=20):
    rows = [('TRAIN', code, side)] * train + [('VAL', code, side)] * val + [('TEST', code, side)] * test
    return [dict(canonical_box_id=f'{code}-{i}', image_path='x', bbox_xyxy='[0,0,1,1]', split=s, code=code,
                 short_side_px=float(side_), source='v1_2_released') for i, (s, code, side_) in enumerate(rows)]


def test_class_list_separates_main_low_support_and_untrained_by_crops_of_16_px_or_more():
    rows = boxes_for('P.127', 30, 5) + boxes_for('P.105', 29, 9) + boxes_for('R.420', 1, 0) + boxes_for('W.999', 0, 3, 2)
    rows += boxes_for('DP.134', 40, 6, side=10)   # too small to count
    boxes = pd.DataFrame(rows)
    result = build_class_list(boxes, ['P.127', 'DP.134', BACKGROUND])
    assert result['main'] == ['P.127']
    assert result['low_support'] == ['P.105', 'R.420']
    assert result['untrained'] == ['DP.134', 'W.999'] or sorted(result['untrained']) == ['DP.134', 'W.999']
    assert result['dropped_old'] == ['DP.134']
    # old codes keep their relative order, new ones follow, background is last
    assert result['classes'] == ['P.127', 'P.105', 'R.420', BACKGROUND]


def test_count_table_reports_all_sizes_and_size_filtered_counts():
    boxes = pd.DataFrame(boxes_for('P.127', 2, 1, 1, side=12) + boxes_for('P.127', 3, 0, 0, side=40))
    table = count_table(boxes)
    assert table.loc['P.127', 'train_all'] == 5 and table.loc['P.127', 'train_ge16'] == 3
    assert table.loc['P.127', 'val_all'] == 1 and table.loc['P.127', 'val_ge16'] == 0


def test_natural_code_order():
    codes = ['W.203a', 'P.124e', 'DP.135', 'I.442', 'R.301c', 'P.103b']
    assert sorted(codes, key=natural_key) == ['P.103b', 'P.124e', 'DP.135', 'R.301c', 'W.203a', 'I.442']
