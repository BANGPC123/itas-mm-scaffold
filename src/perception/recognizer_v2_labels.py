"""Training and evaluation boxes plus the class list for recognizer v2.

Sources: released V1.2 semantic labels (QCVN codes) and the AI-reviewed
ontology extension (two agreeing AI passes). Neither is human ground truth.
Boxes marked unreadable are never turned into background crops here.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd

BACKGROUND = 'unknown'
NOT_A_CODE = ('not_qcvn_sign', 'uncertain')
OUT_OF_ONTOLOGY = 'EXCLUDED_OUT_OF_ONTOLOGY_V1_2'
UNREADABLE = 'EXCLUDED_UNREADABLE_V1_2'
MIN_SIDE = 16
MAIN_MIN_TRAIN = 30
MAIN_MIN_VAL = 5
CODE_PATTERN = re.compile(r'^(DP|[PRWIS])\.\d{3}[a-z]?$')
BOX_COLUMNS = ['canonical_box_id', 'image_path', 'bbox_xyxy', 'split', 'code', 'short_side_px', 'source']


def read_extension_labels(path):
    """Agreed AI-reviewed extension labels; rejects records that claim human review."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('Extension label file is empty')
    ids = [r['canonical_box_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate canonical_box_id in extension labels')
    for row in rows:
        if row.get('provenance') != 'ai_reviewed':
            raise ValueError(f"{row['canonical_box_id']}: provenance must be ai_reviewed")
        if not str(row.get('reviewer', '')).startswith('ai-review'):
            raise ValueError(f"{row['canonical_box_id']}: reviewer must be an AI reviewer")
        agreed = bool(row.get('agreed'))
        code = row.get('code')
        if agreed != (code is not None):
            raise ValueError(f"{row['canonical_box_id']}: agreed rows need a code, others none")
        if code is not None:
            if not CODE_PATTERN.match(code) or row['pass_a'] != code or row['pass_b'] != code:
                raise ValueError(f"{row['canonical_box_id']}: invalid or non-agreeing code {code!r}")
    frame = pd.DataFrame(rows)
    return frame[frame.agreed].reset_index(drop=True)


def read_semantic_labels(path):
    frame = pd.read_csv(path, keep_default_na=False)
    frame['short_side_px'] = pd.to_numeric(frame['short_side_px'])
    return frame


def build_boxes(labels, extension):
    """One row per labelled box: released V1.2 codes plus agreed extension codes."""
    released = labels[labels.semantic_final_disposition.str.startswith('RELEASED')].copy()
    released['code'] = released['canonical_sign_code']
    if (released.code == '').any():
        raise ValueError('Released box without a code')
    released['source'] = 'v1_2_released'
    indexed = labels.set_index('canonical_box_id')
    missing = set(extension.canonical_box_id) - set(indexed.index)
    if missing:
        raise ValueError(f'{len(missing)} extension boxes are not in the V1.2 labels')
    joined = indexed.loc[extension.canonical_box_id]
    if (joined.semantic_final_disposition != OUT_OF_ONTOLOGY).any():
        raise ValueError('Extension labels must only cover out-of-ontology boxes')
    if (joined.split.to_numpy() != extension.split.to_numpy()).any():
        raise ValueError('Extension split differs from the V1.2 split')
    extended = joined.reset_index()
    extended['code'] = extension.code.to_numpy()
    extended['source'] = 'ai_ontology_extension'
    boxes = pd.concat([released[BOX_COLUMNS], extended[BOX_COLUMNS]], ignore_index=True)
    return boxes.sort_values('canonical_box_id').reset_index(drop=True)


def natural_key(code):
    match = re.match(r'(DP|[PRWIS])\.(\d+)([a-z]*)$', code)
    order = {'P': 0, 'DP': 1, 'R': 2, 'W': 3, 'I': 4, 'S': 5}
    return (order[match.group(1)], int(match.group(2)), match.group(3)) if match else (9, 0, code)


def count_table(boxes, min_side=MIN_SIDE):
    """Boxes per code and split: all sizes and at least min_side pixels (short side)."""
    big = boxes[boxes.short_side_px >= min_side]
    table = pd.DataFrame(index=sorted(boxes.code.unique(), key=natural_key))
    for split in ('TRAIN', 'VAL', 'TEST'):
        table[f'{split.lower()}_all'] = boxes[boxes.split == split].groupby('code').size().reindex(table.index, fill_value=0)
        table[f'{split.lower()}_ge{min_side}'] = big[big.split == split].groupby('code').size().reindex(table.index, fill_value=0)
    return table


def build_class_list(boxes, old_classes, min_side=MIN_SIDE, main_min_train=MAIN_MIN_TRAIN, main_min_val=MAIN_MIN_VAL):
    """Main, low-support and untrained codes; output order keeps old classes first and background last."""
    counts = count_table(boxes, min_side)
    train = counts[f'train_ge{min_side}']
    val = counts[f'val_ge{min_side}']
    main = [c for c in counts.index if train[c] >= main_min_train and val[c] >= main_min_val]
    low = [c for c in counts.index if train[c] >= 1 and c not in main]
    untrained = [c for c in counts.index if train[c] == 0]
    old = [c for c in old_classes if c != BACKGROUND]
    trained = set(main) | set(low)
    kept_old = [c for c in old if c in trained]
    new = sorted([c for c in trained if c not in old], key=natural_key)
    classes = kept_old + new + [BACKGROUND]
    return dict(main=main, low_support=low, untrained=untrained, classes=classes,
                old_classes=old, dropped_old=[c for c in old if c not in trained], counts=counts)
