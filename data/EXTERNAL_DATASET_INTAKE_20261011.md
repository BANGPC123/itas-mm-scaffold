# ITAS-MM — External Dataset Intake Candidates (2026-10-11)

## Decision
Do NOT immediately download tens of gigabytes. Project already includes 5 traffic-sign Vietnam sources (Kaggle/Roboflow) and canonical 30,558 labeled boxes. One existing source has contradictory license declarations; dataset normalization and group-level deduplication matter more than gross size.

New targeted dataset actually added: 21 de-duplicated-by-SHA development hard-negative crops under data/audit/ai_removed_boxes_hardneg_20261011; their source is ONE AI-reviewed Vietnam driving video, no human verified GT. Split TRAIN_CANDIDATE_ONLY; do not use them for independent VAL or TEST. Script scripts/prepare_development_negatives_20261011.py reconstructs them and scripts/audit_hardneg_provenance_20261011.py audits each file.

## External candidates (NOT downloaded or auto-licensed)
1. BDD100K (https://github.com/bdd100k/bdd100k) — lane-marking and drivable-area labels with rain/night/evolving lighting. Data license research/education/not-for-profit; BSD-3-Clause toolkit license does NOT license imagery for commercial use. Can only import after acceptance of source data terms, video/image labels downloaded from the official provider and source-level split. Domain transfer USA -> Vietnam uncertain.
2. Mapillary Vistas v2 (https://www.mapillary.com/dataset/vistas) — dense road scene and traffic sign/road markings, CC BY-NC-SA 4.0; download may require accepting provider terms; commercial usage prohibited under NC terms. Domain generalization study only; exact class conversion requires review.
3. TT100K (https://cg.cs.tsinghua.edu.cn/traffic-sign/) — small signs in outdoor scenes, CC BY-NC 2.0; 18 GB in Ultralytics full data, likely more storage than justified until small-sign recall review. Chinese signs/labels do not directly equal Vietnamese QCVN codes.
4. VIL-100 video lane instance dataset (https://github.com/yujun0-0/MMA-Net), lane tracking/occlusion training candidate. VIL is *Video Instance Lane*, NOT a Vietnam-specific road dataset. Confirm terms and authorized download before adding.

## Quality and promotion gates
- Acquire official archive (never redistributed mirror without license confirmation), immutable URL/version/license and SHA256, count images, verify image decoding and annotation checks, mark any uncertain/ambiguous labels UNKNOWN not forced positive.
- Group by source photographer/video/route/time. Cross-split pHash duplicates and neighboring frames forbidden. Only TRAIN/VAL candidates; do not touch canonical TEST to choose experiments.
- Prioritize **hard negatives** (dashboard gauge, false-color brand logo, car lamps, advertisement panels, street lamps) and **Vietnam lane overlays** (dense bikes, parking occlusions, intersections, rain/night). Existing 21 negatives have no independent field specificity.
- Minimum release evidence: known-sign VAL macro-F1 and classwise recall not worse, separate human-verified source-disjoint negative false-positive rate decreases, lane F1 on annotated Vietnam roads increases, inference VRAM/FPS budget maintained.
- Missing data means don't auto-promote. Keep audio for uncertain signs disabled.
