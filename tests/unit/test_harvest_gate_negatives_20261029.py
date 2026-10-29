from scripts.harvest_gate_negatives_20261029 import (
    balanced_sample, group_split, iou, overlaps_labeled, segment_group, temporal_thin)


def test_iou_and_overlap_rules():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert overlaps_labeled([0, 0, 10, 10], [[5, 0, 15, 10]])            # IoU 1/3
    assert not overlaps_labeled([0, 0, 10, 10], [[50, 50, 60, 60]])
    # small box inside a big labelled box: IoU is tiny but 100% of the detector box is covered
    assert overlaps_labeled([40, 40, 50, 50], [[0, 0, 100, 100]])
    # sliver touch: below IoU 0.1 and below 30% cover of both boxes
    assert not overlaps_labeled([0, 0, 100, 100], [[96, 0, 200, 100]])


def test_segment_group_uses_60_second_windows():
    assert segment_group("abc", 0.5) == "abc@0"
    assert segment_group("abc", 59.9) == "abc@0"
    assert segment_group("abc", 60.0) == "abc@60"
    assert segment_group("abc", 3661) == "abc@3660"


def test_temporal_thin_drops_repeats_of_the_same_box_within_a_second():
    box = [100, 100, 140, 140]
    rows = [{"video_id": "v", "t": t, "bbox_xyxy": box} for t in (1.0, 1.5, 2.0, 2.5)]
    rows.append({"video_id": "v", "t": 1.5, "bbox_xyxy": [500, 500, 540, 540]})
    kept = temporal_thin(rows, min_gap_s=1.0)
    assert [(k["t"], k["bbox_xyxy"][0]) for k in kept] == [(1.0, 100), (1.5, 500), (2.0, 100)]


def test_temporal_thin_keeps_same_box_in_other_video():
    box = [0, 0, 20, 20]
    kept = temporal_thin([{"video_id": "a", "t": 1, "bbox_xyxy": box}, {"video_id": "b", "t": 1, "bbox_xyxy": box}])
    assert len(kept) == 2


def test_balanced_sample_equal_quota_and_spillover():
    pools = {"video": list(range(100)), "train": list(range(1000, 1010))}
    chosen = balanced_sample(pools, 40, seed=5)
    assert len(chosen) == 40 and len(set(chosen)) == 40
    assert sum(item >= 1000 for item in chosen) == 10        # train pool exhausted, video fills the rest
    even = balanced_sample({"video": list(range(100)), "train": list(range(1000, 1100))}, 40, seed=5)
    assert sum(item >= 1000 for item in even) == 20


def test_balanced_sample_is_deterministic():
    pools = {"video": list(range(50)), "train": list(range(100, 150))}
    assert balanced_sample(pools, 20, seed=9) == balanced_sample(pools, 20, seed=9)
    assert balanced_sample(pools, 20, seed=9) != balanced_sample(pools, 20, seed=10)


def test_assign_splits_keeps_a_group_on_one_side_and_only_marks_negatives():
    from scripts.harvest_gate_negatives_20261029 import assign_splits
    records = []
    for index in range(40):
        records.append({"source": "video", "label": "not_a_sign", "group": f"v@{(index % 10) * 60}"})
        records.append({"source": "train", "label": "not_a_sign", "group": f"images/train/{index % 20}.jpg"})
    records.append({"source": "video", "label": "qcvn_sign:P.130", "group": "v@0"})
    assign_splits(records)
    by_group = {}
    for r in records:
        if r["label"] == "not_a_sign":
            by_group.setdefault((r["source"], r["group"]), set()).add(r["split"])
    assert all(len(sides) == 1 for sides in by_group.values())
    video = {g for (s, g), sides in by_group.items() if s == "video" and sides == {"heldout"}}
    train = {g for (s, g), sides in by_group.items() if s == "train" and sides == {"heldout"}}
    assert len(video) == 2 and len(train) == 4            # 20% of 10 video segments and 20 TRAIN images
    assert records[-1]["split"] == "unused"


def test_manifest_record_has_source_reference_but_no_pixels():
    from scripts.harvest_gate_negatives_20261029 import manifest_record
    video = manifest_record({"number": 5, "source": "video", "video_id": "abc", "t": 12.5, "frame": 375, "bbox_xyxy": [1, 2, 3, 4],
                             "group": "abc@0", "det_conf": 0.6, "pred": "P.130"}, "not_a_sign")
    assert video["video_id"] == "abc" and video["frame"] == 375 and "pred" not in video and video["provenance"] == "ai_reviewed"
    train = manifest_record({"number": 6, "source": "train", "image_path": "images/train/ff00.jpg", "bbox_xyxy": [1, 2, 3, 4],
                             "group": "images/train/ff00.jpg", "det_conf": 0.7}, "uncertain")
    assert train["image_sha256"] == "ff00" and "video_id" not in train


def test_group_split_is_group_level_and_deterministic():
    groups = [f"g{i % 10}" for i in range(100)]
    held = group_split(groups, 0.2, seed=3)
    assert len(held) == 2 and held == group_split(groups, 0.2, seed=3)
    assert group_split(["only"], 0.2, seed=3) == set()
    assert len(group_split(["a", "b"], 0.2, seed=3)) == 1
