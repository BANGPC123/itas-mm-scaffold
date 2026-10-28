from scripts.field_review_pack_20261029 import RUN_NAME, confirmed_items, merge_evidence, pick_context_frames


def frame(t, source_frame, *signs):
    return {"timestamp": t, "source_frame": source_frame, "signs": list(signs)}


def sign(track, label, source="temporal_best_crop", age=None, bbox=(10, 10, 50, 50)):
    return {"track_id": track, "label": label, "recognition_source": source, "crop_age_ms": age, "bbox_xyxy": list(bbox)}


def test_confirmed_items_use_the_best_crop_frame_and_skip_unknown():
    frames = [
        frame(10.0, 600, sign(1, "unknown", "temporal_pending", None, (10, 10, 30, 30))),
        frame(10.5, 630, sign(1, "P.131a", "temporal_best_crop", 500, (12, 11, 52, 51))),
        frame(11.0, 660, sign(1, "P.131a", "temporal_best_crop", 1000, (14, 12, 54, 52)), sign(2, "unknown", "temporal_pending", None)),
    ]
    items, lost = confirmed_items(frames)
    assert lost == []
    assert items == [{"track": 1, "prediction": "P.131a", "best_time": 10.0, "source_frame": 600,
                      "bbox_xyxy": [10, 10, 30, 30]}]


def test_confirmed_items_report_unresolved_best_frames():
    frames = [frame(5.0, 300, sign(7, "W.224", "temporal_best_crop", 5000)), frame(6.0, 360, sign(8, "W.225", "temporal_best_crop", None))]
    items, lost = confirmed_items(frames)
    assert items == [] and {u["reason"] for u in lost} == {"best frame or track unavailable", "missing crop age"}


def test_merge_evidence_reviews_shared_crops_once_and_keeps_each_prediction():
    base = {"video_id": "v", "start": 30, "track": 4, "best_time": 12.0, "source_frame": 720, "bbox_xyxy": [100, 100, 140, 140]}
    runs = [{**base, "config": "A", "prediction": "P.130"}, {**base, "config": "B", "prediction": "P.131a"},
            {**base, "config": "C", "prediction": "P.131a"},
            {**base, "config": "C", "prediction": "P.131a"},                                  # duplicate is ignored
            {**base, "track": 5, "config": "A", "prediction": "W.224"}]
    merged = merge_evidence(runs)
    assert len(merged) == 2
    first = next(m for m in merged if m["track"] == 4)
    assert first["members"] == [{"config": "A", "prediction": "P.130"}, {"config": "B", "prediction": "P.131a"},
                                {"config": "C", "prediction": "P.131a"}]


def test_merge_evidence_separates_different_best_crops():
    base = {"video_id": "v", "start": 30, "track": 4, "best_time": 12.0, "source_frame": 720, "bbox_xyxy": [100, 100, 140, 140],
            "config": "A", "prediction": "P.130"}
    other = {**base, "source_frame": 750, "best_time": 12.5, "config": "B"}
    assert len(merge_evidence([base, other])) == 2


def test_run_name_pattern():
    match = RUN_NAME.match("B_6JuxWcAf-wM_830")
    assert match["config"] == "B" and match["video"] == "6JuxWcAf-wM" and match["start"] == "830"
    assert RUN_NAME.match("run_status") is None


def test_pick_context_frames_stay_near_best_time_and_apart():
    rows = [(s, t, int(t * 30), [0, 0, s, s]) for s, t in [(20, 10.0), (30, 10.2), (40, 10.6), (45, 11.0), (50, 11.1), (60, 20.0)]]
    chosen = pick_context_frames(rows, best_time=10.0, near_s=3.0, spacing_s=0.45)
    times = [c[1] for c in chosen]
    assert 20.0 not in times and 10.2 not in times          # too far / too close to the best crop
    assert times == sorted(times)
    assert all(b - a >= 0.45 for a, b in zip(times, times[1:]))
