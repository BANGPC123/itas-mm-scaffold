from importlib import import_module


def test_clip_frame_scores_detector_and_pseudolabel_agreement():
    score_frame = import_module("scripts.benchmark_clip_test_20261028").score_frame
    source = {"signs": [{"id": "a", "bbox_xyxy": [0, 0, 10, 10]},
                        {"id": "b", "bbox_xyxy": [20, 20, 30, 30]}]}
    review = {"lane_status": "annotated", "signs": [
        {"id": "a", "judgment": "sign", "label": "P.130"},
        {"id": "b", "judgment": "uncertain", "label": "unknown"}]}
    predictions = [{"bbox_xyxy": [0, 0, 10, 10], "label": "P.130"},
                   {"bbox_xyxy": [40, 40, 50, 50], "label": "unknown"}]
    score = score_frame(source, review, predictions)
    assert (score["tp"], score["fp"], score["fn"]) == (1, 1, 1)
    assert (score["code_correct"], score["code_matched"], score["code_eligible"]) == (1, 1, 1)
    assert score["lane_proposed"] is True


def test_clip_bootstrap_samples_whole_groups_and_is_reproducible():
    bootstrap = import_module("scripts.benchmark_clip_test_20261028").bootstrap
    groups = {"a": {"baseline": {"tp": 1, "fp": 0, "fn": 1}, "candidate": {"tp": 2, "fp": 0, "fn": 0}},
              "b": {"baseline": {"tp": 1, "fp": 0, "fn": 1}, "candidate": {"tp": 2, "fp": 0, "fn": 0}}}
    result = bootstrap(groups, iterations=100, seed=391)
    assert result == bootstrap(groups, iterations=100, seed=391)
    assert result["recall_delta_95ci"] == [0.5, 0.5]
