import json

from scripts.field_proxy_pack_20261029 import proxy_times, readable_annotations, recall_table


def test_proxy_times_are_one_per_five_seconds_and_inside_the_segment():
    times = proxy_times(30)
    assert len(times) == 12 and times[0] == 32.5 and times[-1] == 87.5
    assert all(30 + 2 <= t <= 30 + 60 - 2 for t in times)               # a +-2 s window never leaves the segment
    assert all(b - a == 5 for a, b in zip(times, times[1:]))


def test_only_coded_signs_of_at_least_24_px_count():
    items = [{"number": 1, "video_id": "v", "start": 30, "t": 32.5, "bbox_xyxy": [0, 0, 30, 40]},
             {"number": 2, "video_id": "v", "start": 30, "t": 32.5, "bbox_xyxy": [0, 0, 23, 40]},       # too small
             {"number": 3, "video_id": "v", "start": 30, "t": 37.5, "bbox_xyxy": [0, 0, 40, 40]},
             {"number": 4, "video_id": "v", "start": 30, "t": 42.5, "bbox_xyxy": [0, 0, 40, 40]},
             {"number": 5, "video_id": "v", "start": 30, "t": 47.5, "bbox_xyxy": [0, 0, 40, 40]}]
    labels = {"1": "P.131a", "2": "P.130", "3": "not_sign", "4": "unreadable"}
    kept = readable_annotations(items, labels)
    assert [k["number"] for k in kept] == [1]
    assert kept[0]["code"] == "P.131a"


def test_recall_table_per_config(tmp_path):
    def write(config, label):
        folder = tmp_path / f"{config}_v_30" / "segment_30"
        folder.mkdir(parents=True)
        frames = [{"timestamp": 32.0 + 0.5 * i, "source_frame": i,
                   "signs": [{"track_id": 1, "bbox_xyxy": [0, 0, 30, 40], "label": label}]} for i in range(8)]
        (folder / "predictions.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames), encoding="utf-8")
    write("A", "P.130")
    write("B", "P.131a")
    items = [{"number": 1, "video_id": "v", "start": 30, "t": 33.0, "bbox_xyxy": [0, 0, 30, 40]}]
    table = recall_table(items, {"1": "P.131a"}, lambda c, v, s: tmp_path / f"{c}_{v}_{s}" / f"segment_{s}" / "predictions.jsonl", ["A", "B"])
    assert table["A"]["recall"] == 0.0 and table["B"]["recall"] == 1.0
