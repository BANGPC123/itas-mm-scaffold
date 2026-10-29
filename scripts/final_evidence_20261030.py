"""Issue #2: seeded blind second review, sealed labels and descriptive statistics.

Preparation reads source metadata inside this process and exposes only shuffled
crop numbers and original sizes. The reviewer must not open private_mapping.json
until all labels have been written and their SHA256 recorded. Source labels and
predictions are never printed by prepare. TEST is used only for reporting.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output/kiet_wp_20261029"
OUT = ROOT / "output/kiet_wp_20261030/second_review"
SEED = 20261030
PROVENANCE = "ai_reviewed_codex_sol"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize(label):
    return {"not_qcvn_sign": "not_a_sign", "not_sign": "not_a_sign",
            "unreadable": "uncertain"}.get(label, label)


def blind_order(rows, seed=SEED):
    ordered = sorted((dict(r) for r in rows), key=lambda r: (r["source"], r["original_key"]))
    random.Random(seed).shuffle(ordered)
    for i, row in enumerate(ordered, 1):
        row["item_id"] = f"{i:04d}"
    return ordered


def public_blind_items(rows):
    return [{"item_id": r["item_id"], "short_side_px": r["short_side_px"]} for r in rows]


def stratified_sample(rows, total=150, per_code=10, seed=SEED):
    """Code-balanced round robin; alternate TRAIN/VAL within each code, cap by code."""
    rng = random.Random(seed)
    groups = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda r: r["canonical_box_id"]):
        if row["split"] in ("TRAIN", "VAL"):
            groups[row["code"]][row["split"]].append(row)
    queues = {}
    for code, splits in sorted(groups.items()):
        for entries in splits.values():
            rng.shuffle(entries)
        queue = []
        while any(splits.values()):
            for split in ("TRAIN", "VAL"):
                if splits[split]:
                    queue.append(splits[split].pop())
        queues[code] = queue[:per_code]
    codes = sorted(queues)
    rng.shuffle(codes)
    selected = []
    while len(selected) < total:
        before = len(selected)
        for code in codes:
            if queues[code] and len(selected) < total:
                selected.append(queues[code].pop(0))
        if len(selected) == before:
            raise ValueError("Not enough TRAIN/VAL rows under the per-code cap")
    return selected


def agreement(left, right):
    if len(left) != len(right):
        raise ValueError("Reviewer arrays differ in length")
    n = len(left)
    if not n:
        return {"n": 0, "agreement": None, "kappa": None, "disagreements": []}
    a, b = Counter(left), Counter(right)
    observed = sum(x == y for x, y in zip(left, right)) / n
    expected = sum(a[k] * b[k] for k in a.keys() | b.keys()) / n**2
    pairs = Counter((x, y) for x, y in zip(left, right) if x != y)
    return {"n": n, "agreement": observed,
            "kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
            "disagreements": [{"claude": x, "sol": y, "n": c}
                              for (x, y), c in pairs.most_common()]}


def precision_summary(rows, draws=2000):
    from scripts.field_video_eval_20261029 import bootstrap_precision, decidable, precision, tally, verdict
    counted = [(r["cluster"], verdict(r["prediction"], normalize(r["label"]))) for r in rows]
    counts = tally(v for _, v in counted)
    lo, hi = bootstrap_precision(counted, n_boot=draws, seed=SEED)
    return {**counts, "decidable": decidable(counts), "precision": precision(counts),
            "ci95": [lo, hi], "clusters": len({c for c, _ in counted})}


def uniform_frame_times(duration, count):
    if duration <= 0 or count < 1:
        raise ValueError("Positive duration and frame count required")
    return [(i + .5) * duration / count for i in range(count)]


def lane_prepare(out):
    import cv2
    import numpy as np
    import torch
    from scripts.field_video_eval_20261029 import read_sources
    from src.perception.lane_detector import LaneDetector
    from src.utils.config import load_config
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    sources = [r for r in read_sources(ROOT / "data/manifests/field_video_20261029/sources.csv") if r["role"] == "eval"]
    if len(sources) != 3:
        raise ValueError("Expected exactly three Hanoi EVAL videos")
    if out.exists():
        raise FileExistsError(out)
    (out / "overlays").mkdir(parents=True)
    cfg = load_config("perception")
    model = LaneDetector(cfg)
    if model.backend != "ufld_v2" or model.learned is None:
        raise RuntimeError("Default lane model unavailable")
    rows = []
    for kind in ("day", "night"):
        group = [r for r in sources if r["day_night"] == kind]
        if not group or 30 % len(group):
            raise ValueError("Cannot balance 30 frames over this lighting group")
        for source in group:
            video = BASE / "videos" / (source["video_id"] + ".mp4")
            cap = cv2.VideoCapture(str(video))
            fps, total = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps <= 0 or total <= 0:
                raise RuntimeError(f"Cannot read video {video}")
            for timestamp in uniform_frame_times(total / fps, 30 // len(group)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp * fps))
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode {video} at {timestamp}")
                lane = model.detect(frame)
                display = frame.copy()
                for poly in lane.polylines:
                    if len(poly) >= 2:
                        cv2.polylines(display, [np.asarray(poly, dtype=np.int32)], False, (0, 255, 0), 5)
                for line in lane.lane_lines:
                    cv2.line(display, tuple(line[:2]), tuple(line[2:]), (0, 255, 0), 5)
                number = len(rows) + 1
                display = cv2.resize(display, (960, 540))
                cv2.putText(display, str(number), (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imwrite(str(out / "overlays" / f"{number:03d}.jpg"), display)
                rows.append({"item_id": f"{number:03d}", "video_id": source["video_id"],
                             "timestamp": timestamp, "source_frame": round(timestamp * fps),
                             "day_night": kind, "backend": lane.backend, "polylines": lane.polylines,
                             "source_sha256_recorded": source["sha256"]})
            cap.release()
    write_json(out / "frames.json", rows)
    write_json(out / "protocol.json", {"n": len(rows), "day": 30, "night": 30,
               "sampling": "Equally spaced midpoint frames over each complete EVAL video, 15 from each day video and 30 from night video.",
               "lane_model_sha256": hashlib.sha256((ROOT / cfg["lane_detector"]["weights_path"]).read_bytes()).hexdigest(),
               "config_sha256": hashlib.sha256((ROOT / "configs/perception.yaml").read_bytes()).hexdigest(),
               "scope": "LaneDetector(load_config('perception')).detect, the pipeline default with lane_visual_roi=False; no temporal smoothing or optional camera-aware display fallback.",
               "provenance": PROVENANCE, "human_gt": False})
    print(json.dumps({"frames": len(rows), "backend": model.backend}))


def lane_summarize(out):
    frames = read_json(out / "frames.json")
    ratings = [json.loads(line) for line in (out / "ratings.jsonl").read_text(encoding="utf-8-sig").splitlines() if line]
    indexed = {r["item_id"]: r for r in ratings}
    if len(frames) != 60 or len(indexed) != len(ratings) or set(indexed) != {r["item_id"] for r in frames}:
        raise ValueError("Expected exactly one rating for each of the 60 frames")
    labels = ("good", "partial", "fail", "no_visible_lane")
    manifest = []
    for frame in frames:
        rating = indexed[frame["item_id"]]
        if rating["label"] not in labels or not rating["description"]:
            raise ValueError("Invalid lane rating")
        manifest.append({**{k: frame[k] for k in ("item_id", "video_id", "timestamp", "source_frame", "day_night", "backend", "source_sha256_recorded")},
                         "label": rating["label"], "description": rating["description"],
                         "provenance": PROVENANCE, "human_gt": False})
    destination = ROOT / "data/manifests/field_video_20261029/lane_proxy_20261030.jsonl"
    content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest)
    if destination.exists() and destination.read_text(encoding="utf-8") != content:
        raise FileExistsError(destination)
    destination.write_text(content, encoding="utf-8", newline="\n")
    groups = {}
    for kind in ("day", "night", "all"):
        selected = [r for r in manifest if kind == "all" or r["day_night"] == kind]
        counts = {label: sum(r["label"] == label for r in selected) for label in labels}
        n = len(selected)
        visible = n - counts["no_visible_lane"]
        groups[kind] = {"n": n, "counts": counts, "fraction_all": {k: v/n for k,v in counts.items()},
                        "visible_n": visible, "fraction_visible": {k: counts[k]/visible if visible else None for k in labels[:3]}}
    result = {"status": "AI_QUALITATIVE_PROXY_NOT_LANE_F1", "manifest": str(destination.relative_to(ROOT)).replace("\\", "/"),
              "manifest_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
              "protocol": read_json(out / "protocol.json"), "groups": groups}
    write_json(out / "results.json", result)
    print(json.dumps(groups))


def verify_seal(path, digest, expected_ids):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError("Blind labels differ from sealed SHA256")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    ids = [r["item_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError("Missing, extra or duplicate blind labels")
    for row in rows:
        if not row.get("description") or not 0 <= row["confidence"] <= 1:
            raise ValueError("Missing observation or invalid confidence")
        code = row["label_sol"]
        if code not in ("not_a_sign", "uncertain") and not re.fullmatch(r"(?:DP|[PRWIS])\.\d{3}[a-z]?", code):
            raise ValueError("Invalid QCVN label")
        if code not in ("not_a_sign", "uncertain") and row["confidence"] < .8:
            raise ValueError("QCVN code confidence below 0.8")
    return rows


def prepare(out=OUT):
    import pandas as pd
    from PIL import Image
    from scripts.field_review_pack_20261029 import key_string
    if out.exists():
        raise FileExistsError(out)
    records = []
    for k in range(1, 9):
        pack = BASE / f"review/pack_{k:02d}"
        for row in read_json(pack / "private_mapping.json")["items"]:
            x1, y1, x2, y2 = row["bbox_xyxy"]
            records.append({"source": "a", "original_key": f"pack_{k:02d}:{row['number']}",
                            "evidence_key": key_string(row), "record": row,
                            "crop_path": str(pack / "crops" / f"{row['number']:04d}.png"),
                            "short_side_px": min(x2-x1, y2-y1)})
    v0 = ROOT / "output/kiet_wp_20261028/codex_crosscheck_20261029/video_blind_review"
    for row in read_json(v0 / "private_mapping.json")["items"]:
        crop = v0 / "crops" / f"{row['number']:04d}.png"
        if not crop.is_file():
            raise FileNotFoundError(crop)
        x1, y1, x2, y2 = row["bbox_xyxy"]
        records.append({"source": "b", "original_key": str(row["number"]), "record": row,
                        "crop_path": str(crop), "short_side_px": min(x2-x1, y2-y1)})
    ext = [json.loads(l) for l in (ROOT / "data/manifests/dataset_v2/ai_ontology_extension_labels.jsonl").read_text(encoding="utf-8").splitlines()]
    ext = [r for r in ext if r["agreed"]]
    chosen_test = [r for r in ext if r["split"] == "TEST"]
    if len(chosen_test) != 303:
        raise ValueError(f"Expected 303 agreed TEST extension rows, found {len(chosen_test)}")
    boxes = pd.read_csv(ROOT / "output/kiet_wp_20261028/recognizer_v2/boxes.csv", keep_default_na=False).set_index("canonical_box_id")
    for source, chosen in (("c", chosen_test), ("d", stratified_sample(ext))):
        for row in chosen:
            box = boxes.loc[row["canonical_box_id"]].to_dict()
            records.append({"source": source, "original_key": row["canonical_box_id"],
                            "extension": row, "record": box, "short_side_px": float(box["short_side_px"])})
    for key in read_json(BASE / "human_check/_key_human_check.json"):
        if key["attention_check"]:
            continue
        pack = BASE / "review" / key["pack"]
        records.append({"source": "e", "original_key": str(key["id"]), "owner_key": key,
                        "crop_path": str(pack / "crops" / f"{int(key['number']):04d}.png"),
                        "short_side_px": None})
    if sum(r["source"] == "e" for r in records) != 40:
        raise ValueError("Expected 40 non-attention owner spot-checks")
    records = blind_order(records)
    (out / "crops").mkdir(parents=True)
    loc = ROOT / "data/processed/canonical/itas_mm_v1/localization"
    for row in records:
        destination = out / "crops" / f"{row['item_id']}.png"
        if "crop_path" in row:
            shutil.copyfile(row["crop_path"], destination)
            if row["short_side_px"] is None:
                owner = row["owner_key"]
                source_row = next(r for r in records if r["source"] == "a" and
                                  r["original_key"] == f"{owner['pack']}:{int(owner['number'])}")
                row["short_side_px"] = source_row["short_side_px"]
        else:
            box = row["record"]
            with Image.open(loc / box["image_path"]) as image:
                x1, y1, x2, y2 = json.loads(box["bbox_xyxy"])
                dx, dy = (x2-x1)*.15, (y2-y1)*.15
                bounds = (max(0, math.floor(x1-dx)), max(0, math.floor(y1-dy)),
                          min(image.width, math.ceil(x2+dx)), min(image.height, math.ceil(y2+dy)))
                image.crop(bounds).convert("RGB").save(destination)
    write_json(out / "private_mapping.json", {"seed": SEED, "items": records})
    write_json(out / "public_items.json", public_blind_items(records))
    write_json(out / "preparation.json", {"seed": SEED, "n": len(records),
               "counts": dict(Counter(r["source"] for r in records)), "margin_extension": .15,
               "private_mapping_sha256": hashlib.sha256((out / "private_mapping.json").read_bytes()).hexdigest()})
    print(json.dumps({"n": len(records), "counts": dict(Counter(r["source"] for r in records))}))


def seal(out=OUT):
    path = out / "blind_labels.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    public = read_json(out / "public_items.json")
    rows = verify_seal(path, digest, [r["item_id"] for r in public])
    sizes = {r["item_id"]: r["short_side_px"] for r in public}
    for row in rows:
        if sizes[row["item_id"]] < 16 and row["label_sol"] not in ("uncertain", "not_a_sign"):
            raise ValueError("Readable code assigned below native 16 px")
    write_json(out / "seal.json", {"sha256": digest, "n": len(rows), "provenance": PROVENANCE,
                                  "label_counts": dict(Counter(r["label_sol"] for r in rows))})
    print(json.dumps(read_json(out / "seal.json")))


def record(out=OUT):
    """Append the reviewer's item|code|confidence|observation lines, before unlocking."""
    path = out / "blind_labels.jsonl"
    sizes = {r["item_id"]: r["short_side_px"] for r in read_json(out / "public_items.json")}
    existing = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    seen = {r["item_id"] for r in existing}
    descriptions = {r["description"] for r in existing}
    additions = []
    for line in sys.stdin.read().strip().splitlines():
        item, label, confidence, description = line.split("|", 3)
        item = item.zfill(4)
        if item in seen or item not in sizes or description in descriptions:
            raise ValueError("Duplicate or unknown item/observation")
        if sizes[item] < 16 and label not in ("not_a_sign", "uncertain"):
            raise ValueError("Code below native-size floor")
        row = {"item_id": item, "label_sol": label, "confidence": float(confidence),
               "description": description, "provenance": PROVENANCE}
        if label not in ("not_a_sign", "uncertain") and row["confidence"] < .8:
            raise ValueError("Code confidence below 0.8")
        seen.add(item)
        descriptions.add(description)
        additions.append(row)
    with path.open("a", encoding="utf-8") as stream:
        for row in additions:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps({"recorded": len(additions), "total": len(seen), "expected": len(sizes)}))


def analyze(out=OUT):
    import csv
    import pandas as pd
    from scripts.field_video_eval_20261029 import decide_recommendations
    from scripts.field_review_pack_20261029 import collect_runs, key_string, merge_evidence
    sealed = read_json(out / "seal.json")
    public = read_json(out / "public_items.json")
    labels = {r["item_id"]: r for r in verify_seal(out / "blind_labels.jsonl", sealed["sha256"],
                                                [r["item_id"] for r in public])}
    mapping = read_json(out / "private_mapping.json")
    prepared = read_json(out / "preparation.json")
    if hashlib.sha256((out / "private_mapping.json").read_bytes()).hexdigest() != prepared["private_mapping_sha256"]:
        raise ValueError("Private mapping changed since preparation")
    store = read_json(BASE / "review/label_store.json")
    v0_labels = read_json(BASE / "v0_review/blind_labels.json")["labels"]
    owner = read_json(BASE / "human_check/owner_answers_20261029.json")["answers"]
    current = {key_string(r): r for r in merge_evidence(collect_runs(BASE / "field_eval")[0])}
    pack_keys = {r["evidence_key"] for r in mapping["items"] if r["source"] == "a"}
    if set(current) != pack_keys:
        raise ValueError("Field runs contain evidence outside the completed blind pack")
    gate_identity = {}
    for gate, base in (("D", "C"), ("E", "B")):
        signatures = {c: {(key, member["prediction"]) for key, item in current.items()
                          for member in item["members"] if member["config"] == c} for c in (gate, base)}
        gate_identity[gate] = {"base": base, "same_emitted_evidence_and_predictions": signatures[gate] == signatures[base],
                               "gate_n": len(signatures[gate]), "base_n": len(signatures[base])}
    manifest, field, owner_rows = [], [], []
    for item in mapping["items"]:
        row = labels[item["item_id"]]
        source = item["source"]
        if source == "a":
            claude = store[item["evidence_key"]]
        elif source == "b":
            claude = v0_labels[item["original_key"]]
        elif source in ("c", "d"):
            claude = item["extension"]["code"]
        else:
            claude = item["owner_key"]["ai_label"]
        claude = normalize(claude)
        merged = {**row, "source": source, "original_key": item["original_key"],
                  "short_side_px": item["short_side_px"], "claude_label": claude,
                  "claude_provenance": "ai_reviewed", "blind_labels_sha256": sealed["sha256"]}
        manifest.append(merged)
        if source == "e":
            answer = owner.get(item["original_key"], "unsure")
            sol = row["label_sol"]
            owner_rows.append({"item_id": item["item_id"], "owner_id": item["original_key"],
                               "owner_answer": answer, "shown": item["owner_key"]["shown"],
                               "sol_label": sol, "sol_accepts_shown": sol == normalize(item["owner_key"]["shown"])
                               if sol != "uncertain" else None})
        if source in ("a", "b"):
            record = item["record"]
            # Later configurations reused already reviewed evidence without rebuilding older packs.
            members = current[item["evidence_key"]]["members"] if source == "a" else [record]
            for member in members:
                field.append({"item_id": item["item_id"], "config": member["config"],
                              "domain": "hanoi" if source == "a" else "v0",
                              "cluster": f"{record['video_id']}@{record['start']}" if source == "a" else f"V0@{record['segment']}",
                              "prediction": member["prediction"], "claude": claude, "sol": row["label_sol"]})
    manifest_path = ROOT / "data/manifests/dataset_v2/ai_second_review_sol_20261030.jsonl"
    manifest_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest)
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != manifest_text:
        raise ValueError("Existing second-review manifest differs")
    if not manifest_path.exists():
        manifest_path.write_text(manifest_text, encoding="utf-8", newline="\n")
    agreements = {}
    for source in "abcde":
        source_rows = [r for r in manifest if r["source"] == source]
        groups = {"all": source_rows, "<16": [r for r in source_rows if r["short_side_px"] < 16],
                  "16-32": [r for r in source_rows if 16 <= r["short_side_px"] < 32],
                  "32-64": [r for r in source_rows if 32 <= r["short_side_px"] < 64],
                  ">=64": [r for r in source_rows if r["short_side_px"] >= 64]}
        agreements[source] = {}
        for name, selected in groups.items():
            result = agreement([r["claude_label"] for r in selected], [r["label_sol"] for r in selected])
            result["sol_uncertain"] = sum(r["label_sol"] == "uncertain" for r in selected)
            result["claude_uncertain"] = sum(r["claude_label"] == "uncertain" for r in selected)
            determinate = [r for r in selected if "uncertain" not in (r["claude_label"], r["label_sol"])]
            result["both_decidable"] = agreement([r["claude_label"] for r in determinate],
                                                  [r["label_sol"] for r in determinate])
            agreements[source][name] = result
    owner_decidable = [r for r in owner_rows if r["owner_answer"] in ("ok", "not_ok") and r["sol_accepts_shown"] is not None]
    owner_result = {"n": len(owner_rows), "owner_unsure": sum(r["owner_answer"] == "unsure" for r in owner_rows),
                    "sol_uncertain": sum(r["sol_accepts_shown"] is None for r in owner_rows),
                    "both_decidable": len(owner_decidable),
                    "agreement_with_owner_assertion": sum((r["owner_answer"] == "ok") == r["sol_accepts_shown"] for r in owner_decidable) / len(owner_decidable) if owner_decidable else None,
                    "note": "Owner supplies ok/not_ok/unsure, not exact corrected codes; this is assertion agreement, not code accuracy or human GT.",
                    "rows": owner_rows}
    precision_tables = {}
    original_summary = read_json(BASE / "summary_all.json")
    for domain in ("hanoi", "v0", "combined"):
        rows = [r for r in field if domain == "combined" or r["domain"] == domain]
        precision_tables[domain] = {}
        for mode in ("claude", "sol", "consensus"):
            per_config = {}
            for config in "ABC":
                selected = [r for r in rows if r["config"] == config and (mode != "consensus" or r["claude"] == r["sol"])]
                per_config[config] = precision_summary([{**r, "label": r["claude"] if mode == "claude" else r["sol"]} for r in selected])
                expected = original_summary["hanoi"]["all"][config] if domain == "hanoi" else original_summary["v0"][config] if domain == "v0" else original_summary["combined_hanoi_v0"][config]
                if mode == "claude":
                    for count in ("n", "correct", "wrong_code", "not_a_sign", "uncertain"):
                        if per_config[config][count] != expected[count]:
                            raise ValueError(f"Claude baseline mismatch {domain}/{config}/{count}")
            precision_tables[domain][mode] = per_config
    rules = {}
    for mode in ("claude", "sol", "consensus"):
        table = {c: {**r, "proxy_recall": original_summary["proxy_recall"][c]["recall"]}
                 for c, r in precision_tables["combined"][mode].items()}
        rules[mode] = decide_recommendations(table)["t2"]
    predictions = pd.read_csv(ROOT / "output/kiet_wp_20261028/recognizer_v2/eval/test_predictions.csv", keep_default_na=False)
    classes = read_json(ROOT / "output/kiet_wp_20261028/recognizer_v2/class_list.json")
    old_codes = set(classes["classes"][:30])
    # The committed class list preserves the old class order before appending new codes.
    new_main = set(classes["main"]) - old_codes
    if len(new_main) != 25:
        raise ValueError("Expected 25 new main classes")
    eligible = [r for r in manifest if r["source"] == "c" and r["claude_label"] == r["label_sol"] and r["claude_label"] in new_main]
    indexed = predictions.set_index("canonical_box_id")
    accuracy = {}
    for group, low, high in (("<16", 0, 16), ("16-32", 16, 32), ("32-64", 32, 64), (">=64", 64, math.inf), (">=32", 32, math.inf), ("all", 0, math.inf)):
        selected = [r for r in eligible if low <= r["short_side_px"] < high]
        correct = sum(indexed.loc[r["original_key"], "new_pred"] == r["label_sol"] for r in selected)
        accuracy[group] = {"n": len(selected), "correct": correct, "accuracy": correct/len(selected) if selected else None,
                           "represented_classes": len({r["label_sol"] for r in selected})}
    disagreements = [r for r in manifest if r["claude_label"] != r["label_sol"]]
    with (out / "disagreements.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["item_id", "source", "original_key", "short_side_px", "claude_label", "label_sol", "confidence", "description"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in disagreements)
    result = {"status": "AI_SECOND_REVIEW_COMPLETE_NOT_HUMAN_GT", "seed": SEED, "n": len(manifest),
              "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "blind_seal": sealed,
              "agreement_by_source_size": agreements, "owner_spot_check": owner_result,
              "field_precision": precision_tables, "threshold_rules": rules,
              "new_main_consensus_test": {"declared_new_main_classes": len(new_main), "groups": accuracy},
              "disagreement_n": len(disagreements), "proxy_recall": original_summary["proxy_recall"], "gate_v2_identity": gate_identity,
              "ci_note": "2000 bootstrap draws over video-id/60-second segment clusters, seed 20261030; uncertain labels excluded from precision denominator.",
              "selection_note": "Consensus-only accuracy/precision is a selected subset, not a full-data or human-GT estimate. TEST was not used to change defaults."}
    write_json(out / "results.json", result)
    print(json.dumps({"n": len(manifest), "disagreements": len(disagreements), "kappa": {s: agreements[s]["all"]["kappa"] for s in "abcde"},
                      "owner": {k:v for k,v in owner_result.items() if k != "rows"},
                      "combined_precision": {m: {c:r["precision"] for c,r in v.items()} for m,v in precision_tables["combined"].items()},
                      "rules": rules, "new_main_consensus": accuracy["all"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "seal", "record", "analyze", "lane-prepare", "lane-summarize"])
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    {"prepare": prepare, "seal": seal, "record": record, "analyze": analyze, "lane-prepare": lane_prepare, "lane-summarize": lane_summarize}[args.command](args.output)
