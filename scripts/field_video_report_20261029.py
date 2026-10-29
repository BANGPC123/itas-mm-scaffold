"""Aggregate field-video evaluation results into JSON and Markdown tables (issue #2, WP3).

Inputs are the run folders of scripts/run_field_eval_20261029.py, the blind-label store of
scripts/field_review_pack_20261029.py, the graded V0 review and the proxy-recall labels. Nothing is typed by hand:
every number in docs/FIELD_VIDEO_EVAL_20261029.md comes from the JSON written here. Labels are AI-reviewed.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.field_review_pack_20261029 import RUN_NAME, collect_runs, grade_members, merge_evidence
from scripts.field_video_eval_20261029 import (
    bootstrap_precision, decidable, decide_recommendations, precision, read_sources, tally)

CONFIG_TEXT = {"A": "old 0.63 + old gate", "B": "new 0.87", "C": "new T2", "D": "new T2 + gate v2", "E": "new 0.87 + gate v2"}


def run_metrics(run_dir):
    """Counts of one run: tracks, confirmed (track, label) pairs, displayed unknown samples, alerts and inference speed."""
    run_dir = Path(run_dir)
    match = RUN_NAME.match(run_dir.name)
    start = int(match["start"])
    report = json.loads((run_dir / "temporal_benchmark.json").read_text(encoding="utf-8"))
    segment = report["segments"][0]
    tracks, confirmed, unknown = set(), set(), 0
    for line in (run_dir / f"segment_{start}/predictions.jsonl").read_text(encoding="utf-8").splitlines():
        for sign in json.loads(line)["signs"]:
            if sign.get("track_id") is not None:
                tracks.add(sign["track_id"])
            if sign.get("label") == "unknown":
                unknown += 1
            elif sign.get("recognition_source") == "temporal_best_crop":
                confirmed.add((sign.get("track_id"), sign["label"]))
    return {"samples": segment["samples"], "tracks": len(tracks), "confirmed_pairs": len(confirmed), "unknown_samples": unknown,
            "alert_samples": segment["alert_samples"], "alert_episodes": segment["alert_episodes"],
            "call_seconds": segment["samples"] / segment["inference_fps"] if segment["inference_fps"] else 0.0,
            "verified_full_decode": bool(segment["verified_full_decode"])}


def aggregate_metrics(rows):
    keys = ("samples", "tracks", "confirmed_pairs", "unknown_samples", "alert_samples", "alert_episodes", "call_seconds")
    total = {key: sum(r[key] for r in rows) for key in keys}
    total["runs"] = len(rows)
    total["inference_fps"] = total["samples"] / total["call_seconds"] if total["call_seconds"] else None
    return total


def summarize_graded(rows, cluster_key, n_boot=2000):
    counts = tally(r["verdict"] for r in rows)
    low, high = bootstrap_precision([(cluster_key(r), r["verdict"]) for r in rows], n_boot=n_boot)
    return {**counts, "decidable": decidable(counts), "precision": precision(counts), "ci_low": low, "ci_high": high}


def fmt(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def precision_cell(summary):
    if summary["precision"] is None:
        return "-"
    return f"{summary['precision']:.3f} [{fmt(summary['ci_low'])}, {fmt(summary['ci_high'])}]"


def config_table(title, per_config, extra=None):
    """Markdown table of verdict counts and precision per configuration."""
    head = "| Config | Setting | Confirmed labels | correct | wrong_code | not_a_sign | uncertain | Precision [95% CI] |"
    lines = [f"**{title}**", "", head, "|---|---|---:|---:|---:|---:|---:|---|"]
    for name in sorted(per_config):
        s = per_config[name]
        lines.append(f"| {name} | {CONFIG_TEXT.get(name, '')} | {s['n']} | {s['correct']} | {s['wrong_code']} | {s['not_a_sign']} | "
                     f"{s['uncertain']} | {precision_cell(s)} |")
    return "\n".join(lines)


def run_table(title, per_config):
    lines = [f"**{title}**", "", "| Config | Runs | Tracks | Confirmed pairs | Unknown samples | Alert samples | Alert episodes | Inference FPS |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in sorted(per_config):
        m = per_config[name]
        lines.append(f"| {name} | {m['runs']} | {m['tracks']} | {m['confirmed_pairs']} | {m['unknown_samples']} | "
                     f"{m['alert_samples']} | {m['alert_episodes']} | {fmt(m['inference_fps'], 2)} |")
    return "\n".join(lines)


def build(args):
    field_eval = Path(args.field_eval)
    sources = {r["video_id"]: r for r in read_sources(args.sources)}
    store = json.loads(Path(args.store).read_text(encoding="utf-8"))
    items, _, runs = collect_runs(field_eval, args.configs)
    records = merge_evidence(items)
    graded = grade_members(records, store)
    for row in graded:
        row["day_night"] = sources[row["video_id"]]["day_night"]
        row["cluster"] = f"{row['video_id']}@{row['start']}"
    metrics = {}
    for name in runs:
        match = RUN_NAME.match(name)
        metrics.setdefault(match["config"], []).append({**run_metrics(field_eval / name), "video_id": match["video"]})
    configs = sorted(metrics)

    def by(rows, pick):
        return {c: summarize_graded([r for r in rows if r["config"] == c and pick(r)], lambda r: r["cluster"], args.boot) for c in configs}

    hanoi = {"all": by(graded, lambda r: True), "day": by(graded, lambda r: r["day_night"] == "day"),
             "night": by(graded, lambda r: r["day_night"] == "night")}
    run_all = {c: aggregate_metrics(metrics[c]) for c in configs}
    run_day = {c: aggregate_metrics([m for m in metrics[c] if sources[m["video_id"]]["day_night"] == "day"]) for c in configs}
    run_night = {c: aggregate_metrics([m for m in metrics[c] if sources[m["video_id"]]["day_night"] == "night"]) for c in configs}

    v0_rows = json.loads(Path(args.v0_graded).read_text(encoding="utf-8"))
    for r in v0_rows:
        r["cluster"] = f"V0@{r['segment']}"
    v0 = {c: summarize_graded([r for r in v0_rows if r["config"] == c], lambda r: r["cluster"], args.boot) for c in sorted({r["config"] for r in v0_rows})}

    proxy = None
    if args.proxy_items:
        from scripts.field_proxy_pack_20261029 import recall_table
        items_p = json.loads(Path(args.proxy_items).read_text(encoding="utf-8"))
        labels_p = json.loads(Path(args.proxy_labels).read_text(encoding="utf-8"))["labels"]
        trained = set(json.loads((ROOT / "output/kiet_wp_20261028/recognizer_v2/class_list.json").read_text(encoding="utf-8"))["classes"])
        proxy = recall_table(items_p, labels_p, lambda c, v, s: field_eval / f"{c}_{v}_{s}" / f"segment_{s}" / "predictions.jsonl", configs)
        for c in configs:
            proxy[c]["in_class_list"] = sum(d["code"] in trained for d in proxy[c]["details"])
            proxy[c]["details"] = [{k: d[k] for k in ("number", "video_id", "start", "code", "caught")} for d in proxy[c]["details"]]

    combined = {}
    for c in sorted(set(hanoi["all"]) & set(v0)):
        merged = [r for r in graded if r["config"] == c] + [r for r in v0_rows if r["config"] == c]
        combined[c] = summarize_graded(merged, lambda r: r["cluster"], args.boot)
    table = {c: {**combined[c], "proxy_recall": proxy[c]["recall"] if proxy else None} for c in combined}
    for c in hanoi["all"]:
        table.setdefault(c, {**hanoi["all"][c], "proxy_recall": proxy[c]["recall"] if proxy else None})
    gate_pairs = {g: b for g, b in (("D", "C"), ("E", "B")) if g in hanoi["all"] and b in hanoi["all"]}
    margin = json.loads(Path(args.margin).read_text(encoding="utf-8")) if args.margin else None
    f1_drop = {}
    if margin:
        for g, point in (("D", "T2"), ("E", "0.87")):
            f1_drop[g] = margin["chosen"]["points"][point]["f1_drop"] if margin.get("chosen") else None
    # D/E have no V0 runs: the gate rule is evaluated on the Hanoi videos only
    decision_table = {c: ({**hanoi["all"][c], "proxy_recall": proxy[c]["recall"] if proxy else None} if c in ("D", "E") else table[c])
                      for c in table}
    t2_table = {c: decision_table[c] for c in "ABC" if c in decision_table}
    gate_table = {c: {**hanoi["all"][c], "proxy_recall": proxy[c]["recall"] if proxy else None} for c in hanoi["all"]}
    decisions = decide_recommendations(t2_table, 30, None, None)
    gate_decisions = decide_recommendations(gate_table, 30, gate_pairs, f1_drop)["gate"] if gate_pairs else {}
    hanoi_only = decide_recommendations({c: gate_table[c] for c in "ABC" if c in gate_table}, 30)["t2"] if all(c in gate_table for c in "ABC") else None
    result = {"hanoi": hanoi, "hanoi_runs": {"all": run_all, "day": run_day, "night": run_night}, "v0": v0, "combined_hanoi_v0": combined,
              "proxy_recall": proxy, "decisions": {"t2_combined": decisions["t2"], "t2_hanoi_only": hanoi_only, "gate": gate_decisions},
              "runs": runs, "evidence_items": len(records), "member_labels": len(graded)}
    Path(args.out_json).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    md = [config_table("Hanoi EVAL, all videos", hanoi["all"]), "", config_table("Hanoi EVAL, day", hanoi["day"]), "",
          config_table("Hanoi EVAL, night", hanoi["night"]), "", config_table("V0 (development video, TEST group)", v0), "",
          config_table("Hanoi EVAL + V0 combined (rule input)", combined), "", run_table("Run counts, Hanoi all", run_all), "",
          run_table("Run counts, day", run_day), "", run_table("Run counts, night", run_night)]
    if proxy:
        md += ["", "**Proxy recall (AI-annotated, conditional on detector proposals)**", "", "| Config | Annotated signs | Caught | Proxy recall |",
               "|---|---:|---:|---:|"] + [f"| {c} | {proxy[c]['annotated']} | {proxy[c]['caught']} | {fmt(proxy[c]['recall'])} |" for c in configs]
    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"configs": configs, "evidence_items": len(records), "member_labels": len(graded),
                      "t2": decisions["t2"]["decision"], "gate": {k: v["decision"] for k, v in gate_decisions.items()}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-eval", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--v0-graded", required=True)
    parser.add_argument("--proxy-items")
    parser.add_argument("--proxy-labels")
    parser.add_argument("--margin")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--configs", nargs="*", help="restrict to these configurations (default: every finished run)")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
