"""M13: replay the forward-warning logic on cached COCO detections (no GPU).

Input: JSONL from scripts/forward_detect_cache_20261114.py. Uses the same ForwardObjectWarner, filters
and VoiceGate as the renderer. Prints alerts and spoken cues for the config given on the command line.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.perception.forward_objects import (ForwardObjectWarner, ForwardWarnConfig, VoiceGate,  # noqa: E402
                                            forward_cues)
from src.utils.config import load_config  # noqa: E402


def replay(cache: Path, cfg: ForwardWarnConfig, analysis_hz: float = 6.0):
    warner = ForwardObjectWarner(cfg, analysis_hz / max(1, cfg.detect_every))
    voice = VoiceGate(cfg)
    alerts, cues = [], []
    with cache.open(encoding='utf8') as fh:
        for line in fh:
            rec = json.loads(line)
            now = warner.update(rec['t'], rec['w'], rec['h'], rec['dets'])
            if now:
                alerts.extend(now)
                cues.extend(forward_cues(now, False, rec['t'], gate=voice))
    return alerts, cues


def summarise(alerts, cues, duration_s):
    return {
        'alerts': len(alerts),
        'by_level_cls': dict(Counter(f"{a['level']}/{a['cls']}" for a in alerts)),
        'spoken': len(cues),
        'spoken_per_minute': round(len(cues) / max(1e-9, duration_s / 60.0), 2),
        'alert_list': [(a['t'], a['track_id'], a['cls'], a['level'], bool(a.get('spoken'))) for a in alerts],
    }


def score_against_review(alerts, sheet: Path, window_s: float = 1.5):
    """Match each alert to a reviewed row: same track id, same class and level, within window_s seconds.
    Returns counts per verdict, alerts with no reviewed row (new) and reviewed rows that no alert matched."""
    import csv
    rows = [{'t': float(r['timestamp']), 'track': int(r['track_id']), 'cls': r['class'], 'level': r['level'],
             'verdict': r['ai_verdict'], 'id': r['id'], 'used': False}
            for r in csv.DictReader(sheet.open(encoding='utf8'))]
    out = Counter()
    matches = []
    for a in alerts:
        hit = None
        for r in rows:
            if (not r['used'] and r['track'] == a['track_id'] and r['cls'] == a['cls'] and r['level'] == a['level']
                    and abs(r['t'] - a['t']) <= window_s):
                hit = r
                break
        if hit:
            hit['used'] = True
            out[hit['verdict']] += 1
            matches.append((a['t'], a['track_id'], a['cls'], a['level'], hit['id'], hit['verdict']))
        else:
            out['sem_phieu'] += 1
            matches.append((a['t'], a['track_id'], a['cls'], a['level'], None, 'sem_phieu'))
    out['phieu_khong_khop'] = sum(1 for r in rows if not r['used'])
    return {'counts': dict(out), 'matches': matches}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cache', type=Path)
    ap.add_argument('--duration', type=float, required=True)
    ap.add_argument('--review', type=Path, help='M11 review sheet to score the alerts against')
    ap.add_argument('--set', action='append', default=[], help='key=value overrides of forward_warning config')
    a = ap.parse_args()
    cfg = ForwardWarnConfig.from_dict(load_config('perception').get('forward_warning', {}))
    cfg.enabled = True
    for kv in a.set:
        k, v = kv.split('=', 1)
        setattr(cfg, k, type(getattr(cfg, k))(json.loads(v)) if not isinstance(getattr(cfg, k), bool) else v == 'true')
    alerts, cues = replay(a.cache, cfg)
    result = summarise(alerts, cues, a.duration)
    if a.review:
        result['review'] = score_against_review(alerts, a.review)
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
