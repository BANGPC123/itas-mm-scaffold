"""Compare a render against the M0 baseline: sign boxes/labels/applicability per sample, and speech events.

Usage: compare_gate_20261114.py <baseline_render_dir> <candidate_render_dir> <out_json>
"""
import json, sys
from pathlib import Path

def load_preds(d):
    rows = [json.loads(l) for l in (Path(d) / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    return rows

def sig(sign):
    return (sign.get('label'), tuple(round(v, 1) for v in sign.get('bbox_xyxy') or ()))

def main():
    base, cand, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    bp, cp = load_preds(base), load_preds(cand)
    report = {'baseline_samples': len(bp), 'candidate_samples': len(cp),
              'sample_count_equal': len(bp) == len(cp), 'box_label_mismatch_samples': [],
              'applicability_mismatch_samples': [], 'speech_equal': None}
    for i, (b, c) in enumerate(zip(bp, cp)):
        bs = sorted(sig(s) for s in b.get('signs', []))
        cs = sorted(sig(s) for s in c.get('signs', []))
        if bs != cs:
            report['box_label_mismatch_samples'].append({'index': i, 'timestamp': b.get('timestamp'),
                                                        'baseline': bs, 'candidate': cs})
        ba = sorted((s.get('label'), tuple(round(v, 1) for v in s.get('bbox_xyxy') or ()), s.get('applicability')) for s in b.get('signs', []))
        ca = sorted((s.get('label'), tuple(round(v, 1) for v in s.get('bbox_xyxy') or ()), s.get('applicability')) for s in c.get('signs', []))
        if ba != ca:
            report['applicability_mismatch_samples'].append({'index': i, 'timestamp': b.get('timestamp'),
                                                            'baseline': ba, 'candidate': ca})
    bsum = json.loads((base / 'summary.json').read_text(encoding='utf-8'))
    csum = json.loads((cand / 'summary.json').read_text(encoding='utf-8'))
    report['speech_equal'] = bsum.get('speech_events') == csum.get('speech_events')
    report['baseline_speech_events'] = len(bsum.get('speech_events') or [])
    report['candidate_speech_events'] = len(csum.get('speech_events') or [])
    report['box_label_mismatch_count'] = len(report['box_label_mismatch_samples'])
    report['applicability_mismatch_count'] = len(report['applicability_mismatch_samples'])
    report['gate_boxes_labels_100pct'] = report['sample_count_equal'] and report['box_label_mismatch_count'] == 0
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('_samples')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
