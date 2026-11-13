"""M1.5(e) gate: sign codes per clip, baseline vs candidate. Lists lost/added codes with timestamps.

Usage: compare_codes_20261114.py <baseline_render_dir> <candidate_render_dir> <out_json>
Counts a code once per (sample timestamp, label) for labels other than 'unknown'.
"""
import json, sys
from pathlib import Path
from collections import Counter

def codes(d):
    out = []
    for line in (Path(d) / 'predictions.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for s in r.get('signs', []):
            if s.get('label') and s['label'] != 'unknown':
                out.append((round(r['timestamp'], 3), s['label']))
    return out

def main():
    b, c, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    bc, cc = codes(b), codes(c)
    bcount, ccount = Counter(bc), Counter(cc)
    lost = sorted((bcount - ccount).elements())
    added = sorted((ccount - bcount).elements())
    base_n, cand_n = len(bc), len(cc)
    drop_pct = 100.0 * (base_n - cand_n) / base_n if base_n else 0.0
    rep = {'baseline_code_detections': base_n, 'candidate_code_detections': cand_n,
           'change_pct': round(-drop_pct, 2), 'lost_count': len(lost), 'added_count': len(added),
           'lost': [{'t': t, 'label': l} for t, l in lost], 'added': [{'t': t, 'label': l} for t, l in added],
           'gate_drop_le_5pct': drop_pct <= 5.0}
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in rep.items() if k not in ('lost', 'added')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
