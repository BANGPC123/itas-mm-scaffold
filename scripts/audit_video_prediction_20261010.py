"""Summarize model predictions from a development video. No truth labels."""
from pathlib import Path
from collections import Counter
import json
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'output/video_tests/test_video_20261010_after_hardening'
signs=[]
suspicious=[]
with (folder/'predictions.jsonl').open(encoding='utf-8') as f:
    for line in f:
        row=json.loads(line)
        for s in row['signs']:
            signs.append(s)
            if 20 <= row['timestamp_seconds'] <= 30 and s['label']=='P.127':
                suspicious.append({'second':row['timestamp_seconds'],'box':s['bbox_xyxy'],'conf':s['confidence'],'recognition_conf':s['recognition_confidence']})
labels=Counter(s['label'] for s in signs)
applicability=Counter(s['applicability'] for s in signs)
sources=Counter(s['recognition_source'] for s in signs)
causes=Counter(s['applicability_reason'] for s in signs)
summary={'scope':'development replay; cannot infer correctness or advertising false-positive rate',
         'observations':len(signs),'known':len(signs)-labels['unknown'],
         'labels':dict(labels.most_common()),
         'applicability':dict(applicability),
         'applicability_reasons':dict(causes.most_common()),
         'recognition_sources':dict(sources.most_common()),
         'P127_20_to_30_sec':suspicious[:20]}
(folder/'prediction_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
