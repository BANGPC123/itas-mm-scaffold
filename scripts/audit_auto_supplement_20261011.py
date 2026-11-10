"""Development-only grouping audit vs existing AI-reviewed, NOT human GT."""
from pathlib import Path
import json
from src.context.sign_applicability import SignApplicability
from src.utils.types import SignDetection

root=Path(__file__).resolve().parents[1]
base=json.loads((root/'output/review/test_video_ai_review/ai_reviewed.json').read_text(encoding='utf-8'))
engine=SignApplicability()
tp=fp=fn=0
cases=[]
for frame in base['samples']:
    input_signs=[SignDetection(s['label'],1.,tuple(s['bbox_xyxy'])) for s in frame['signs']]
    converted=engine._propose_panel_roles(input_signs)
    original={tuple(s['bbox_xyxy']):s for s in frame['signs']}
    candidates=[(i,s) for i,s in enumerate(converted) if s.panel_role=='supplementary' and s.panel_role_source=='geometric_candidate']
    hits=0
    for i,s in candidates:
        truth=original[tuple(s.bbox_xyxy)]
        if truth['panel_role']=='supplementary':
            tp+=1;hits+=1
        else:fp+=1
    expected=sum(s['panel_role']=='supplementary' for s in frame['signs'])
    fn+=expected-hits
    cases.append({'frame':frame['id'],'true_plates':expected,'proposed_plates':len(candidates),'matched':hits})
out=root/'output/evaluation/supplement_candidates_ai_review_20261011.json'
out.write_text(json.dumps({'source':'AI-reviewed development snapshots, ZERO human-confirmed frames',
 'tp':tp,'fp':fp,'fn':fn,'precision_proxy':tp/(tp+fp) if tp+fp else None,
 'recall_proxy':tp/(tp+fn) if tp+fn else None,'frames':cases,
 'not_independent_accuracy':True},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'tp':tp,'fp':fp,'fn':fn,'precision_proxy':tp/(tp+fp) if tp+fp else None,
 'recall_proxy':tp/(tp+fn) if tp+fn else None},ensure_ascii=False),flush=True)
