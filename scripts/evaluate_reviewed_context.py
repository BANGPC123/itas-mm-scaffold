"""Score context rules only against explicitly human-reviewed video labels."""
from pathlib import Path
import argparse,json,sys,math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context.sign_applicability import SignApplicability
from src.utils.types import SignDetection,LaneInfo


def validate(packet,draft):
    if packet.get('source_sha256')!=draft.get('source_sha256'):raise ValueError('Source video mismatch')
    originals={s['id']:s for s in draft['samples']};seen=set();accepted=[]
    for sample in packet['samples']:
        sid=sample['id']
        if sid in seen or sid not in originals:raise ValueError('Duplicate/unknown frame')
        seen.add(sid);base=originals[sid]
        for key in ['images','width','height','timestamp_seconds']:
            if sample[key]!=base[key]:raise ValueError('Frame identity changed')
        if not sample.get('human_confirmed'):continue
        if not sample.get('reviewer','').strip() or not sample.get('reviewed_at'):raise ValueError('Missing human reviewer/date')
        if sample.get('visibility') not in ['clear','uncertain','unusable']:raise ValueError('Invalid visibility')
        if sample.get('vehicle') not in ['car','truck','motorcycle']:raise ValueError('Invalid vehicle')
        lanes=sample['ego_boundaries'];w=sample['width'];h=sample['height']
        if lanes and (len(lanes)!=2 or any(len(lane)<3 for lane in lanes)):raise ValueError('Need two boundaries, >=3 points each, or []')
        if any(not all(math.isfinite(v) for v in p) or not(0<=p[0]<w and 0<=p[1]<h) for lane in lanes for p in lane):raise ValueError('Lane point outside frame')
        ids={s['id'] for s in sample['signs']}
        if len(ids)!=len(sample['signs']):raise ValueError('Duplicate sign id')
        for s in sample['signs']:
            x1,y1,x2,y2=s['bbox_xyxy']
            if not all(math.isfinite(v) for v in [x1,y1,x2,y2]) or not(0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('Invalid box')
            if s['panel_role'] not in ['primary','supplementary','unknown']:raise ValueError('Invalid panel role')
            if s['expected_applicability'] not in ['applies','ignored','uncertain']:raise ValueError('Invalid expected result')
            if s['overhead'] is not None and not isinstance(s['overhead'],bool):raise ValueError('Invalid overhead')
            if not set(s['allowed_vehicles'])<=set(['car','truck','motorcycle']):raise ValueError('Invalid vehicle condition')
            if s.get('parent_id') is not None:
                parents=[p for p in sample['signs'] if p['id']==s['parent_id'] and p['panel_role']=='primary']
                if s['panel_role']!='supplementary' or len(parents)!=1:raise ValueError('Invalid supplementary parent')
        accepted.append(sample)
    if not accepted:raise ValueError('No human-confirmed frames. Draft predictions cannot be ground truth.')
    return accepted


def score(samples):
    cases=[];grouping=[];rules=SignApplicability()
    for sample in samples:
        if sample['visibility']=='unusable':continue
        signs=[SignDetection(s['label'],1.,tuple(s['bbox_xyxy']),panel_role=s['panel_role'],allowed_vehicles=s['allowed_vehicles'],overhead=s['overhead']) for s in sample['signs']]
        lane=LaneInfo([],ego_boundaries=sample['ego_boundaries'])
        outputs=rules.apply(signs,lane,(sample['height'],sample['width'],3),sample['vehicle'])
        for s in sample['signs']:
            if s['panel_role']=='supplementary':continue
            matching=[p for p in outputs if tuple(p.bbox_xyxy)==tuple(s['bbox_xyxy'])]
            predicted=matching[0].applicability if len(matching)==1 else 'missing'
            cases.append({'frame':sample['id'],'sign':s['id'],'expected':s['expected_applicability'],'predicted':predicted,'pass':predicted==s['expected_applicability']})
            expected=sorted([s['label']]+[p['label'] for p in sample['signs'] if p.get('parent_id')==s['id']])
            actual=sorted(matching[0].panel_labels or [matching[0].label]) if len(matching)==1 else []
            grouping.append({'frame':sample['id'],'sign':s['id'],'expected_labels':expected,'predicted_labels':actual,'has_supplementary':len(expected)>1,'pass':expected==actual})
    determinate=[c for c in cases if c['expected']!='uncertain']
    positive_groups=[c for c in grouping if c['has_supplementary']]
    return {'human_reviewed_frames':len(samples),'applicability_cases':cases,'grouping_cases':grouping,
            'applicability_accuracy':sum(c['pass'] for c in cases)/len(cases) if cases else None,
            'determinate_applicability_n':len(determinate),
            'determinate_applicability_accuracy':sum(c['pass'] for c in determinate)/len(determinate) if determinate else None,
            'uncertain_expected_n':len(cases)-len(determinate),
            'supplementary_group_n':len(positive_groups),
            'supplementary_group_accuracy':sum(c['pass'] for c in positive_groups)/len(positive_groups) if positive_groups else None,
            'grouping_accuracy':sum(c['pass'] for c in grouping)/len(grouping) if grouping else None,
            'scope':'Conditional logic using human boxes, roles, vehicles and lanes; NOT end-to-end perception accuracy. Grouping compared by label multiset, not parent identity when labels repeat.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('reviewed',type=Path);p.add_argument('--draft',type=Path,default=Path('output/review/test_video_development/draft.json'))
    p.add_argument('--output',type=Path,default=Path('output/evaluation/improvements_20261009/human_context.json'));a=p.parse_args()
    packet=json.loads(a.reviewed.read_text(encoding='utf-8'));draft=json.loads(a.draft.read_text(encoding='utf-8'))
    result=score(validate(packet,draft));a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
