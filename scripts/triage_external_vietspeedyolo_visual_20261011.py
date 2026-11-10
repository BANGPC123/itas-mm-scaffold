"""Source-specific AI visual quality review of the 15 new candidate frames.

Only assistant visual assessment; NOT human scene quality certification.
"""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'output/review/external_vietspeedyolo_20261011'
src=json.loads((PACK/'review_manifest.json').read_text(encoding='utf-8'))
bad={
 'ext_frame_0000':'Upside-down scene, likely 180-degree rotation augmentation.',
 'ext_frame_0003':'Repeated/composited trucks/sign in one frame; synthetic edit.',
 'ext_frame_0006':'Road sign appears pasted onto rear of truck; synthetic augmentation.',
 'ext_frame_0008':'Upside-down scene, likely 180-degree rotation augmentation.',
 'ext_frame_0014':'Road sign appears pasted onto vehicle; synthetic augmentation.'}
rows=[]
for f in src['frames']:
    flag=f['id'] in bad
    rows.append({'frame_id':f['id'],
       'original_source':f['source_filename'],
       'visual_quality':'AI_REJECT_PLAUSIBLE_AUGMENTATION' if flag else 'AI_PLAUSIBLE_ROAD_PHOTO',
       'reason':bad.get(f['id'],'Appears to show an outdoor road scene; originality, mirroring and annotation still unverified.'),
       'human_verified':False,'usable_as_independent_test':False,
       'approved_for_training':False})
report={'assessed':len(rows),
        'ai_rejected_obvious_augmentation':sum(x['visual_quality']=='AI_REJECT_PLAUSIBLE_AUGMENTATION' for x in rows),
        'ai_plausible_road_photos':sum(x['visual_quality']=='AI_PLAUSIBLE_ROAD_PHOTO' for x in rows),
        'human_verified':0,'rows':rows,
        'statement':'These observations are AI visual triage from source image previews, not independently signed labels.'}
(PACK/'ai_visual_quality_triage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False),flush=True)
