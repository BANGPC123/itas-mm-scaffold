"""Assistant visual QA for all 82 review candidate boxes.

IMPORTANT: This is AI-only visual interpretation, NOT human ground truth.
Does not modify original review_manifest.json or any TRAIN/VAL/TEST sets.
"""
from pathlib import Path
from collections import Counter
import json,hashlib
root=Path(r'D:\itas-mm')
pack=root/'output/review/field_review_pack_20261011'
src=json.loads((pack/'review_manifest.json').read_text(encoding='utf-8'))
# S=true traffic sign/panel, N=clearly NOT traffic sign, U=not enough visual evidence.
labels=[
 'SSN',      # 00
 'SSS',      # 01
 'SSS',      # 02
 'SNN',      # 03
 'SSSN',     # 04
 'SSN',      # 05
 'SSSNU',    # 06
 'SSNN',     # 07
 'SSS',      # 08
 'NU',       # 09
 'SSSS',     # 10
 'UNN',      # 11
 'SSS',      # 12
 'SSNNSSSS', # 13
 'SNN',      # 14
 'SNNN',     # 15
 'UN',       # 16
 'SN',       # 17
 'UNN',      # 18
 'NNN',      # 19
 'NN',       # 20
 'SNNNN',    # 21
 'SSSNS',    # 22
 'SS',       # 23
]
assert len(labels)==len(src['frames'])==24
assert sum(len(v) for v in labels)==82

# Context evidence keyed by (frame_number,candidate_index).
notes={
 (0,2):'Vehicle/cabin portion, not W.224 warning sign.',
 (1,1):'Real blue lane-use diagram, but exact Vietnamese code cannot be resolved.',
 (1,2):'White/red height clearance restriction sign; numeric restriction is not machine-readable here.',
 (2,0):'Real red prohibition symbol, subtype P.124d not confidently distinguished at source resolution.',
 (3,1):'Orange/yellow commercial or shop advertisement.',
 (3,2):'Blue commercial signage, not traffic sign I.409.',
 (4,0):'Blue street/navigation sign; code not fully legible.',
 (4,3):'Amber traffic-signal lamp; not a road sign.',
 (5,2):'Red facade/company logo, not warning W.207.',
 (6,2):'Real blue vehicle/lane-direction panel; exact code unresolved.',
 (6,3):'Yellow/blue circular commercial or vehicle object, not a traffic sign.',
 (6,4):'Blurred vegetation/advertisement, shape insufficient to determine.',
 (7,1):'Rectangular supplementary sign with time/conditions under main prohibition sign; OCR not confirmed.',
 (7,2):'Blue radio/Wi-Fi-like symbol on advertisement, not mandatory direction R.303.',
 (7,3):'Interior windshield A-pillar, not sign.',
 (8,2):'Appears to contain a circular roadside restriction sign; subtype not readable.',
 (9,0):'Red traffic-signal light, not traffic sign.',
 (9,1):'Low-res reflective roadside region: uncertain.',
 (10,2):'Circular red prohibition roadside sign; exact class unconfirmed.',
 (10,3):'Blue traffic lane-use panel, class unknown.',
 (11,0):'Tiny blue/red roadside shape; not enough pixels for confident object decision.',
 (11,1):'Leaves/vegetation, not sign.',
 (11,2):'Street-vendor umbrella, not sign.',
 (12,0):'Blue overhead/roadside guide sign for street navigation.',
 (12,1):'Red-bordered prohibition sign; exact class unclear.',
 (12,2):'Second red-bordered traffic prohibition sign; exact class unclear.',
 (13,0):'Real vehicle restriction (truck/vehicle) circular sign; exact code unverified.',
 (13,2):'Bounding box covers trees to the side of a real no-parking sign; not a usable sign box.',
 (13,3):'Bounding box covers foliage/edge outside restriction sign; not a usable box.',
 (13,4):'Blue lane allocation/vehicle category panel; code unverified.',
 (13,5):'Same or adjacent blue lane use panel; potential duplicate of nearby box.',
 (13,6):'White supplementary panel displaying times under red vehicle restriction.',
 (13,7):'Portion of a blue lane-allocation sign; I.409 subtype and bounding box not verified.',
 (14,0):'Red circular prohibition with white time panel beneath, exact prohibition code unclear.',
 (14,1):'Traffic signal assembly, not roadside traffic sign.',
 (14,2):'Red traffic signal or reflection, not sign.',
 (15,1):'Dark storefront sign, not a regulatory traffic sign.',
 (15,2):'Red traffic-signal lamp, not sign.',
 (15,3):'Red decorative logo/retail sign, not a traffic sign.',
 (16,0):'Real no-parking sign immediately RIGHT of bounding box; current box primarily covers tree foliage, needs relabeling/reboxing.',
 (16,1):'Light pole or equipment cabinet, not sign.',
 (17,1):'Traffic light support/pole, not road sign.',
 (18,0):'Blurred roadside pole/sign area, object identity uncertain.',
 (18,1):'Commercial street sign or red storefront graphic, not traffic regulatory sign.',
 (18,2):'Small commercial shop sign, not a roadside traffic sign.',
 (19,0):'Tree canopy/branches outside sign, not sign.',
 (19,1):'Tree trunk, not sign.',
 (19,2):'Cabin windshield A-pillar, not W.220.',
 (20,0):'Cabin windshield A-pillar, not W.220.',
 (20,1):'Dashboard circular gauge, not P.127 speed-limit sign.',
 (21,0):'Circular red vehicle restriction sign; previous P.130 no-stopping code is incorrect (blue field + red cross absent).',
 (21,1):'Blank building wall, not sign.',
 (21,2):'Shop awning and frame, not W.225.',
 (21,3):'Retail sign/awning, not traffic sign.',
 (21,4):'Building facade/window edge, not sign.',
 (22,0):'Actual no-stopping sign at frame left; current bbox offset to its right.',
 (22,1):'Another vehicle restriction sign to right; subtype unknown.',
 (22,2):'W.225 warning triangle (children crossing), but candidate bbox misses substantial part of sign.',
 (22,3):'Tree/foliage overlap, no readable traffic sign within box.',
 (22,4):'White supplementary plate with hours under right restriction sign, bbox partial.',
 (23,1):'Blue road/lane use information panel; I.409 class not verified.',
}
bbox_issues={
 (0,0),(0,1),(1,0),(2,1),(2,2),(6,1),
 (7,1),(13,2),(13,3),(13,5),(13,6),(13,7),
 (16,0),(22,0),(22,2),(22,4)
}
unknown_code={
 (1,1),(2,0),(3,0),(4,0),(6,2),(7,1),(8,2),(9,1),
 (10,2),(10,3),(11,0),(12,0),(12,1),(12,2),
 (13,0),(13,4),(13,5),(13,6),(13,7),(14,0),
 (16,0),(18,0),(21,0),(22,1),(22,4),(23,1)
}
plate={(7,1),(13,6),(22,4)}
out=[]
for index,(f,decisions) in enumerate(zip(src['frames'],labels)):
    assert len(f['signs'])==len(decisions), (f['id'],len(f['signs']),decisions)
    pred=[]
    for j,(s,decision) in enumerate(zip(f['signs'],decisions)):
        origin=s['label']
        real={'S':'sign','N':'not_sign','U':'uncertain'}[decision]
        label=(origin if decision=='S' and origin!='unknown' and (index,j) not in unknown_code else 'unknown')
        if decision=='N':label='background'
        subtype='supplementary' if (index,j) in plate else s['panel_role']
        # Code confidence is only visual support; never use as reviewed=TRUE.
        code_supported=bool(decision=='S' and label not in ('unknown','background'))
        new=dict(s)
        new.update({'label':label,'judgment':'unreviewed',
            'ai_visual_assessment':real,
            'ai_label_suggestion':label,
            'ai_code_visually_supported':code_supported,
            'ai_bbox_needs_correction':(index,j) in bbox_issues,
            'ai_visual_note':notes.get((index,j), 'Roadside regulatory sign visually apparent.' if decision=='S' else
                ('Clearly not a mounted traffic sign at current bounding box.' if decision=='N' else
                 'Insufficient crop detail for reliable distinction.')),
            'ai_visual_assessment_provenance':'assistant_nonhuman_only',
            'ai_panel_role_suggestion':subtype})
        # Keep original label, box and role for auditing.
        new['ai_original_label']=origin
        new['label']=origin     # Human review UI MUST NOT silently accept our proposed code.
        pred.append(new)
    copy=dict(f)
    copy['signs']=pred
    copy['human_confirmed']=False
    copy['reviewer']=''
    copy['reviewed_at']=None
    copy['all_visible_signs_annotated']=False
    copy['lane_status']='unreviewed'
    copy['lanes']=[]
    out.append(copy)
base={'schema_version':'itasmm-field-review-v1','provenance':{
  'type':'assistant_visual_review_NOT_human','source_video_group':'test_video_development',
  'human_confirmed':False,'independent_test':False,
  'evidence':'Source frames and 82 box crops viewed; where ambiguous use uncertain',
  'warning':'Human confirmation required; do NOT use AI-only suggestions to claim supervised ground truth.'},
  'frames':out}
folder=pack/'ai_visual_review'
(folder/'ai_assessment.json').write_text(json.dumps(base,ensure_ascii=False,indent=2),encoding='utf-8')
counts=Counter(s['ai_visual_assessment'] for f in out for s in f['signs'])
codestatus=Counter('supported' if s['ai_code_visually_supported'] else 'unverified'
  for f in out for s in f['signs'] if s['ai_visual_assessment']=='sign')
issues=sum(s['ai_bbox_needs_correction'] for f in out for s in f['signs'])
summary={'frames':len(out),'candidates':sum(len(f['signs']) for f in out),
  'ai_judgments':dict(counts),'code_status_on_real_signs':dict(codestatus),
  'bbox_needs_correction':issues,'human_confirmed_frames':0,'independent_field_gt':False,
  'full_image_exhaustive':False,'new_background_images_added_to_train':0}
(folder/'ai_assessment_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False),flush=True)
