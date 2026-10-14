"""Create AI-only lane-mark proposals on all 600 reviewed-source frames.

UFLDv2 + YOLOPv2 segmentation from frozen original checkpoints.
All output remains UNVERIFIED. No self-label training and no reported GT.
"""
from pathlib import Path
import sys
import cv2,torch,json,time,collections,numpy as np
from src.perception.lane_detector import LaneDetector
from src.perception.yolopv2_lane import YOLOPv2Lane
from src.utils.config import load_config
ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'data/lane_vietnam_review_20261011'
manifest=json.loads((PACK/'manifest.json').read_text(encoding='utf-8'))
if (PACK/'ai_drafts.jsonl').exists():raise FileExistsError('AI draft exists, refuse overwrite')
(PACK/'ai_masks').mkdir(exist_ok=True)
torch.set_num_threads(4);cv2.setNumThreads(1)
lane=LaneDetector(load_config('perception'))
seg=YOLOPv2Lane(ROOT/'models/lane/candidates/yolopv2_official_20261011/yolopv2.pt',
                device='cuda:0' if torch.cuda.is_available() else 'cpu',half=True,roi=.80,threshold=.56)
started=time.perf_counter();records=[];counts=collections.Counter()
with (PACK/'ai_drafts.partial.jsonl').open('w',encoding='utf-8') as fp:
    for i,row in enumerate(manifest['frames']):
        img=cv2.imread(str(PACK/row['image']))
        if img is None:raise RuntimeError('Missing '+row['image'])
        u=lane.detect(img)
        mask=seg.segment(img)
        candidate=seg.detect(img)
        mask_file=f'ai_masks/{row["id"]}.png'
        cv2.imwrite(str(PACK/mask_file),(mask*255).astype('uint8'),[cv2.IMWRITE_PNG_COMPRESSION,5])
        proposals=[]
        for j,pts in enumerate(u.polylines[:6]):
            if len(pts)<4:continue
            proposals.append({'id':f'ufld_{j}','points':[[int(x),int(y)] for x,y in pts],
                'type':'unknown_lane_line','source':'UFLDv2_CULane_AI_UNVERIFIED','human_approved':False})
        for j,pts in enumerate(candidate.polylines[:6]):
            if len(pts)<4:continue
            proposals.append({'id':f'v2_{j}','points':[[int(x),int(y)] for x,y in pts],
                'type':'unknown_lane_line','source':'YOLOPv2_BDD_AI_UNVERIFIED','human_approved':False})
        record={'id':row['id'],'image_sha256':row['sha256'],
                'mask':mask_file,'ai_proposals':proposals,
                'ai_u_lines':sum(p['source'].startswith('UFLDv2') for p in proposals),
                'ai_v2_lines':sum(p['source'].startswith('YOLOPv2') for p in proposals),
                'ai_suggested_no_lane':not bool(proposals),
                'ai_only':True,'human_verified':False}
        fp.write(json.dumps(record,ensure_ascii=False)+'\n')
        records.append(record)
        counts['ufld_candidates']+=record['ai_u_lines']
        counts['yolopv2_candidates']+=record['ai_v2_lines']
        counts['frames_no_ai_line']+=int(record['ai_suggested_no_lane'])
        if (i+1)%50==0:
            fp.flush()
            print('AI_PROPOSALS',i+1,'/',len(manifest['frames']),
                  'elapsed_s',round(time.perf_counter()-started,1),flush=True)
(PACK/'ai_drafts.partial.jsonl').rename(PACK/'ai_drafts.jsonl')
summary={'status':'AI_PROPOSALS_ONLY_NOT_GROUND_TRUTH',
         'frames':len(records),'human_verified_frames':0,
         'proposals_counts':dict(counts),
         'models':['UFLDv2 original CULane','YOLOPv2 BDD100K'],
         'checkpoint_modified':False,'ai_mask_is_lane_ground_truth':False,
         'elapsed_seconds':round(time.perf_counter()-started,1)}
(PACK/'ai_draft_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('AI_DRAFT_PACK_READY',json.dumps(summary,ensure_ascii=False),flush=True)
