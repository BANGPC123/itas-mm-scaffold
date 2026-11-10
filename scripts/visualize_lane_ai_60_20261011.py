"""Representative AI lane-proposal visualization: human-auditable 60/600 frames."""
from pathlib import Path
import cv2,json,numpy as np,collections
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'data/lane_vietnam_review_20261011'
out=P/'ai_visual_quality'
out.mkdir(exist_ok=True)
rows=json.loads((P/'manifest.json').read_text(encoding='utf-8'))['frames']
drafts={x['id']:x for x in map(json.loads,(P/'ai_drafts.jsonl').read_text(encoding='utf-8').splitlines())}
# sample exactly 2 per time block: one near midpoint and one at end
chosen=[rows[i] for i in range(600) if i%20 in (6,16)]
notes=[]
for batch in range(3):
    items=chosen[batch*20:(batch+1)*20]
    tiles=[]
    for r in items:
        im=cv2.imread(str(P/r['image']))
        d=drafts[r['id']]
        for line in d['ai_proposals']:
            clr=(0,235,255) if line['source'].startswith('YOLOPv2') else (255,220,40)
            if len(line['points'])>1:
                cv2.polylines(im,[np.asarray(line['points'],np.int32)],False,clr,3)
        tile=cv2.resize(im,(360,202))
        cv2.rectangle(tile,(0,0),(360,28),(20,25,30),-1)
        cv2.putText(tile,r['id']+' '+str(int(r['source_video_seconds']))+'s U'+str(d['ai_u_lines'])+' V'+str(d['ai_v2_lines']),
                    (4,20),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
        tiles.append(tile)
        notes.append({'frame':r['id'],'timestamp':r['source_video_seconds'],
                      'source_group':r['temporal_group'],
                      'AI_UFLD_candidates':d['ai_u_lines'],'AI_YOLOPv2_candidates':d['ai_v2_lines'],
                      'human_verified':False,'requires_visual_review':True})
    image=np.vstack([np.hstack(tiles[i:i+5]) for i in (0,5,10,15)])
    cv2.imwrite(str(out/f'contact_{batch+1}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,89])
(out/'representative_ai_proposals.json').write_text(json.dumps({'frames_examined_in_contact':len(notes),
     'AI_PROPOSALS_NOT_LABELS':True,'no_human_GT_claim':True,'frames':notes},ensure_ascii=False,indent=2),encoding='utf-8')
print('REVIEW_CONTACT',len(notes),'frames in 3 contact sheets',flush=True)
