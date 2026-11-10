"""Source-disjoint Vietnam road smoke test (not a pixel-accuracy evaluation).

Use previously downloaded Commons street frames, never the original user
video or the development predictions. Ground truth is NOT human reviewed.
"""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline.orchestrator import Pipeline
import cv2,numpy as np


def main():
    pack=ROOT/'data/lane_external_source_review_20261011'
    manifest=json.loads((pack/'manifest.json').read_text(encoding='utf-8'))
    rows=defaultdict(list)
    for r in manifest['frames']:
        rows[r['temporal_group']].append(r)
    chosen=[]
    for group,rr in sorted(rows.items()):
        chosen.extend(sorted(rr,key=lambda x:hashlib.sha256(x['id'].encode()).digest())[:10])
    assert len(rows)>=2 and len(chosen)>=16
    print('EXTERNAL_GROUPS',sorted(rows),len(chosen),flush=True)
    model=Pipeline()
    reports=[];errors=[];cnt=Counter()
    for i,row in enumerate(chosen):
        p=pack/row['image']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('External image SHA256 mismatch')
        image=cv2.imread(str(p))
        if image is None:raise RuntimeError('Cannot decode external road image')
        t=time.perf_counter()
        try:
            res=model.run(image,0.,0.,synthesize_audio=False,annotate=False,
                           lane_visual_roi=True,road_bottom=1.)
            details={'id':row['id'],'group':row['temporal_group'],
               'lane_backend':res.perception.lane.backend,
               'camera_view':res.perception.lane.camera_view,
               'predicted_line_count':len(res.perception.lane.display_polylines or []),
               'recognized_count':sum(s.label!='unknown' for s in res.perception.signs),
               'strict_verified_applies':sum(s.applicability=='applies' for s in res.perception.signs),
               'car_candidate_count':sum(getattr(s,'car_context',{}).get('status')=='applies_candidate'
                     for s in res.perception.signs),
               'legal_speech_empty':not bool(res.reasoning.speech_text),
               'elapsed_ms':round((time.perf_counter()-t)*1000,1)}
            if details['strict_verified_applies'] or not details['legal_speech_empty']:
                errors.append('unverified_legal_assertion:'+row['id'])
            reports.append(details)
            cnt['success']+=1
            cnt['lines']+=details['predicted_line_count']
            cnt['recognized']+=details['recognized_count']
        except Exception as exc:
            errors.append(row['id']+':'+type(exc).__name__+':'+str(exc)[:200])
            cnt['failed']+=1
        print('SOURCE_DISJOINT_SMOKE',i+1,'/',len(chosen),'errors',len(errors),flush=True)
    data={'run_type':'external_source_disjoint_operational_test',
          'human_verified_ground_truth':False,'visual_accuracy_F1_available':False,
          'independent_original_video_groups':list(sorted(rows)),
          'tested_images':len(chosen),'success':cnt['success'],
          'failed':cnt['failed'],'recognized_detections':cnt['recognized'],
          'predicted_lane_segments':cnt['lines'],
          'unverified_legal_assertions':sum(not r['legal_speech_empty'] or r['strict_verified_applies']
                                         for r in reports),
          'checks_passed':not errors,'failures':errors,
          'frames':reports}
    out=ROOT/'output/evaluation/source_disjoint_smoke_20261014.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print('EXTERNAL_SMOKE_RESULT',json.dumps({k:v for k,v in data.items() if k!='frames'},
        ensure_ascii=False),flush=True)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
