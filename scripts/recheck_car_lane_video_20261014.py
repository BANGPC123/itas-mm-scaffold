"""Visual recheck from verified cached ML, for selected ranges of test_video.

Keeps original frames, sign boxes and legal uncertainty. Applies ONLY the
additional long road-edge contour veto; recomputes machine car wording.
This does not re-run sign/lane neural models or provide ground truth.
"""
from pathlib import Path
import argparse
import copy
import json
import sys
import cv2
import numpy as np
from PIL import ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.perception.road_surface_veto import RoadSurfaceVeto
from src.perception.road_edge_veto import suppress_road_edge_contours
from src.context.car_context import assess_car,car_summary
from src.perception.overlay_motion import OverlayMotion
from src.perception.video_text_overlay import VideoTextOverlay
from src.utils.types import SignDetection


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='output/video_recheck_20261014')
    parser.add_argument('--ranges',nargs='+',default=['27:35','42:49','55:63','145:154','538:546','735:743'])
    args=parser.parse_args()
    out=ROOT/args.output
    out.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'output/video_quality_20261013/full_v4'
    verified=json.loads((cache/'verification.json').read_text(encoding='utf8'))
    if not verified.get('passed'):
        raise RuntimeError('Cached ML video is not verified')
    summary=json.loads((cache/'summary.json').read_text(encoding='utf8'))
    source=Path(summary['source'])
    preds=[json.loads(s)['after'] for s in (cache/'predictions.jsonl').open(encoding='utf8')]
    cap=cv2.VideoCapture(str(source))
    fps=cap.get(cv2.CAP_PROP_FPS)
    step=round(fps/summary['analysis_hz'])
    w,h=summary['output_size']
    writer=cv2.VideoWriter(str(out/'lane_car_scope_preview.mp4'),
                           cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():
        raise RuntimeError('Unable to open MP4 preview output')
    import torch
    surface=RoadSurfaceVeto(ROOT/'models/lane/candidates/segformer_cityscapes_20261011',
          'cuda:0' if torch.cuda.is_available() else 'cpu')
    renderer=VideoTextOverlay(ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18),fps,fps/step)
    stats={'ranges':args.ranges,'source':str(source),'original_model_outputs_reused':True,
           'new_neural_model_inference':False,'ground_mask_recomputed':True,
           'video_fps':fps,'frames':0,'inference_frames':0,'edge_contours_suppressed':0,
           'timestamps_with_edge_suppression':[],'car_context_recomputed':True,
           'legal_certification':False,'independent_lane_ground_truth':False}
    try:
        for item in args.ranges:
            start,end=[float(x) for x in item.split(':')]
            motion=OverlayMotion();lines=[];boxes=[];payload=None
            lo,hi=int(round(start*fps)),int(round(end*fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES,lo)
            for index in range(lo,hi):
                ok,frame=cap.read()
                if not ok:
                    raise RuntimeError(f'Source ended at frame {index}')
                rendered=cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA)
                if index%step==0:
                    payload=copy.deepcopy(preds[index//step])
                    stats['inference_frames']+=1
                    lane=payload['lane']
                    old=lane['display_polylines']
                    if old:
                        small=cv2.resize(frame,(w,h),interpolation=cv2.INTER_AREA)
                        ground,_=surface.scene_probabilities(small)
                        n,ids,cc,_=cv2.connectedComponentsWithStats((ground>.55).astype(np.uint8),8)
                        if n>1: ground=np.where(ids==1+int(np.argmax(cc[1:,cv2.CC_STAT_AREA])),ground,0)
                        kept,removed=suppress_road_edge_contours(old,ground,coordinate_scale=w/frame.shape[1])
                        stats['edge_contours_suppressed']+=removed
                        if removed: stats['timestamps_with_edge_suppression'].append(round(index/fps,3))
                        old_set={tuple(p) for segment in old for p in segment}
                        if any(tuple(p) not in old_set for segment in kept for p in segment):
                            raise RuntimeError('Veto invented lane points')
                    else:kept=[]
                    lane.update(polylines=kept,display_polylines=kept,
                        lane_lines=[[*a,*b] for segment in kept for a,b in zip(segment,segment[1:])])
                    signs=[SignDetection(**s) for s in payload['signs']]
                    for sign in signs:
                        sign.car_context=assess_car(sign,frame,payload['supplementary_candidates'],
                            video=True,camera_view=lane.get('camera_view','unknown'))
                    from dataclasses import asdict
                    payload['signs']=[asdict(sign) for sign in signs]
                    payload['car_summary']=car_summary(signs)
                    sx=w/frame.shape[1];sy=h/frame.shape[0]
                    lines=[[(round(x*sx),round(y*sy)) for x,y in segment] for segment in kept]
                    boxes=[[round(v*(sx if j%2==0 else sy)) for j,v in enumerate(sign.bbox_xyxy)]
                           for sign in signs]
                    motion.reset(rendered,lines,boxes)
                elif payload is not None:
                    lines,boxes=motion.advance(rendered)
                if payload is not None:
                    writer.write(renderer.draw(rendered,payload,lines,boxes,index/fps))
                    stats['frames']+=1
            print('PREVIEW_RANGE_DONE',item,'frames',hi-lo,'total_removed',stats['edge_contours_suppressed'],flush=True)
    finally:
        writer.release()
        cap.release()
    (out/'preview_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf8')
    print('PREVIEW_COMPLETE',json.dumps(stats,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
