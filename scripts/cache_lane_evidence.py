"""Freeze model evidence on development frames for road-paint ablation.

Cached probabilities are diagnostic input, not lane ground truth. Final video
must still run the selected pipeline on the source frames.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,time
import cv2,numpy as np,torch
from src.utils.config import load_config
from src.perception.lane_detector import LaneDetector
from src.perception.camera_lane import CameraLaneVisual,camera_view_from_ground
from src.perception.yolopv2_lane import traces_from_mask


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('source',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--times',nargs='+',type=float,default=[0,90,149.333,207.333,270,360,400,480,540,600,660,710,740,770,800,840,870,900,915,930,945,960,990,1030,1073,1130,1180])
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4);cv2.setNumThreads(2)
    from threadpoolctl import threadpool_limits
    limit=threadpool_limits(limits=1,user_api='blas')
    detector=LaneDetector(load_config('perception'))
    camera=CameraLaneVisual(detector.learned,device='cuda:0' if torch.cuda.is_available() else 'cpu')
    cap=cv2.VideoCapture(str(args.source));rows=[]
    for t in args.times:
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,image=cap.read()
        if not ok:raise RuntimeError('Source decode failed')
        h,w=image.shape[:2]
        if w>1280:image=cv2.resize(image,(1280,round(h*1280/w)),interpolation=cv2.INTER_AREA)
        h,w=image.shape[:2];start=time.perf_counter()
        baseline=detector.detect(image)
        road,paint=camera.scene.scene_probabilities(image)
        ground,sidewalk=camera.surface.scene_probabilities(image)
        view=camera_view_from_ground(ground,image)
        n,ids,stats,_=cv2.connectedComponentsWithStats((ground>.55).astype(np.uint8),8)
        if n>1:
            main=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));ground=np.where(ids==main,ground,0)
        yy,xx=np.where(ground[:,round(w*.15):round(w*.85)]>.65)
        seeds=list(baseline.polylines)
        if len(yy)>=h*w*.01:
            bottom=float(np.clip(np.quantile(yy,.99)/h+.01,.5,.97));crop=image[:round(h*bottom)]
            if bottom<.94 and camera.ufld is not None:seeds+=camera.ufld.detect(crop).polylines
            if camera.clr is not None:seeds+=camera.clr.detect(crop).polylines
        seeds+=traces_from_mask((paint>.56).astype(np.uint8),min_span_ratio=.09)
        boxes=camera.occlusion.get_boxes(image)
        boxes=[b for b in boxes if not ((b[2]-b[0])*(b[3]-b[1])>h*w*.40 and b[3]>h*.94)]
        name=f'{round(t*1000):07d}'
        cv2.imwrite(str(args.output/(name+'.jpg')),image,[cv2.IMWRITE_JPEG_QUALITY,97])
        np.savez_compressed(args.output/(name+'.npz'),road=road,paint=paint,ground=ground,sidewalk=sidewalk,image=image)
        row={'name':name,'timestamp':t,'seeds':seeds,'boxes':boxes,'camera_view':view,'road_visible':len(yy)>=h*w*.01,'ms':(time.perf_counter()-start)*1000}
        (args.output/(name+'.json')).write_text(json.dumps(row),encoding='utf-8');rows.append({k:v for k,v in row.items() if k not in ('seeds','boxes')})
        print('CACHE',t,round(row['ms']),flush=True)
    cap.release()
    with args.source.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    report={'source':str(args.source),'source_sha256':digest,'frames':rows,'human_verified':False,'independent_test':False,
            'camera_source_sha256':hashlib.sha256((ROOT/'src/perception/camera_lane.py').read_bytes()).hexdigest()}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
