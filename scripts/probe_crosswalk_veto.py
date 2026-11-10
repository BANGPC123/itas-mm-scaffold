"""Inspect a rejection-only filter on frozen development camera evidence."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,cv2,numpy as np
from src.perception import camera_lane
from src.perception.crosswalk_veto import crossing_mask,reject_crosswalk_segments
from scripts.compare_lane_paint_evidence import lines


def main():
    ap=argparse.ArgumentParser();ap.add_argument('cache',type=Path);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);rows=[]
    manifest=json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'))
    for frame in manifest['frames']:
        name=frame['name'];data=np.load(args.cache/(name+'.npz'))
        row=json.loads((args.cache/(name+'.json')).read_text(encoding='utf-8'));image=data['image']
        before,_=lines(camera_lane,image,data,row);after=reject_crosswalk_segments(image,before,ground_probability=data['ground'])
        rows.append({'timestamp':row['timestamp'],'before_segments':len(before),'after_segments':len(after),
                     'before_points':sum(map(len,before)),'after_points':sum(map(len,after))})
        if row['timestamp'] in [0,90,400,540,660,740,900]:
            canvases=[]
            for title,result in [('BEFORE',before),('AFTER',after)]:
                view=image.copy()
                for p in result:cv2.polylines(view,[np.asarray(p,np.int32)],False,(0,230,255),2)
                cv2.putText(view,f'{title} {row["timestamp"]}s',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,230,255),2)
                canvases.append(cv2.resize(view,(640,360)))
            cv2.imwrite(str(args.output/(name+'.jpg')),np.hstack(canvases))
            cv2.imwrite(str(args.output/(name+'_mask.png')),crossing_mask(image,data['ground'])*255)
    (args.output/'report.json').write_text(json.dumps({'frames':rows,'human_gt':False,'filter_only_removes_existing_points':True},indent=2),encoding='utf-8')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
