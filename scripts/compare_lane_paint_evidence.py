"""Compare frozen development evidence; this is visual QA, never accuracy GT."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,subprocess,types
import cv2,numpy as np
from src.perception import camera_lane as current
from src.perception.road_surface_veto import filter_sidewalk_overlap


def lines(module,image,data,row):
    if not row['road_visible']:return [],np.zeros(image.shape[:2],np.uint8)
    evidence=module.paint_evidence(image,data['road'],row['boxes'],data['paint'],data['ground'])
    seeds=module.observed_paint_seeds(evidence)+row['seeds']
    result=module.refine_candidates(seeds,evidence,image.shape)
    result,_=filter_sidewalk_overlap(result,data['sidewalk'])
    return module.visible_paint_segments(result,evidence),evidence


def main():
    ap=argparse.ArgumentParser();ap.add_argument('cache',type=Path);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--baseline',default='b2634dd3430714c4adfad5a54b59030aa33cd248');args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    old=types.ModuleType('old_camera_lane')
    source=subprocess.check_output(['git','show',args.baseline+':src/perception/camera_lane.py'],cwd=ROOT)
    exec(compile(source,'old_camera_lane.py','exec'),old.__dict__)
    manifest=json.loads((args.cache/'manifest.json').read_text(encoding='utf-8'));rows=[];gallery=[]
    for frame in manifest['frames']:
        name=frame['name'];data=np.load(args.cache/(name+'.npz'))
        row=json.loads((args.cache/(name+'.json')).read_text(encoding='utf-8'));image=data['image']
        before,bmask=lines(old,image,data,row);after,amask=lines(current,image,data,row)
        canvases=[]
        for title,result,mask in [('BEFORE',before,bmask),('AFTER',after,amask)]:
            view=image.copy()
            for line in result:cv2.polylines(view,[np.asarray(line,np.int32)],False,(0,230,255),2)
            cv2.putText(view,f'{title} {row["timestamp"]}s: {len(result)} paint pieces',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,230,255),2)
            cv2.imwrite(str(args.output/(name+'_'+title.lower()+'.jpg')),view)
            cv2.imwrite(str(args.output/(name+'_'+title.lower()+'_mask.png')),mask*255)
            canvases.append(cv2.resize(view,(640,360)))
        rows.append({'timestamp':row['timestamp'],'before_segments':len(before),'after_segments':len(after),
                     'before_points':sum(map(len,before)),'after_points':sum(map(len,after)),
                     'before_pixels':int(bmask.sum()),'after_pixels':int(amask.sum()),'camera_view':row['camera_view']})
        if row['timestamp'] in [0,400,540,740,900,915,930,960,1073]:gallery.append(np.hstack(canvases))
    cv2.imwrite(str(args.output/'comparison.jpg'),np.vstack(gallery))
    report={'baseline':args.baseline,'source_sha256':manifest['source_sha256'],'frames':rows,
            'human_gt':False,'independent_test':False,'accuracy_claim':False}
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
