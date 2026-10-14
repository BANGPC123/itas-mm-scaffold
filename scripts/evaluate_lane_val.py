"""Compare the legacy filter and input guard on the official CULane VAL split."""
from pathlib import Path
import sys,json,hashlib,importlib.util,time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np,torch
from src.perception.ufld_v2 import UFLDv2
from src.perception.image_quality import insufficient_information

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('culane_metric',ROOT/'scripts/evaluate_culane.py')
metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
OUT=ROOT/'output/evaluation/improvements_20261009/lane_val'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/raw/culane/val'
    names=[x.split()[0].lstrip('/') for x in (ROOT/'data/raw/culane/test/list/val.txt').read_text().splitlines()]
    assert len(names)==9675 and len(set(names))==9675
    missing=[n for n in names if not (data/n).is_file() or not (data/Path(n).with_suffix('.lines.txt')).is_file()]
    if missing:raise RuntimeError(f'{len(missing)} missing VAL pairs')
    # The checkpoint has seen official VAL during upstream development: do not call it independent holdout.
    manifest={'split':'official CULane VAL','images':names,'holdout':False,'test_used':False,
              'policy':'Select input guard without legacy geometry filtering only if VAL F1 improves',
              'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/perception/ufld_v2.py',ROOT/'src/perception/image_quality.py',ROOT/'scripts/evaluate_lane_val.py']}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    torch.set_num_threads(4);cv2.setNumThreads(1)
    model=UFLDv2(ROOT/'models/lane/culane_res18.pth','cuda:0' if torch.cuda.is_available() else 'cpu',half=True)
    rows=[];started=time.perf_counter();pending=deque();rejected=0
    with (OUT/'frames.jsonl').open('w') as log,ThreadPoolExecutor(max_workers=2) as pool:
        def save(row):
            rows.append(row);log.write(json.dumps(row)+'\n')
            if len(rows)%250==0:
                log.flush();progress={'evaluated':len(rows),'total':len(names),'seconds':time.perf_counter()-started}
                (OUT/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
        for name in names:
            image=cv2.imread(str(data/name))
            if image is None:raise RuntimeError(name)
            rejected+=insufficient_information(image)
            truth=[]
            for line in (data/Path(name).with_suffix('.lines.txt')).read_text().splitlines():
                points=np.fromstring(line,sep=' ')
                if len(points)>=4:truth.append(points.reshape(-1,2).tolist())
            pred=model.detect(image,filter_geometry=False).polylines
            pending.append(pool.submit(metric.evaluate_prediction,name,[],pred,truth))
            if len(pending)>=4:save(pending.popleft().result())
        for future in pending:save(future.result())
    aggregate=metric.aggregate(rows)
    raw=aggregate['raw_adapter']['all'];legacy=aggregate['deployed_filter']['all']
    report={'split':'VAL','n':len(rows),'input_guard_rejections':int(rejected),'guard_without_geometry_filter':raw,'guard_with_legacy_filter':legacy,
            'selected_filter_geometry':not(raw['f1']>legacy['f1']),
            'independent_holdout_verified':False,'seconds':time.perf_counter()-started,
            'metric':'Python port; no TTA; full image; no TEST threshold tuning'}
    (OUT/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
