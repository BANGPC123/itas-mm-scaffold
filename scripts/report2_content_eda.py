"""Describe suspected edge padding on TRAIN/VAL without changing images or boxes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
LOC=ROOT/'data/processed/canonical/itas_mm_v1/localization'
OUT=ROOT/'output/pdf'

def content_rect(gray, threshold):
    h,w=gray.shape
    dark=gray<=threshold
    rows=dark.mean(axis=1)>=.995
    cols=dark.mean(axis=0)>=.995
    def run(v):
        return int(np.flatnonzero(~v)[0]) if (~v).any() else len(v)
    top,bottom,left,right=run(rows),run(rows[::-1]),run(cols),run(cols[::-1])
    if not (min(top,bottom)>=4 and abs(top-bottom)<=max(4,.02*h)):
        top=bottom=0
    if not (min(left,right)>=4 and abs(left-right)<=max(4,.02*w)):
        left=right=0
    if (h-top-bottom)<.25*h or (w-left-right)<.25*w:
        return (0,0,w,h)
    interior=gray[top:h-bottom,left:w-right]
    if interior.mean()<20:
        return (0,0,w,h)
    return left,top,w-right,h-bottom

def luma_metrics(im):
    rgb=np.asarray(im.resize((64,64),Image.Resampling.BOX),dtype=float)
    y=rgb @ np.array([.2126,.7152,.0722])
    return float(y.mean()),float((y<32).mean()),float((y>223).mean())

def inspect(row):
    with Image.open(LOC/row.image_path) as image:
        im=image.convert('RGB')
        gray=np.asarray(im.convert('L'))
        rect=content_rect(gray,12)
        strict=content_rect(gray,3)
        full=(0,0,im.width,im.height)
        a=luma_metrics(im)
        q=luma_metrics(im.crop(rect))
        return [row.sample_id,row.split,row.representative_source,*rect,rect!=full,strict!=full,
                1-(rect[2]-rect[0])*(rect[3]-rect[1])/(im.width*im.height),*a,*q]

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    samples=pd.read_csv(LOC/'samples.csv',keep_default_na=False)
    dev=samples[samples.split.ne('TEST')].sort_values('sample_id')
    fingerprint={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [LOC/'samples.csv',LOC/'boxes.csv',Path(__file__)]}
    summary_path=OUT/'report2_content_summary.json'
    csv_path=OUT/'report2_content_regions_train_val.csv'
    if summary_path.exists() and csv_path.exists():
        prior=json.loads(summary_path.read_text())
        if prior.get('input_sha256')==fingerprint:
            return pd.read_csv(csv_path),prior
    with ThreadPoolExecutor(max_workers=6) as pool:
        content=pd.DataFrame(pool.map(inspect,dev.itertuples()),columns=[
            'sample_id','split','source','left','top','right','bottom','padding_candidate','strict_candidate',
            'padding_fraction','raw_mean_luma','raw_dark_fraction','raw_bright_fraction',
            'content_mean_luma','content_dark_fraction','content_bright_fraction'])
    content.to_csv(csv_path,index=False)
    summary={'input_sha256':fingerprint,'method':{
        'split':'TRAIN/VAL only','padding':'native PIL L <=12 for >=99.5% of a row/column; contiguous outer strips; paired sides >=4 px; symmetry tolerance max(4 px, 2% dimension); retain >=25% each dimension; interior mean L >=20',
        'sensitivity':'same rule with L <=3','brightness':'raw and inferred content independently BOX-resized to 64x64; encoded RGB luma .2126R+.7152G+.0722B; dark <32, bright >223',
        'limitation':'Heuristic candidate regions, not verified letterbox ground truth; originals and frozen annotations unchanged.'},'splits':{}}
    for sp,g in content.groupby('split'):
        d={'n':len(g),'padding_candidates':int(g.padding_candidate.sum()),'strict_candidates':int(g.strict_candidate.sum()),'mean_padding_fraction':float(g.padding_fraction.mean())}
        for domain in ['raw','content']:
            v=g[f'{domain}_mean_luma']
            d[domain]={'p10':float(v.quantile(.1)),'median':float(v.median()),'p90':float(v.quantile(.9)),
                'mean_dark_fraction':float(g[f'{domain}_dark_fraction'].mean()),'mean_bright_fraction':float(g[f'{domain}_bright_fraction'].mean())}
        summary['splits'][sp]=d
    b=pd.read_csv(LOC/'boxes.csv',keep_default_na=False).query("split != 'TEST'")
    merged=b.merge(content,on=['sample_id','split']).merge(dev[['sample_id','image_width_px','image_height_px']],on='sample_id')
    xy=np.array([list(map(float,json.loads(t))) for t in merged.canonical_xyxy_normalized_json])
    dims=merged[['image_width_px','image_height_px']].to_numpy()
    absolute=xy*np.tile(dims,(1,2))
    lo=merged[['left','top']].to_numpy();hi=merged[['right','bottom']].to_numpy()
    extent=hi-lo
    clipped_lo=np.maximum(absolute[:,:2],lo);clipped_hi=np.minimum(absolute[:,2:],hi)
    raw_area=np.prod(absolute[:,2:]-absolute[:,:2],axis=1)/np.prod(dims,axis=1)*100
    content_area=np.prod(np.maximum(clipped_hi-clipped_lo,0),axis=1)/np.prod(extent,axis=1)*100
    centers=(absolute[:,:2]+absolute[:,2:])/2
    normalized=(centers-lo)/extent
    inside=((normalized>=0)&(normalized<=1)).all(axis=1)
    geom=pd.DataFrame({'canonical_box_id':merged.canonical_box_id,'sample_id':merged.sample_id,'split':merged.split,
        'raw_area_pct':raw_area,'content_clipped_area_pct':content_area,'raw_center_x':xy[:,[0,2]].mean(axis=1),
        'raw_center_y':xy[:,[1,3]].mean(axis=1),'content_center_x':normalized[:,0],'content_center_y':normalized[:,1],
        'center_inside_content':inside,'crosses_content_edge':((absolute[:,:2]<lo)|(absolute[:,2:]>hi)).any(axis=1)})
    geom.to_csv(OUT/'report2_content_geometry_train_val.csv',index=False)
    summary['geometry']={'n':len(geom),'centers_outside_inferred_content':int((~inside).sum()),'boxes_crossing_content_edge':int(geom.crosses_content_edge.sum()),
        'raw_area_pct_quantiles':{str(k):float(v) for k,v in pd.Series(raw_area).quantile([.1,.5,.9]).items()},
        'content_area_pct_quantiles':{str(k):float(v) for k,v in pd.Series(content_area).quantile([.1,.5,.9]).items()}}
    summary_path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return content,summary

if __name__=='__main__':
    _,summary=run()
    print(json.dumps(summary,indent=2))
