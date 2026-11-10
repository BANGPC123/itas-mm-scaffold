"""Contact sheets for AI review of boxes outside the V1.2 ontology.

`refs`   : QCVN reference figures (from build_qcvn_reference_table_20261028.py) tiled with code and name.
`crops`  : crops of EXCLUDED_OUT_OF_ONTOLOGY_V1_2 boxes with a 15% margin, numbered per sheet.
           No recognizer, detector or earlier-label column is read or drawn.
Outputs stay under output/ (ignored by Git).
"""
from __future__ import annotations
import argparse,json,math,random,re,sys
from pathlib import Path
import cv2,numpy as np,pandas as pd
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'output/kiet_wp_20261028/recognizer_v2'
LOC=ROOT/'data/processed/canonical/itas_mm_v1/localization'
LABELS=ROOT/'data/audit/semantic/final_v1_2/semantic_labels_v1_2.csv'
OUT_OF_ONTOLOGY='EXCLUDED_OUT_OF_ONTOLOGY_V1_2'
FONT='C:/Windows/Fonts/arial.ttf'
SERIES={'P':0,'DP':1,'R':2,'W':3,'I':4,'S':5}


def font(size):
    try:return ImageFont.truetype(FONT,size)
    except OSError:return ImageFont.load_default()


def natural(code):
    m=re.match(r'([A-Z]+)\.(\d+)([a-z]*)',code)
    return (SERIES.get(m.group(1),9),int(m.group(2)),m.group(3)) if m else (9,0,code)


def ref_sheets(per_sheet=12,cols=3,tile=(480,360)):
    refs=OUT/'qcvn_refs';index=json.loads((refs/'index.json').read_text(encoding='utf-8'))
    figures={}
    for code,row in index['codes'].items():
        for image in row['figures']:figures.setdefault(image,set()).add(code)
    def key(image):
        stem=Path(image).stem;fig,caption=stem.split('_',1);return natural(caption)+(fig,)
    ordered=sorted(figures,key=key);sheets=[];(OUT/'ref_sheets').mkdir(parents=True,exist_ok=True)
    for sheet_no,start in enumerate(range(0,len(ordered),per_sheet),1):
        rows=math.ceil(min(per_sheet,len(ordered)-start)/cols);canvas=Image.new('RGB',(cols*tile[0],rows*tile[1]),'white');d=ImageDraw.Draw(canvas)
        for n,image in enumerate(ordered[start:start+per_sheet]):
            fig,caption=Path(image).stem.split('_',1);x=(n%cols)*tile[0];y=(n//cols)*tile[1]
            name=(index['headings'].get(caption) or {}).get('name_vi','')
            d.rectangle([x,y,x+tile[0]-1,y+tile[1]-1],outline=(120,120,120))
            d.rectangle([x,y,x+tile[0]-1,y+28],fill=(235,240,250))
            d.text((x+6,y+4),f'{fig}  {caption}  {name}'[:62],fill=(0,0,0),font=font(15))
            picture=Image.open(refs/image).convert('RGB');scale=min((tile[0]-8)/picture.width,(tile[1]-36)/picture.height,1.0)
            picture=picture.resize((max(1,round(picture.width*scale)),max(1,round(picture.height*scale))),Image.LANCZOS)
            canvas.paste(picture,(x+(tile[0]-picture.width)//2,y+32))
        path=OUT/'ref_sheets'/f'ref_{sheet_no:02d}.png';canvas.save(path)
        sheets.append(dict(sheet=path.name,figures=[Path(i).stem for i in ordered[start:start+per_sheet]]))
    (OUT/'ref_sheets/index.json').write_text(json.dumps(sheets,ensure_ascii=False,indent=1),encoding='utf-8')
    print('ref sheets',len(sheets),'figures',len(ordered))


def crop_with_margin(row,margin=.15):
    image=cv2.imread(str(LOC/row.image_path))
    if image is None:return None
    h,w=image.shape[:2];x1,y1,x2,y2=json.loads(row.bbox_xyxy);bw=x2-x1;bh=y2-y1
    a=max(0,math.floor(x1-margin*bw));b=max(0,math.floor(y1-margin*bh));c=min(w,math.ceil(x2+margin*bw));d=min(h,math.ceil(y2+margin*bh))
    return image[b:d,a:c],(x1-a,y1-b,x2-a,y2-b)


def crop_sheets(name,seed,per_sheet=20,cols=5,tile=(280,280)):
    df=pd.read_csv(LABELS,keep_default_na=False)
    df=df[df.semantic_final_disposition==OUT_OF_ONTOLOGY].copy()
    assert len(df)==1531,len(df)
    if seed is None:df=df.sort_values(['image_path','canonical_box_id'])
    else:
        ids=sorted(df.canonical_box_id);random.Random(seed).shuffle(ids);order={b:i for i,b in enumerate(ids)};df=df.assign(_o=df.canonical_box_id.map(order)).sort_values('_o')
    folder=OUT/f'sheets_{name}';folder.mkdir(parents=True,exist_ok=True);manifest=[]
    rows=list(df.itertuples())
    for sheet_no,start in enumerate(range(0,len(rows),per_sheet),1):
        chunk=rows[start:start+per_sheet];nrows=math.ceil(len(chunk)/cols);canvas=Image.new('RGB',(cols*tile[0],nrows*tile[1]),(255,255,255));d=ImageDraw.Draw(canvas)
        entries=[]
        for n,row in enumerate(chunk,1):
            x=((n-1)%cols)*tile[0];y=((n-1)//cols)*tile[1]
            d.rectangle([x,y,x+tile[0]-1,y+tile[1]-1],outline=(150,150,150))
            result=crop_with_margin(row)
            if result is not None:
                crop,_=result;rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
                scale=min((tile[0]-8)/w,(tile[1]-30)/h);interp=cv2.INTER_CUBIC if scale>1 else cv2.INTER_AREA
                rgb=cv2.resize(rgb,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=interp)
                canvas.paste(Image.fromarray(rgb),(x+(tile[0]-rgb.shape[1])//2,y+26+(tile[1]-30-rgb.shape[0])//2))
            d.rectangle([x,y,x+tile[0]-1,y+22],fill=(30,30,30))
            d.text((x+6,y+3),f'{n}',fill=(255,255,0),font=font(16))
            d.text((x+tile[0]-120,y+4),f'short {float(row.short_side_px):.0f}px',fill=(200,200,200),font=font(13))
            entries.append(dict(idx=n,canonical_box_id=row.canonical_box_id,split=row.split))
        path=folder/f'{name}_{sheet_no:03d}.png';canvas.save(path);manifest.append(dict(sheet=sheet_no,file=path.name,entries=entries))
    (folder/'manifest.json').write_text(json.dumps(dict(name=name,seed=seed,per_sheet=per_sheet,sheets=manifest),indent=1),encoding='utf-8')
    print(name,'sheets',len(manifest),'boxes',len(rows))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['refs','crops']);ap.add_argument('--name',default='A')
    ap.add_argument('--seed',type=int);args=ap.parse_args()
    ref_sheets() if args.command=='refs' else crop_sheets(args.name,args.seed)


if __name__=='__main__':main()
