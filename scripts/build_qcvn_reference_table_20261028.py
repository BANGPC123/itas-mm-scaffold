"""Crop QCVN 41:2024 sample sign figures (caption 'Hình B.30 - Biển số P.130') into a reference table.

Reads the SHA-tracked official PDF only. Output (images and index) is local and ignored by Git.
"""
from __future__ import annotations
import argparse,hashlib,json,re,sys,unicodedata
from pathlib import Path
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PDF=ROOT/'data/regulations/official_20261011/QCVN41_2024.pdf'
OUT=ROOT/'output/kiet_wp_20261028/recognizer_v2/qcvn_refs'
SCALE=3.0
CODE=r'[a-z]{1,2}\.\d{3}(?:[a-z](?!\.\d))?'
# The PDF prints 'Biển số 219' (no series letter) in Hình C.21, so the series prefix is optional here
# and inferred from the appendix letter (B=P, C=W, D=R, E=I, F=S).
SERIES_OF_APPENDIX={'B':'P','C':'W','D':'R','E':'I','F':'S'}
CAPTION=re.compile(r'hinh([a-h])\.(\d{1,3}[a-z]?)[-–—]bienso((?:[a-z]{1,2}\.)?\d{3}(?:[a-z](?!\.\d))?)')
HEADING=re.compile(r'([a-h])\.(\d{1,3}[a-z]?)bienso('+CODE+r')(?:\(([a-z,]+)\))?["“]')
MENTION=re.compile(r'bienso('+CODE+')["“]')
CODE_IN_LABEL=re.compile(r'bienso('+CODE+')')


def fold(ch):
    """ASCII-fold one character: drop Vietnamese diacritics, lowercase."""
    ch=ch.replace('đ','d').replace('Đ','D')
    return ''.join(c for c in unicodedata.normalize('NFD',ch) if not unicodedata.combining(c)).lower()


def compact(text):
    chars=[];index=[]
    for i,ch in enumerate(text):
        if ch.isspace() or ch=='￾':continue
        for f in fold(ch):chars.append(f);index.append(i)
    return ''.join(chars),index


def normalize_code(code,appendix=None):
    if '.' not in code:
        return SERIES_OF_APPENDIX[appendix.upper()]+'.'+code
    prefix,rest=code.split('.',1)
    return prefix.upper()+'.'+rest


def clean_text(text):
    return re.sub(r'\s+',' ',text.replace('￾',' ')).strip()


def page_rows(images,tol=6):
    """Group embedded images that sit on the same row (shared vertical span)."""
    rows=[]
    for box in sorted(images,key=lambda b:-b[3]):
        for row in rows:
            if min(box[3],row['top'])-max(box[1],row['bottom'])>tol:
                row['boxes'].append(box);row['top']=max(row['top'],box[3]);row['bottom']=min(row['bottom'],box[1]);break
        else:rows.append(dict(boxes=[box],top=box[3],bottom=box[1]))
    for row in rows:
        row['left']=min(b[0] for b in row['boxes']);row['right']=max(b[2] for b in row['boxes'])
    return rows


def page_data(pdf,number):
    page=pdf[number];textpage=page.get_textpage();text=textpage.get_text_range()
    images=[tuple(o.get_bounds()) for o in page.get_objects(filter=(pdfium_raw.FPDF_PAGEOBJ_IMAGE,),max_depth=3)]
    return page,textpage,text,images


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args()
    out=args.output;(out/'figures').mkdir(parents=True,exist_ok=True)
    pdf=pdfium.PdfDocument(str(PDF));pages={}
    captions=[];headings={};mentions={}
    for number in range(len(pdf)):
        page,textpage,text,images=page_data(pdf,number);pages[number]=(page,textpage,text,images)
        flat,index=compact(text)
        for m in CAPTION.finditer(flat):
            start=index[m.start()];box=textpage.get_charbox(start)
            captions.append(dict(page=number,figure=m.group(1).upper()+'.'+m.group(2),code=normalize_code(m.group(3),m.group(1)),
                top=box[3],bottom=box[1],left=box[0],right=box[2],text_index=start,flat_end=m.end()))
        def quoted_name(m):
            quote_at=index[m.end()-1]
            for k in range(quote_at+1,min(len(text),quote_at+160)):
                if text[k] in '"”':return clean_text(text[quote_at+1:k])
            return None
        for m in HEADING.finditer(flat):
            name=quoted_name(m)
            if not name:continue
            code=normalize_code(m.group(3));item=m.group(1).upper()+'.'+m.group(2)
            headings.setdefault(code,dict(name_vi=name,item=item,page=number+1))
            # 'Biển số W.203 (a,b,c)' names the whole group; each variant keeps the group heading.
            for letter in (m.group(4) or '').split(','):
                if letter:headings.setdefault(code+letter,dict(name_vi=name,item=item,page=number+1,group_heading=True))
        for m in MENTION.finditer(flat):
            name=quoted_name(m)
            if name:mentions.setdefault(normalize_code(m.group(1)),dict(name_vi=name,item=None,page=number+1,mention_only=True))
    for code,row in mentions.items():headings.setdefault(code,row)
    entries={};missing=[];claimed=set()
    for cap in sorted(captions,key=lambda c:(c['page'],-c['top'])):
        number=cap['page'];page,textpage,text,images=pages[number]
        source_page=number
        height=page.get_size()[1]
        # Rows of embedded images between the previous caption (above) and this caption belong to this figure.
        ceiling=min([c['bottom'] for c in captions if c['page']==number and c['top']>cap['top']+1],default=height)
        rows=[r for r in page_rows(images) if (number,round(r['top']),round(r['bottom'])) not in claimed
              and r['bottom']>=cap['bottom']-2 and r['top']<=ceiling+2]
        if not rows and number>0:
            # Caption starts the page, figure ended the previous one (below that page's last caption).
            source_page=number-1;height=pages[source_page][0].get_size()[1]
            last_caption=min([c['bottom'] for c in captions if c['page']==source_page],default=height)
            rows=[r for r in page_rows(pages[source_page][3]) if (source_page,round(r['top']),round(r['bottom'])) not in claimed
                  and r['top']<=last_caption+2]
        if not rows:
            missing.append(dict(figure=cap['figure'],code=cap['code'],page=number+1,reason='no embedded image row above caption'));continue
        for r in rows:claimed.add((source_page,round(r['top']),round(r['bottom'])))
        row=dict(left=min(r['left'] for r in rows),right=max(r['right'] for r in rows),top=max(r['top'] for r in rows),bottom=min(r['bottom'] for r in rows))
        src_page=pages[source_page][0];src_textpage=pages[source_page][1];width,height=src_page.get_size()
        label_bottom=cap['bottom']-3 if source_page==number else 20
        left=max(0,row['left']-12);right=min(width,row['right']+12);top=min(height,row['top']+8);bottom=max(0,label_bottom)
        region=src_textpage.get_text_bounded(left,bottom,right,top)
        flat,_=compact(region)
        codes=sorted({normalize_code(c) for c in CODE_IN_LABEL.findall(flat)}|{cap['code']})
        bitmap=src_page.render(scale=SCALE).to_pil()
        box=(round(left*SCALE),round((height-top)*SCALE),round(right*SCALE),round((height-bottom)*SCALE))
        crop=bitmap.crop(box)
        name=f"{cap['figure']}_{cap['code']}.png".replace('/','_');crop.save(out/'figures'/name)
        entry=dict(figure=cap['figure'],caption_code=cap['code'],codes=codes,page_caption=number+1,page_image=source_page+1,
            image=f'figures/{name}',size=list(crop.size))
        for code in codes:entries.setdefault(code,[]).append(entry)
    index=dict(pdf=str(PDF.relative_to(ROOT)),pdf_sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(),scale=SCALE,
        codes={code:dict(figures=[e['image'] for e in rows],figure_ids=sorted({e['figure'] for e in rows}),
            appendix_heading=headings.get(code)) for code,rows in sorted(entries.items())},
        headings=headings,captions_found=len(captions),figures_cropped=len({e['image'] for rows in entries.values() for e in rows}),
        missing_figures=missing)
    (out/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(captions=len(captions),codes=len(entries),cropped=index['figures_cropped'],missing=len(missing),headings=len(headings)),ensure_ascii=False))


if __name__=='__main__':main()
