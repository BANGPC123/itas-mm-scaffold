"""Save a corrected Report 1 copy, preserving existing OOXML formatting."""
from pathlib import Path
import json
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('D:/DSP/DSP391m_FA26_Group2_Report1.docx')
TARGET=ROOT/'output/docx/DSP391m_FA26_Group2_Report1_QCVN2024.docx'
TARGET.parent.mkdir(parents=True,exist_ok=True)
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
edits=[]
with ZipFile(SOURCE) as zin,ZipFile(TARGET,'w',ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data=zin.read(item.filename)
        if item.filename.startswith('word/') and item.filename.endswith('.xml'):
            tree=etree.fromstring(data)
            changed=False
            for p in tree.xpath('//w:p',namespaces=NS):
                texts=p.xpath('.//w:t',namespaces=NS)
                full=''.join(t.text or '' for t in texts)
                old='QCVN 41:2019/BGTVT'
                # Work backwards so character positions stay valid for repeated mentions.
                starts=[];idx=full.find(old)
                while idx!=-1:
                    starts.append(idx);idx=full.find(old,idx+len(old))
                for start in reversed(starts):
                    end=start+len(old);pos=0
                    for t in texts:
                        value=t.text or '';a=pos;pos+=len(value)
                        if pos<=start or a>=end:continue
                        first=max(0,start-a);last=min(len(value),end-a)
                        replacement='QCVN 41:2024/BGTVT' if a<=start<pos else ''
                        t.text=value[:first]+replacement+value[last:]
                    changed=True
                if starts:
                    revised=''.join(t.text or '' for t in texts)
                    if revised.lstrip().startswith('[12]'):
                        texts[-1].text=(texts[-1].text or '')+' Ban hành kèm Thông tư 51/2024/TT-BGTVT ngày 15/11/2024, có hiệu lực từ 01/01/2025; thay thế bản 2019. https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt.pdf'
                    edits.append({'part':item.filename,'before':full,'after':''.join(t.text or '' for t in texts)})
            if changed:data=etree.tostring(tree,xml_declaration=True,encoding='UTF-8',standalone=True)
        zout.writestr(item,data)
assert edits,'No QCVN references found'
(TARGET.parent/'report1_qcvn_changes.json').write_text(json.dumps(edits,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'{len(edits)} paragraphs updated; {TARGET}')
