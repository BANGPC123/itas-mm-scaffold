"""Fetch pinned official Vietnamese legal PDFs with traceable provenance.

The files are references for an offline CAPSTONE RAG demo, not a claim that
all amendments/current provisions were reviewed by counsel.
"""
from pathlib import Path
import requests,hashlib,datetime,json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/regulations/official_20261011'
if OUT.exists() and (OUT/'sources.json').exists():
    raise FileExistsError('Do not silently overwrite legal corpus')
OUT.mkdir(parents=True,exist_ok=True)
urls=[
('36_2024_QH15_part1.pdf','https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/9/36-2024-qh15.pdf','36/2024/QH15'),
('36_2024_QH15_part2.pdf','https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/9/36-2024-qh15_tiep.pdf','36/2024/QH15'),
('168_2024_NDCP.pdf','https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/01/168-nd-cp.signed.pdf','168/2024/NĐ-CP'),
('QCVN41_2024.pdf','https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf','QCVN 41:2024/BGTVT')
]
rows=[]
for name,url,doc in urls:
    dest=OUT/name
    h=hashlib.sha256()
    if dest.is_file():
        stream=dest.read_bytes();h.update(stream);length=len(stream)
        if not stream.startswith(b'%PDF'):raise RuntimeError('Invalid cached PDF')
    else:
        resp=requests.get(url,timeout=(18,180),stream=True)
        resp.raise_for_status()
        tmp=dest.with_suffix('.partial')
        length=0
        with tmp.open('wb') as fp:
            for chunk in resp.iter_content(1024*512):
                if not chunk:continue
                h.update(chunk);length+=len(chunk);fp.write(chunk)
        if tmp.read_bytes()[:4]!=b'%PDF':raise RuntimeError('Invalid PDF header '+name)
        tmp.rename(dest)
    entry={'file':name,'document':doc,'source_url':url,'bytes':length,
        'sha256':h.hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    rows.append(entry);print('OFFICIAL_PDF',json.dumps(entry,ensure_ascii=False),flush=True)
(OUT/'sources.json').write_text(json.dumps({'official_documents':rows,
       'legal_effectiveness_reviewed':False,
       'note':'Source documents are official snapshots; consolidated latest amendments and per-case legal applicability NOT independently audited.'},ensure_ascii=False,indent=2),encoding='utf-8')
print('OFFICIAL_LEGAL_DOWNLOAD_DONE',len(rows),flush=True)
