"""Offline searchable PDF-page corpus, with exact official PDF provenance.

Retains source URL and page; no legal interpretation is generated. A legal
review and any later amendments remain outside this automatic extraction.
"""
from pathlib import Path
from pypdf import PdfReader
from collections import Counter
import hashlib,json,re,datetime
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/regulations/official_20261011'
OUT=ROOT/'data/processed/legal_reference/official_corpus_20261011'
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'manifest.json').is_file():raise FileExistsError('No overwriting an existing legal corpus')
sources=json.loads((SOURCE/'sources.json').read_text(encoding='utf-8'))['official_documents']
chunks=[];stats=[]
for d in sources:
    file=SOURCE/d['file']
    if hashlib.sha256(file.read_bytes()).hexdigest()!=d['sha256']:
        raise RuntimeError('Official PDF source hash mismatch')
    reader=PdfReader(str(file),strict=False)
    pages_with_text=0
    for idx,page in enumerate(reader.pages):
        try: text=page.extract_text(extraction_mode='plain') or ''
        except Exception: text=''
        text=re.sub(r'[ \t]+',' ',text).strip()
        text=re.sub(r'\n{3,}','\n\n',text)
        if len(text)<90:continue
        pages_with_text+=1
        # Small chunks; never blend legislation across document/pages.
        lines=text.splitlines()
        piece=[];size=0;index=0
        def push():
            global chunks
            nonlocal_vars=None
        for line in lines:
            if len(line)>2300:
                blocks=[line[i:i+1500] for i in range(0,len(line),1500)]
            else:blocks=[line]
            for segment in blocks:
                if size+len(segment)>1050 and piece:
                    body='\n'.join(piece)
                    chunks.append({'id':f"{d['file']}:p{idx+1}:c{index}",
                        'document':d['document'],'file':d['file'],
                        'page':idx+1,'text':body,
                        'source_url':d['source_url'],'source_sha256':d['sha256']})
                    index+=1
                    piece=[];size=0
                piece.append(segment);size+=len(segment)
        if piece:
            chunks.append({'id':f"{d['file']}:p{idx+1}:c{index}",
                'document':d['document'],'file':d['file'],'page':idx+1,
                'text':'\n'.join(piece),
                'source_url':d['source_url'],'source_sha256':d['sha256']})
    stats.append({'file':d['file'],'document':d['document'],
        'total_pdf_pages':len(reader.pages),'pages_with_extracted_text':pages_with_text})
    print('PDF_EXTRACTED',json.dumps(stats[-1],ensure_ascii=False),flush=True)
if not chunks:raise RuntimeError('No real source texts indexed')
with (OUT/'chunks.jsonl').open('w',encoding='utf-8') as f:
    for chunk in chunks:f.write(json.dumps(chunk,ensure_ascii=False)+'\n')
report={'source_count':len(sources),'chunks':len(chunks),'source_stats':stats,
        'source_manifest':str(SOURCE/'sources.json'),
        'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'official_document_snapshot':True,'all_current_amendments_checked':False,
        'juridical_scope_of_each_sign_verified':False,
        'status':'SOURCE_INDEXED_OFFLINE_DEMO_NOT_CURATED_LEGAL_ADVICE'}
(OUT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('CORPUS_READY',json.dumps(report,ensure_ascii=False),flush=True)
