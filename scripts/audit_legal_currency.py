"""Apply source-pinned amendments and audit evidence, without legal certification.

OCR content always keeps its extraction provenance; never consolidated into
original enactments and never changes validated_corpus to true.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,datetime,json,hashlib,requests,shutil
from src.reasoning.document_loader import load_regulation_corpus,chunk_regulation_text
ROOT=Path(__file__).resolve().parents[1]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');args=ap.parse_args()
 path=ROOT/'data/regulations/corpus.json';corpus=load_regulation_corpus(path)
 updates=json.loads((ROOT/'data/regulations/legal_updates_20261011.json').read_text(encoding='utf-8'))
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();integrity=[]
 for s in corpus['sources']:
  p=path.parent/s['raw_file'];data=p.read_bytes() if p.exists() else None
  actual=hashlib.sha256(data).hexdigest() if data is not None else None
  restored=False
  if actual!=s['sha256'] and data is not None and s['source_kind']=='official_html':
   candidate=data.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
   if hashlib.sha256(candidate).hexdigest()==s['sha256'] and args.apply:
    p.write_bytes(candidate);actual=s['sha256'];restored=True
  integrity.append({'document_id':s['document_id'],'raw_sha256_matches':actual==s['sha256'],'restored_original_CRLF_bytes':restored})
 if not all(r['raw_sha256_matches'] for r in integrity):raise ValueError('Existing source integrity failure')
 for u in updates['documents']:
  p=path.parent/u['raw_file'];p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists():
   response=requests.get(u['pdf_url'],timeout=(30,90));response.raise_for_status()
   if hashlib.sha256(response.content).hexdigest()!=u['sha256']:raise ValueError('Source changed: '+u['document_id'])
   p.write_bytes(response.content)
  if hashlib.sha256(p.read_bytes()).hexdigest()!=u['sha256']:raise ValueError('Bad update source')
  if args.apply and u['document_id'] not in {s['document_id'] for s in corpus['sources']}:
   text='\n\n'.join('[Trang PDF '+str(i+1)+'; OCR tự động]\n'+t for i,t in enumerate(u['ocr_pages']))
   chunks=chunk_regulation_text(u['document_id'],text)
   source={k:u[k] for k in ['document_id','document_number','title','source_url','raw_file','sha256','effective_from']}
   source.update({'source_kind':'official_pdf_ocr','extraction_method':'tesseract5_vie_220dpi_machine_ocr_unverified',
     'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'retrieved_at':updates['generated_at'],
     'chunk_count':len(chunks),'consolidated_text':False,'ocr_text_human_verified':False})
   corpus['sources'].append(source);corpus['chunks'].extend(chunks)
 if args.apply:
  for s in corpus['sources']:
   if s['document_id'] in updates['amendment_links']:
    s['amendments']=updates['amendment_links'][s['document_id']];s['consolidated_text']=False
  corpus['manifest'].update({'source_count':len(corpus['sources']),'chunk_count':len(corpus['chunks']),
    'generated_at':updates['generated_at'],'currency_review_at':updates['as_of_date'],'currency_review_complete':False})
  backup=ROOT/'output/evaluation/auto_legal_20261011/original_corpus.json';backup.parent.mkdir(parents=True,exist_ok=True)
  if not backup.exists():shutil.copy2(path,backup)
  staged=path.with_suffix('.current.tmp');staged.write_text(json.dumps(corpus,ensure_ascii=False,indent=2),encoding='utf-8')
  load_regulation_corpus(staged);staged.replace(path)
 report={'as_of_date':updates['as_of_date'],'checked_at':now,'source_integrity':integrity,
  'known_amendments':[{'document':u['document_number'],'effective_from':u['effective_from'],'source_url':u['source_url'],'sha256':u['sha256'],'ocr_pages':len(u['ocr_pages'])} for u in updates['documents']],
  'applied':args.apply,'sources':len(corpus['sources']),'chunks':len(corpus['chunks']),
  'legal_currency_verified':False,'all_amendments_exhaustively_checked':False,'consolidated_text':False,
  'validated_corpus_changed':False,'blockers':['OCR text requires page-level verification','not an exhaustive amendment/exception review','no independently verified spatial legal applicability corpus']}
 out=ROOT/'output/evaluation/auto_legal_20261011/audit.json';out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=True),flush=True)
if __name__=='__main__':main()
