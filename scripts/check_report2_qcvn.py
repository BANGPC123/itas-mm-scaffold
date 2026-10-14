"""Verify pinned QCVN version and code-token presence, not visual label correctness."""
from pathlib import Path
import hashlib,json,re
import pandas as pd
import pypdfium2 as pdfium

ROOT=Path(__file__).resolve().parents[1]
pdf_path=ROOT/'data/audit/g1a_v2/references/QCVN_41-2024_BGTVT_Annex_51-2024-TT-BGTVT.pdf'
expected='B6B99D0A0AEB13FE79072831857ECE15BB254D8FB7817E80E4A2739A0D2145CE'
actual=hashlib.sha256(pdf_path.read_bytes()).hexdigest().upper()
assert actual==expected
doc=pdfium.PdfDocument(str(pdf_path))
texts=[]
for page in doc:
    textpage=page.get_textpage()
    texts.append(textpage.get_text_range())
    textpage.close();page.close()
doc.close()
classes=pd.read_csv(ROOT/'data/audit/semantic/final_v1_1/recognition_classes_v1.csv')
results=[]
for code in classes.canonical_sign_code:
    family,num=re.fullmatch(r'([A-Z]+)\.(\d+[a-z]?)',code).groups()
    pattern=re.compile(r'(?<![A-Za-z])'+family+r'\s*\.\s*'+num+r'(?![A-Za-z0-9])')
    pages=[i+1 for i,t in enumerate(texts) if pattern.search(t)]
    results.append({'code':code,'annex_pdf_pages_1based':pages,'token_found':bool(pages),
        'interpretation':'Code-token presence only; no new visual or semantic validation of crop labels.'})
summary={'edition':'QCVN 41:2024/BGTVT','annex_sha256':actual,'pinned_input_match':True,
    'source_url':'https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf',
    'circular_url':'https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt.pdf',
    'codes_found':sum(r['token_found'] for r in results),'released_codes':len(results),'codes':results}
out=ROOT/'output/pdf'
(out/'report2_qcvn_code_check.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(ROOT/'tmp/pdfs/report2_review/qcvn2024.txt').write_text('\f'.join(texts),encoding='utf-8')
print(json.dumps(summary,indent=2))
