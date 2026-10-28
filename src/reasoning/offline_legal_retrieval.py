"""Offline grounded legal retrieval for the desktop capstone DEMO.

Extracts only exact excerpts from SHA256-tracked official PDFs or from the
existing project reference index. Does not generate amounts/penalties, assume
GPS proves a sign legally applies, or claim consolidated legal-current status.
"""
from __future__ import annotations
from pathlib import Path
import json,re,unicodedata,math
from collections import Counter
from src.utils.types import ReasoningResult

ROOT=Path(__file__).resolve().parents[2]
CORPUS=ROOT/'data/processed/legal_reference/official_corpus_20261011'
CODE_INDEX=ROOT/'data/processed/legal_reference/index.json'

def _legacy_appendix_item(text,code):
    """Find the QCVN appendix item that defines a sign code, e.g. 'B.30 Biển số P.130'.

    Returns (letter, number, is_figure_caption) or None. A suffixed code such as
    P.123b also matches an item written 'Biển số P.123 (a,b)' or an item that
    names it right after its sibling ('B.23 Biển số P.123a ... và Biển số P.123b').
    """
    base,suffix=(code[:-1],code[-1]) if code[-1].isalpha() else (code,'')
    head=re.compile(r'(Hình\s+)?\b([A-H])\.(\d{1,3})\s*(?:[-–]\s*)?Biển\s+(?:số\s*)?'
                    +re.escape(base)+r'(?![0-9])')
    found=[]
    for m in head.finditer(text):
        tail=text[m.end():m.end()+250]
        if suffix:
            paren=re.match(r'\s*\(([a-z,\s]+)\)',tail)
            named=(re.match(re.escape(suffix)+r'(?![a-z0-9])',tail) or
                   (paren and suffix in re.findall(r'[a-z]',paren.group(1))) or
                   re.search(r'[Bb]iển\s+số\s*'+re.escape(code)+r'(?![a-z0-9])',tail))
            if not named:
                continue
        elif re.match(r'[a-z0-9]',tail):
            continue
        found.append((m.group(2),int(m.group(3)),bool(m.group(1))))
    found.sort(key=lambda item:item[2])
    return found[0] if found else None


def _extended_appendix_item(text,code):
    """Resolve suffixed appendix headings and codes listed as siblings."""
    base,suffix=(code[:-1],code[-1]) if code[-1].isalpha() else (code,'')
    head=re.compile(r'(Hình\s+)?\b([A-H])\.(\d{1,3})([a-z]?)\s*(?:[-\u2013]\s*)?Biển\s+(?:số\s*)?'
                    +re.escape(base)+r'(?![0-9])')
    found=[]
    for m in head.finditer(text):
        tail=text[m.end():m.end()+250]
        next_item=re.search(r'\b[A-H]\.\d{1,3}[a-z]?\s+(?:Biển|Hình)',tail)
        if next_item:tail=tail[:next_item.start()]
        if suffix:
            paren=re.match(r'\s*\(([a-z,\s]+)\)',tail)
            named=(re.match(re.escape(suffix)+r'(?![a-z0-9])',tail) or
                   (paren and suffix in re.findall(r'[a-z]',paren.group(1))) or
                   re.search(r'(?<![A-Za-z0-9.])'+re.escape(code)+r'(?![a-z0-9])',tail))
            if not named:continue
        elif re.match(r'[a-z0-9]',tail):
            continue
        number=int(m.group(3)) if not m.group(4) else m.group(3)+m.group(4)
        found.append((m.group(2),number,bool(m.group(1))))
    found.sort(key=lambda item:item[2])
    return found[0] if found else None


def appendix_item(text,code):
    """Prefer existing evidenced locators, then try extended heading syntax."""
    return _legacy_appendix_item(text,code) or _extended_appendix_item(text,code)


def appendix_item_rank(text,code):
    item=_legacy_appendix_item(text,code)
    if item is not None:return 1 if item[2] else 0
    return 2 if _extended_appendix_item(text,code) else 3

def with_appendix_locator(hit,code):
    """Replace a heading inherited from the last article by the appendix item.

    The structured corpus assigns the final article heading (QCVN 41:2024
    'Điều 84') to every appendix page that follows it; the excerpt itself
    carries the real appendix item label.
    """
    item=appendix_item(hit.get('excerpt',''),code)
    if item is None:
        return {**hit,'locator_source':'corpus_heading'}
    letter,number,_=item
    return {**hit,'corpus_locator':hit.get('locator'),
            'locator':f'Phụ lục {letter}, mục {letter}.{number} (biển số {code})',
            'locator_source':'appendix_item_in_excerpt'}

def normalize(s):
    s=unicodedata.normalize('NFKD',str(s)).casefold()
    s=''.join(c for c in s if not unicodedata.combining(c))
    return re.findall(r'[a-z0-9]+',s)

class OfflineLegalRetriever:
    def __init__(self,corpus_dir=None,ref_path=None):
        folder=Path(corpus_dir) if corpus_dir else CORPUS
        ref=Path(ref_path) if ref_path else CODE_INDEX
        self.manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8')) if (folder/'manifest.json').is_file() else {}
        self.chunks=[json.loads(s) for s in (folder/'chunks.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()] if (folder/'chunks.jsonl').is_file() else []
        raw=json.loads(ref.read_text(encoding='utf-8')) if ref.is_file() else {}
        self.sign_refs=raw.get('entries',{})
        self.ref_verified=bool(raw.get('legal_currency_validated'))
        self.index=[Counter(normalize(r['text'])) for r in self.chunks]
        n=max(1,len(self.index))
        freq=Counter()
        for row in self.index:freq.update(row.keys())
        self.weights={k:math.log((n+1)/(1+v))+1 for k,v in freq.items()}
        self.structured_index=None
        structured=ROOT/'data/regulations/corpus.json'
        # Custom PDF fixtures stay isolated from the project corpus.
        if corpus_dir is None and structured.is_file():
            from src.reasoning.lexical_retrieval import LexicalLegalIndex
            self.structured_index=LexicalLegalIndex(structured)

    def search_grounded(self,question,top_k=3):
        """Prefer source/locator-preserving Task 6 retrieval; PDF fallback."""
        if self.structured_index is not None:
            hits=self.structured_index.search(question,top_k)
            if hits:return hits
        return self.search(question,top_k)

    def search(self,question,top_k=3,*,document=None):
        if not isinstance(question,str) or not question.strip() or not 1<=top_k<=8:
            raise ValueError('Invalid query/top_k')
        query=Counter(normalize(question))
        scored=[]
        for i,row in enumerate(self.chunks):
            if document and row['document']!=document:continue
            counts=self.index[i]
            score=sum(self.weights.get(k,0)*min(3,counts.get(k,0))*v for k,v in query.items())
            if score<=0:continue
            # Exact code text receives a boost over unrelated law passages.
            if re.search(r'\b[A-Z]{1,3}\.\d{3}[a-z]?\b',question):
                code=re.findall(r'[A-Z]{1,3}\.\d{3}[a-z]?',question)[0]
                if code.casefold() in row['text'].casefold():score+=18
            scored.append((score,i))
        scored.sort(reverse=True)
        results=[]
        for _,i in scored[:top_k]:
            d=self.chunks[i]
            results.append({'document':d['document'],'code':question.strip()[:80],
                'page':d['page'],'excerpt':d['text'][:720],
                'edition':d['document'],'source_url':d['source_url'],
                'source_file':d['file'],'research_only':True,'legal_currency_verified':False})
        return results

    def lookup_sign(self,code):
        if code in self.sign_refs:
            data=dict(self.sign_refs[code])
            data['research_only']=True
            data['legal_currency_verified']=self.ref_verified
            return data
        if self.structured_index is not None:
            base=code[:-1] if code[-1].isalpha() else code
            hits={}
            for query in dict.fromkeys((code,base)):
                for h in self.structured_index.search(query,20,document='qcvn-41-2024-bgtvt',unique_locators=False):
                    hits.setdefault(h['chunk_id'],h)
            hits=[h for h in hits.values() if base.casefold() in h['excerpt'].casefold()]
            ranked=sorted(hits,key=lambda h:appendix_item_rank(h['excerpt'],code))
            if ranked:
                return with_appendix_locator({**ranked[0],'code':code},code)
        hits=self.search(code,1,document='QCVN 41:2024/BGTVT')
        return hits[0] if hits and code.casefold() in hits[0]['excerpt'].casefold() else None

    def for_signs(self,signs,zone_label):
        # Signs can be visually detected but not legally applicable. Only
        # exact source snippets and uncertainty notice are returned.
        codes=list(dict.fromkeys(s.label for s in signs if s.label!='unknown' and s.panel_role!='supplementary'))
        refs=[p for code in codes if (p:=self.lookup_sign(code)) is not None]
        notes=[r['excerpt'][:600] for r in refs]
        if codes:
            summary=('Nhận dạng tham khảo: '+', '.join(codes)+'. '
                'Đã tìm '+str(len(refs))+' mục trong tài liệu lưu trữ. '
                'Cần xác minh biển phụ, làn áp dụng và hiệu lực văn bản trước khi kết luận quy định hoặc mức phạt.')
        else:
            summary='Chưa có mã biển rõ để tra cứu; có thể tìm thủ công trong kho văn bản bên dưới.'
        return ReasoningResult(summary,notes,zone_label,references=refs)
