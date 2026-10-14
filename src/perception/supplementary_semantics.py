"""Syntactic vehicle/time OCR candidates; never establish legal scope."""
import re,unicodedata
from src.perception.plate_text_parser import stable_ocr_candidate

def parse_supplementary(texts,scores):
 if len(texts)!=len(scores):raise ValueError('OCR lengths differ')
 raw='; '.join(texts)
 # Expand explicit printed h notation; do not guess digit substitutions.
 canonical=re.sub(r'(?<!\d)([01]?\d|2[0-3])\s*[hH]\s*([0-5]\d)(?!\d)',r'\1:\2',raw)
 t=stable_ocr_candidate([canonical],[min(scores)] if scores else []) if scores else stable_ocr_candidate([],[])
 plain=unicodedata.normalize('NFKD',raw.lower().replace('đ','d'))
 plain=''.join(x for x in plain if not unicodedata.combining(x))
 names={'car':r'\b(?:o\s+to\s+con|xe\s+con|xe\s+hoi)\b',
  'truck':r'\b(?:xe\s+tai|o\s+to\s+tai)\b',
  'motorcycle':r'\b(?:xe\s+may|mo\s+to)\b',
  'bus':r'\b(?:xe\s+buyt|xe\s+khach|o\s+to\s+khach)\b','bicycle':r'\bxe\s+dap\b'}
 confident=bool(scores) and all(.85<=s<=1 for s in scores)
 vehicles=[v for v,pattern in names.items() if confident and re.search(pattern,plain)]
 if confident and re.search(r'\bo\s+to\b(?!\s+(?:tai|khach|con)\b)',plain):vehicles.append('automobile')
 exclusion=bool(re.search(r'\b(?:tru|ngoai tru|khong ap dung)\b',plain))
 return {'raw_text':raw,'time_windows':[x.split('-') for x in t.windows] if t.complete else [],
  'vehicle_candidates':vehicles,'condition_polarity':'exclusion' if exclusion else 'unspecified',
  'syntax_complete':t.complete,'ocr_confident':confident,'machine_read':True,'human_verified':False,
  'legal_effect_established':False}
