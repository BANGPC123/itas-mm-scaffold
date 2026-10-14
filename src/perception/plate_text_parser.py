"""Parse potential time windows in supplementary road-sign OCR.

This is syntactic parsing only, NOT a legal time restriction or TTS source.
All candidates MUST be verified against the actual road sign.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

# A colon is required so year/phone/building numbers are never parsed as hours.
_INTERVAL = re.compile(
    r"(?<![0-9])(?P<h1>[01]?\d|2[0-3]):(?P<m1>[0-5]\d)\s*[-–—~]\s*"
    r"(?P<h2>[01]?\d|2[0-3]):(?P<m2>[0-5]\d)(?![0-9])"
)

@dataclass(frozen=True)
class TimeTextCandidate:
    raw_text: str
    windows: tuple[str, ...]
    complete: bool
    verified: bool = False
    source: str = 'ocr_unreviewed'

def parse_time_windows(text: str) -> TimeTextCandidate:
    if not isinstance(text,str):
        raise TypeError("OCR text must be a string")
    normalized=text.translate(str.maketrans({'：':':','－':'-','—':'-','–':'-','〜':'~','～':'~'}))
    matches=list(_INTERVAL.finditer(normalized))
    windows=[]
    for m in matches:
        first=f"{int(m['h1']):02d}:{m['m1']}"
        second=f"{int(m['h2']):02d}:{m['m2']}"
        if first==second:
            continue  # 00:00-00:00 could ambiguously mean 24h or empty
        val=f"{first}-{second}"
        if val not in windows:windows.append(val)
    # The parser reports complete only if ALL visible time-form strings form
    # full accepted ranges. Loose isolated times or incorrect punctuation
    # remain unverified even if another valid range is found.
    isolated=re.findall(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?!\d)",normalized)
    complete=bool(windows) and len(isolated)==2*len(matches) and len(windows)==len(matches)
    return TimeTextCandidate(text,tuple(windows),complete)

def stable_ocr_candidate(texts: list[str], confidence_scores: list[float], *, minimum_confidence=.85):
    if len(texts)!=len(confidence_scores):
        raise ValueError('Text / confidence length mismatch')
    if not texts:return TimeTextCandidate('',(),False)
    result=parse_time_windows('; '.join(texts))
    conf=all(isinstance(x,(int,float)) and minimum_confidence<=x<=1 for x in confidence_scores)
    return TimeTextCandidate(result.raw_text, result.windows, result.complete and conf)
