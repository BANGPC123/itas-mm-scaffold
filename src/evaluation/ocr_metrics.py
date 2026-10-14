"""Exact-text OCR evaluation on independently verified supplementary panels.

No inferred legal meaning; accents, words and numbers all matter.
"""
from collections import Counter
import math,unicodedata
def normalize_plate_text(value):
    if not isinstance(value,str):raise TypeError('OCR text must be string')
    s=unicodedata.normalize('NFKC',value).casefold()
    for old,new in {'—':'-','–':'-','−':'-','－':'-','：':':','；':';'}.items():
        s=s.replace(old,new)
    return ''.join(s.split())

def choose_consensus(view_text_scores,threshold=.85,min_agreement=2):
    """Views may be correlated. Agreement is only a heuristic, not a guarantee."""
    counter=Counter()
    for rec in view_text_scores:
        scores=rec.get('scores',[])
        texts=rec.get('text',[])
        if not texts or len(scores)!=len(texts):continue
        if any(not isinstance(p,(int,float)) or not math.isfinite(p) or p<threshold or p>1 for p in scores):
            continue
        text=' '.join(str(v) for v in texts)
        norm=normalize_plate_text(text)
        if norm:counter[norm]+=1
    if not counter:return None
    label,agree=counter.most_common(1)[0]
    return label if agree>=min_agreement else None

def score_verified_text(records):
    """records contain {'human_verified':true,'true_text','predicted_text'}.

    Missing OCR output counts as a miss for end-to-end exact accuracy.
    """
    total=accepted=correct=0
    for rec in records:
        if rec.get('human_verified') is not True:
            raise ValueError('Cannot score unverified supplementary transcription')
        gt=normalize_plate_text(rec['true_text'])
        if not gt:raise ValueError('Empty ground-truth text')
        total+=1
        predicted=rec.get('predicted_text')
        if predicted is not None:
            accepted+=1
            correct+=int(normalize_plate_text(predicted)==gt)
    return {'verified_panels':total,'accepted_panels':accepted,
            'coverage':accepted/total if total else None,
            'exact_accuracy_on_all':correct/total if total else None,
            'exact_accuracy_on_accepted':correct/accepted if accepted else None,
            'note':'Research OCR exact-text only; not legal interpretation or driving validation.'}
