import pytest
from src.perception.plate_text_parser import parse_time_windows,stable_ocr_candidate

@pytest.mark.parametrize('text,expected',[
    ('6:00-8:00',('06:00-08:00',)),
    ('06:00 - 22:00',('06:00-22:00',)),
    ('6：00－8：00; 16：00–18：00',('06:00-08:00','16:00-18:00')),
    ('Số 89/2025, ngày 30/4/2025',()),
    ('28:00 - 48:00',()),
    ('00:00-00:00',()),
    ('7:15~8:00',('07:15-08:00',)),
])
def test_strict_time_range_parser(text,expected):
    assert parse_time_windows(text).windows==expected

@pytest.mark.parametrize('text',[
    '6:00','6:00 - ?', '6:00-8:00; 20:00','6:00-6:00'])
def test_incomplete_text_never_claims_complete(text):
    assert not parse_time_windows(text).complete

def test_ocr_strict_confidence_and_no_legal_verification():
    success=stable_ocr_candidate(['6:00-8:00','16:00-18:00'],[.94,.96])
    assert success.complete and len(success.windows)==2
    assert success.verified is False
    assert success.source=='ocr_unreviewed'
    reject=stable_ocr_candidate(['6:00-8:00'],[.67])
    assert not reject.complete
    assert reject.windows==('06:00-08:00',)

def test_reject_fabricated_scores():
    with pytest.raises(ValueError):stable_ocr_candidate(['6:00-8:00'],[])
    assert not stable_ocr_candidate(['6:00-8:00'],[float('nan')]).complete
