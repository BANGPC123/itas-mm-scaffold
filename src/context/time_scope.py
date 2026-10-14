"""Evaluate reviewed time conditions at an explicitly supplied observation time.

Relative video timestamps and server wall clocks never establish local dates.
QCVN 41:2024 Appendix B.31 distinguishes P.131b (odd) / P.131c (even).
"""
from datetime import datetime, timedelta, timezone
import re

VIETNAM = timezone(timedelta(hours=7))


def minute(value, *, end=False):
    if end and value == '24:00':
        return 1440
    if not isinstance(value,str) or not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d',value):
        raise ValueError('Time must be HH:MM; only an end time may be 24:00')
    h,m = map(int,value.split(':'))
    return 60*h+m


def validate_windows(windows):
    if not windows or len(windows)>8:
        raise ValueError('A reviewed time panel needs 1..8 time windows')
    for start,end in windows:
        if minute(start) == minute(end,end=True):
            raise ValueError('Ambiguous zero/full-day window; use 00:00 to 24:00')


def time_scope(code, conditions, observed_at):
    """True/False/None: AND across panels; OR within each panel's windows."""
    if code not in ('P.131b','P.131c') and not conditions:
        return True
    if observed_at is None or observed_at.tzinfo is None or observed_at.utcoffset() is None:
        return None
    local = observed_at.astimezone(VIETNAM)
    if code == 'P.131b' and local.day % 2 != 1:
        return False
    if code == 'P.131c' and local.day % 2 != 0:
        return False
    now = local.hour*60+local.minute+local.second/60
    for windows in conditions:
        try:
            validate_windows(windows)
        except (ValueError,TypeError):
            return None
        matched = False
        for start,end in windows:
            a,b = minute(start),minute(end,end=True)
            matched |= a<=now<b if a<b else now>=a or now<b
        if not matched:
            return False
    return True
