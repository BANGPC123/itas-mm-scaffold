"""Verified sign-based Vietnamese urban-zone transitions (QCVN 41:2024, D.17/D.18).

Independent of mocked GPS coordinates. R.420 starts and R.421 ends the zone.
UNVERIFIED AI predictions NEVER change the legally operative status.
This tracker NEVER infers any numeric speed limit, TTS advice or road hierarchy.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

@dataclass
class ZoneSignState:
    status: str = "unknown"  # populated | outside | unknown
    last_timestamp: float | None = None
    last_valid_code: str | None = None
    stream_id: str | None = None
    source: str = "no_verified_road_sign_evidence"
    def reset(self,stream_id: str | None = None):
        self.status="unknown"
        self.last_timestamp=None
        self.last_valid_code=None
        self.stream_id=stream_id
        self.source="no_verified_road_sign_evidence"
    def update(self, signs: Iterable, *, stream_id: str, timestamp: float) -> str:
        if not isinstance(stream_id,str) or not stream_id:
            raise ValueError('Stream identification is mandatory')
        if not isinstance(timestamp,(int,float)) or not (0<=timestamp<float('inf')):
            raise ValueError('Monotonic video timestamp required')
        if self.stream_id!=stream_id:
            self.reset(stream_id)
        if self.last_timestamp is not None and timestamp<self.last_timestamp:
            # Seek/replay invalidates previous sign jurisdiction.
            self.reset(stream_id)
        self.last_timestamp=float(timestamp)
        valid=[]
        for s in signs:
            if (s.label not in ('R.420','R.421') or
                not (s.context_verified and s.label_verified) or
                s.applicability!='applies' or
                s.panel_role!='primary' or
                s.is_compound or
                s.placement not in ('roadside','overhead') or
                s.lane_scope not in ('all','ego')):
                continue
            valid.append(s.label)
        if len(set(valid))>1:
            # Both opposing transitions in a single frame are ambiguous.
            self.status='unknown'
            self.source='conflicting_verified_zone_signs'
        elif valid:
            self.last_valid_code=valid[-1]
            self.status='populated' if valid[-1]=='R.420' else 'outside'
            self.source='verified_QCVN41_sign_transition'
        return self.status
