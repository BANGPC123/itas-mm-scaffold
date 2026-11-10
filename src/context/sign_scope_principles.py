"""Explicit, conservative vehicle scope for validated Vietnamese road signs.

QCVN 41:2024/BGTVT, Appendix B.30/B.31 (P.130/P.131) and F.5 (S.505a).
No supplementary S.505a: primary sign's own vehicle class remains.
Verified S.505a: intersection with indicated classes, NOT unrestricted.
Unrecognized/unreadable S.505a: scope unknown, never a legal instruction.

This is the supported capstone subset, not an exhaustive regulatory engine.
"""
from __future__ import annotations

MOTOR_VEHICLES=frozenset({'car','truck','bus','van','motorbike','motorcycle','moped','electric_motorbike'})
NONMOTOR=frozenset({'bicycle','pedestrian','pedal_cycle','horse_cart'})
MOTOR_PARKING_CODES=frozenset({'P.130','P.131a','P.131b','P.131c'})
AUTOMOBILE_CLASSES=frozenset({'car','truck','bus','van'})

ALIASES={
 'ô tô':'automobile','oto':'automobile','xe ô tô':'automobile','automobile':'automobile',
 'ô tô con':'car','oto con':'car','car':'car','xe hơi':'car',
 'xe máy':'motorcycle','motorbike':'motorcycle','motorcycle':'motorcycle',
 'xe tải':'truck','truck':'truck',
 'xe buýt':'bus','bus':'bus',
 'xe đạp':'bicycle','bicycle':'bicycle',
 'xe khách':'bus','van':'van',
 'pedestrian':'pedestrian','người đi bộ':'pedestrian',
}

def normalize_vehicle(value: str) -> str:
    text=value.strip().casefold()
    return ALIASES.get(text,text)

def expand_vehicle_labels(labels):
    """Expand generic automobile plates without treating them as passenger cars only."""
    if not labels:
        return None
    allowed=set()
    for label in labels:
        value=normalize_vehicle(label)
        if value=='automobile':
            allowed.update(AUTOMOBILE_CLASSES)
        elif value in MOTOR_VEHICLES or value in NONMOTOR:
            allowed.add(value)
        else:
            return None
    return allowed or None

def base_vehicle_scope(primary_code: str, vehicle: str) -> str:
    """'included'|'excluded'|'unknown' (not a complete sign applicability)."""
    v=normalize_vehicle(vehicle)
    if primary_code in MOTOR_PARKING_CODES:
        if v in MOTOR_VEHICLES or v=='automobile':return 'included'
        if v in NONMOTOR:return 'excluded'
        return 'unknown'
    return 'not_restricted_by_implemented_rules'

def supplementary_vehicle_scope(allowed: list[str], vehicle: str) -> bool | None:
    """None: plate condition wasn't decoded; caller must abstain."""
    vals=expand_vehicle_labels(allowed)
    v=normalize_vehicle(vehicle)
    if vals is None or v not in MOTOR_VEHICLES|NONMOTOR:
        return None
    return v in vals
