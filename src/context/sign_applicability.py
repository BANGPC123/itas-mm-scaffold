"""Traffic sign scope reasoning. Fail closed when visual/legal evidence is incomplete.

Image projection is only a geometric consistency check: a sign's legal scope
cannot be established from its x coordinate alone (perspective, arrows, gantry
layout, supplementary plates and road direction all matter).
"""
from dataclasses import replace

import numpy as np

from src.perception.sign_catalog import NAMES_VI
from src.perception.lane_consistency import credible_ego
from src.context.sign_scope_principles import base_vehicle_scope, supplementary_vehicle_scope, normalize_vehicle, expand_vehicle_labels
from src.context.time_scope import time_scope, validate_windows


class SignApplicability:
    def __init__(self, config=None):
        self.cfg = config or {}

    def _propose_panel_roles(self, signs):
        """Propose a secondary plate only when its spatial parent is unique.

        No OCR/semantic claim is made: unknown text stays unknown and its
        parent becomes uncertain. Explicit reviewer roles take precedence.
        """
        items = [replace(s, panel_labels=list(s.panel_labels),
                         allowed_vehicles=list(s.allowed_vehicles)) for s in signs]
        if not self.cfg.get('infer_supplementary_candidates', True):
            return items
        for sub in items:
            if sub.panel_role != 'unknown' or sub.label != 'unknown':
                continue
            sx1,sy1,sx2,sy2=sub.bbox_xyxy
            sw,sh=sx2-sx1,sy2-sy1
            if sw<=0 or sh<=0:
                continue
            candidates=[]
            for parent in items:
                if parent is sub or parent.panel_role=='supplementary' or parent.label not in NAMES_VI and parent.label!='unknown':
                    continue
                px1,py1,px2,py2=parent.bbox_xyxy
                pw,ph=px2-px1,py2-py1
                if pw<=0 or ph<=0 or sh>.85*ph or sw>1.35*pw:
                    continue
                overlap=max(0,min(px2,sx2)-max(px1,sx1))/min(pw,sw)
                gap=sy1-py2
                if parent.label=='unknown':
                    # Stricter geometry for unknown main+unknown plate:
                    # a plausible compound candidate, NOT a decoded sign.
                    plausible=(.67*pw<=sw<=1.3*pw and .18*ph<=sh<=.65*ph
                               and overlap>=.8 and -.08*ph<=gap<=.35*ph)
                else:
                    plausible=(overlap>=self.cfg.get('panel_overlap',.5)
                               and -.06*ph<=gap<=self.cfg.get('panel_gap_ratio',.5)*ph)
                if plausible:
                    score=gap/ph+abs((px1+px2)-(sx1+sx2))/(pw+sw)
                    candidates.append((score,parent))
            candidates.sort(key=lambda p:p[0])
            if len(candidates)==1 or (len(candidates)>1 and candidates[1][0]-candidates[0][0]>=.15):
                sub.panel_role='supplementary'
                sub.panel_role_source='geometric_candidate'
                if candidates[0][1].panel_role=='unknown':
                    candidates[0][1].panel_role='primary'
                    candidates[0][1].panel_role_source='geometric_candidate'
        return items

    def group(self, signs):
        # Keep caller's predictions immutable and avoid ambiguous parents.
        signs=self._propose_panel_roles(signs)
        primaries = [
            replace(s, panel_labels=list(s.panel_labels), allowed_vehicles=list(s.allowed_vehicles))
            for s in signs if s.panel_role != 'supplementary'
        ]
        unmatched = []
        for sub in (s for s in signs if s.panel_role == 'supplementary'):
            sx1, sy1, sx2, sy2 = sub.bbox_xyxy
            candidates = []
            for i, main in enumerate(primaries):
                if main.panel_role != 'primary':
                    continue
                x1, y1, x2, y2 = main.bbox_xyxy
                w, h = x2-x1, y2-y1
                overlap = max(0, min(x2, sx2)-max(x1, sx1))/max(1, min(w, sx2-sx1))
                gap = sy1-y2
                if overlap >= self.cfg.get('panel_overlap', .5) and 0 <= gap <= h*self.cfg.get('panel_gap_ratio', .5):
                    alignment = abs((x1+x2)-(sx1+sx2))/max(1, w+(sx2-sx1))
                    candidates.append((gap/max(1, h) + alignment, i))
            candidates.sort()
            if not candidates or (len(candidates) > 1 and candidates[1][0]-candidates[0][0] < .15):
                # An ambiguous plate must also veto audible/visual "applies"
                # decisions from EACH plausible parent, not only itself.
                # Otherwise the main sign could be spoken without its terms.
                if candidates:
                    for _, index in candidates:
                        primaries[index].context_verified = False
                        primaries[index].applicability_reason = (
                            'Unresolved supplementary plate could modify this main sign')
                unmatched.append(replace(sub, applicability='uncertain',
                                         applicability_reason='Supplementary panel has no unique parent'))
                continue
            main = primaries[candidates[0][1]]
            main.is_compound = True
            main.panel_labels = main.panel_labels or [main.label]
            main.panel_labels.append(sub.label)
            # Supplement meaning must be verified, not inferred from an OCR guess.
            if not (sub.context_verified and sub.label_verified):
                main.context_verified = False
            if sub.label in ('S.508a','S.508b'):
                try:
                    validate_windows(sub.time_windows)
                    verified=sub.context_verified and sub.label_verified
                except (ValueError,TypeError):
                    verified=False
                if verified:
                    main.time_conditions=[*main.time_conditions,list(sub.time_windows)]
                else:
                    main.context_verified=False
                    main.applicability_reason='Time panel is unreadable or unverified'
                continue
            # S.505a specifies the vehicle TYPES subjected to the parent.
            # The plate code/symbol must be independently reviewed: geometry
            # and OCR alone cannot authorize a vehicle restriction.
            if sub.label == 'S.505a':
                if sub.context_verified and sub.label_verified and sub.allowed_vehicles:
                    normalized=expand_vehicle_labels(sub.allowed_vehicles)
                    if normalized is None:
                        main.allowed_vehicles=['unresolved_vehicle_condition']
                    elif main.allowed_vehicles:
                        existing=expand_vehicle_labels(main.allowed_vehicles)
                        narrowed=existing & normalized if existing is not None else set()
                        main.allowed_vehicles=sorted(narrowed) if narrowed else ['unresolved_vehicle_condition']
                    else:
                        main.allowed_vehicles=sorted(normalized)
                else:
                    main.allowed_vehicles=['unresolved_vehicle_condition']
                    main.context_verified=False
            elif sub.allowed_vehicles:
                # A human-reviewed custom vehicle plate may also constrain the
                # sign, but unreadable wording must never widen its scope.
                if sub.context_verified and sub.label_verified:
                    incoming=expand_vehicle_labels(sub.allowed_vehicles)
                    existing=expand_vehicle_labels(main.allowed_vehicles) if main.allowed_vehicles else None
                    if incoming is not None and existing is not None:
                        incoming &= existing
                    main.allowed_vehicles=sorted(incoming) if incoming else ['unresolved_vehicle_condition']
                else:
                    main.allowed_vehicles=['unresolved_vehicle_condition']
                    main.context_verified=False
            elif sub.label not in ('S.505a',):
                # S.508(time), S.503(direction), S.504(lane), unknown plate:
                # cannot ignore these semantic modifiers in the MVP.
                main.context_verified=False
                main.applicability_reason='Supplementary condition not interpreted'
        return primaries + unmatched

    def _projection(self, sign, lane, shape):
        if not credible_ego(lane.ego_boundaries, shape):
            return 'unavailable'
        y = shape[0] * self.cfg.get('projection_y_ratio', .8)
        xs = []
        for polyline in lane.ego_boundaries:
            ordered = sorted(polyline, key=lambda p: p[1])
            if not ordered or y < ordered[0][1] or y > ordered[-1][1]:
                return 'unavailable'
            xs.append(float(np.interp(y, [p[1] for p in ordered],
                                      [p[0] for p in ordered])))
        left, right = xs
        width = right-left
        if width < shape[1]*self.cfg.get('minimum_lane_width_ratio', .05):
            return 'unavailable'
        cx = (sign.bbox_xyxy[0]+sign.bbox_xyxy[2])/2
        margin = width * self.cfg.get('boundary_margin_ratio', .1)
        if left+margin < cx < right-margin:
            return 'ego'
        if cx < left-margin or cx > right+margin:
            return 'other'
        return 'borderline'

    def apply(self, signs, lane, image_shape, vehicle='car', *, observed_at=None):
        grouped = self.group(signs)
        for sign in grouped:
            # Invalid/unsupported classes and unresolved plates must not become
            # instructions even when a reviewer has selected a lane.
            if (sign.label not in NAMES_VI or sign.panel_role == 'supplementary'
                    or 'unknown' in sign.panel_labels or 'unresolved_vehicle_condition' in sign.allowed_vehicles):
                sign.applicability = 'uncertain'
                sign.applicability_reason = 'Sign or supplementary meaning is unresolved'
                continue
            # First apply the base sign's intrinsic vehicle scope, then any
            # reviewed supplementary S.505a class restriction.
            base_scope=base_vehicle_scope(sign.label,vehicle)
            if base_scope=='excluded':
                if sign.context_verified and sign.label_verified:
                    sign.applicability='ignored'
                    sign.applicability_reason='Primary parking prohibition applies to motor vehicles, not this class'
                else:
                    sign.applicability='uncertain'
                    sign.applicability_reason='Primary code/scope not verified'
                continue
            if base_scope=='unknown':
                sign.applicability='uncertain'
                sign.applicability_reason='Unknown vehicle category for this parking prohibition'
                continue
            vehicle_condition=supplementary_vehicle_scope(sign.allowed_vehicles,vehicle) if sign.allowed_vehicles else True
            if vehicle_condition is None:
                sign.applicability='uncertain'
                sign.applicability_reason='Unreadable vehicle scope on supplementary plate'
                continue
            if vehicle_condition is False:
                if sign.context_verified and sign.label_verified:
                    sign.applicability = 'ignored'
                    sign.applicability_reason = 'Verified supplementary plate excludes current vehicle'
                else:
                    sign.applicability = 'uncertain'
                    sign.applicability_reason = 'Vehicle restriction has not been verified'
                continue
            if not (sign.context_verified and sign.label_verified):
                sign.applicability = 'uncertain'
                sign.applicability_reason = 'Sign code, placement and lane scope not fully reviewed'
                continue
            temporal=time_scope(sign.label,sign.time_conditions,observed_at)
            if temporal is not True:
                sign.applicability='ignored' if temporal is False else 'uncertain'
                sign.applicability_reason=('Outside reviewed local date/time condition' if temporal is False
                                           else 'Explicit timezone-aware observation time required')
                continue
            if sign.lane_scope == 'other':
                sign.applicability = 'ignored'
                sign.applicability_reason = 'Verified: sign is for a different lane'
                continue
            if sign.placement == 'roadside':
                if sign.lane_scope in ('all', 'ego') and sign.panel_role == 'primary':
                    sign.applicability = 'applies'
                    sign.applicability_reason = 'Reviewed roadside sign and confirmed lane/vehicle scope'
                else:
                    sign.applicability = 'uncertain'
                    sign.applicability_reason = 'Roadside jurisdiction not established'
                continue
            if sign.placement == 'overhead' and sign.panel_role == 'primary':
                projection = self._projection(sign, lane, image_shape)
                if sign.lane_scope == 'ego' and projection == 'ego':
                    sign.applicability = 'applies'
                    sign.applicability_reason = 'Reviewed overhead sign scope corroborated by lane projection'
                elif sign.lane_scope == 'ego' and projection == 'other':
                    sign.applicability = 'uncertain'
                    sign.applicability_reason = 'Overhead sign verification conflicts with image projection'
                else:
                    sign.applicability = 'uncertain'
                    sign.applicability_reason = (
                        f'Overhead image projection suggests {projection}; verified lane scope and geometry required'
                    )
                continue
            sign.applicability = 'uncertain'
            sign.applicability_reason = 'Placement and applicability not established'
        return grouped
