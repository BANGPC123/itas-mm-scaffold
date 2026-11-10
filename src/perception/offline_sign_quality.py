"""Offline, high-resolution rescue of traffic signs for stored-video processing.

Three native detection scales (960, 640, 1280) are used only when user
explicitly requests slower video-quality mode. This never infers legal scope.
All decisions are measurable on frozen VAL before use in the main pipeline.
"""
from __future__ import annotations

from copy import deepcopy
import math

from src.perception.sign_multiscale import recognize_views, crop_side
from src.perception.temporal_signs import overlap


def safe_refinement(primary, high_resolution, *, detector_confidence=.70,
                    recognition_confidence=.93, min_side=16, max_extra=12):
    """Use a third pass to rescue unresolved/crop-small panels, not override
    confidently classified signs. Deduplicate by position and keep provenance.
    """
    merged=deepcopy(primary)
    newly=0
    for candidate in sorted(high_resolution,key=lambda s:s.confidence,reverse=True):
        if candidate.label=='unknown' or candidate.recognition_source in (
            'background_rejected','outside_supported_reference','symbol_shape_unsupported'):
            continue
        if (candidate.confidence<detector_confidence or
            (candidate.recognition_confidence or 0)<recognition_confidence or
            crop_side(candidate.bbox_xyxy, (100000,100000))<min_side):
            continue
        matched=sorted((overlap(candidate.bbox_xyxy,s.bbox_xyxy),i)
                        for i,s in enumerate(merged))
        if matched and matched[-1][0]>=.30:
            previous=merged[matched[-1][1]]
            # Rescue is allowed only from unknown and only for a very
            # convincing, geometrically overlapping high-resolution crop.
            if (previous.label=='unknown' and matched[-1][0]>=.55 and
                candidate.recognition_confidence>=.96):
                upgraded=deepcopy(candidate)
                upgraded.recognition_source='offline_highres_rescue'
                merged[matched[-1][1]]=upgraded
            continue
        if newly>=max_extra:
            continue
        added=deepcopy(candidate)
        added.recognition_source='offline_highres_new'
        merged.append(added)
        newly+=1
    return merged


def detect_offline_high_accuracy(detector,recognizer,image, *,
                                 detector_confidence=.70,
                                 recognition_confidence=.93,
                                 min_short_side=12):
    original=detector.image_size
    try:
        detector.image_size=960
        a=detector.detect(image)
        detector.image_size=640
        b=detector.detect(image)
        main=recognize_views(recognizer,image,a,b,min_short_side=min_short_side)
        detector.image_size=1280
        hi=detector.detect(image)
        classified=recognizer.recognize(image,hi,min_short_side=min_short_side)
        return safe_refinement(main,classified,
            detector_confidence=detector_confidence,
            recognition_confidence=recognition_confidence,
            min_side=max(16,min_short_side))
    finally:
        detector.image_size=original
