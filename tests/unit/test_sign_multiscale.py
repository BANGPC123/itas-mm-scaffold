from copy import deepcopy
import numpy as np
from src.perception.sign_multiscale import recognize_views
from src.perception.sign_detector import SignDetector
from src.utils.types import SignDetection


class Recognized:
    def recognize(self,image,signs,**kwargs):return deepcopy(signs)


def sign(label='P.130',confidence=.98,source='single_frame',box=(10,10,40,40),detection=.8):
    return SignDetection(label,detection,box,recognition_confidence=confidence,recognition_source=source)


def test_secondary_scale_recovers_uncertain_crop_but_preserves_clear_label():
    image=np.zeros((100,100,3),np.uint8);r=Recognized()
    base=sign('unknown',.5,'low_confidence');other=sign()
    result=recognize_views(r,image,[base],[other])[0]
    assert result.label=='P.130' and result.recognition_source=='multiscale_rescue'
    assert base.label=='unknown'
    assert recognize_views(r,image,[sign('P.131a')],[other])[0].label=='P.131a'


def test_reference_symbol_and_background_contradictions_cannot_be_rescued():
    for source in ['outside_supported_reference','symbol_shape_unsupported','background_rejected']:
        result=recognize_views(Recognized(),np.zeros((100,100,3),np.uint8),[sign('unknown',.99,source)],[sign()])
        assert result[0].label=='unknown' and len(result)==1


def test_tiny_crop_requires_matching_high_confidence_code_in_both_views():
    image=np.zeros((100,100,3),np.uint8);box=(10,10,23,23)
    assert recognize_views(Recognized(),image,[sign(box=box)],[sign(box=box)],min_short_side=12)[0].label=='P.130'
    for other in [[],[sign('P.131a',box=box)],[sign(confidence=.94,box=box)]]:
        assert recognize_views(Recognized(),image,[sign(box=box)],other,min_short_side=12)[0].label=='unknown'


def test_one_crop_cannot_rescue_two_ambiguous_overlapping_primary_boxes():
    image=np.zeros((100,100,3),np.uint8)
    result=recognize_views(Recognized(),image,[sign('unknown',.5,'low_confidence'),sign('unknown',.5,'low_confidence')],[sign()])
    assert all(s.label=='unknown' for s in result)


def test_secondary_only_sign_needs_strong_localization_and_code_and_no_duplicate():
    image=np.zeros((100,100,3),np.uint8)
    assert len(recognize_views(Recognized(),image,[],[sign()]))==1
    assert recognize_views(Recognized(),image,[],[sign(detection=.6)])==[]
    assert len(recognize_views(Recognized(),image,[sign()],[sign()]))==1


def test_detector_restores_resolution_after_secondary_failure():
    detector=SignDetector.__new__(SignDetector);detector.image_size=960;detector.secondary_image_size=640
    def detect(image):
        if detector.image_size==640:raise RuntimeError('secondary failed')
        return []
    detector.detect=detect
    import pytest
    with pytest.raises(RuntimeError):detector.detect_views(np.zeros((20,20,3),np.uint8))
    assert detector.image_size==960
