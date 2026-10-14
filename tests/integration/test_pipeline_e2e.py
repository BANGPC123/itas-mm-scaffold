from unittest.mock import Mock
import numpy as np
import pytest
from src.pipeline.orchestrator import Pipeline
from src.context.sign_applicability import SignApplicability
from src.context.zone_classifier import ZoneClassifier
from src.utils.config import load_config
from src.utils.types import LaneInfo,SignDetection

@pytest.fixture
def pipeline():
    p=Pipeline.__new__(Pipeline)
    p.sign_detector=Mock();p.sign_detector.detect.return_value=[]
    p.sign_recognizer=Mock();p.sign_recognizer.recognize.side_effect=lambda image,signs:signs
    p.lane_detector=Mock();p.lane_detector.detect.return_value=LaneInfo([],backend='ufld_v2')
    p.zone_classifier=ZoneClassifier(load_config('context'))
    p.applicability=SignApplicability();p.current_vehicle='car'
    p.tts_engine=Mock();p.rag_chain=None
    return p

def test_no_signs_do_not_trigger_legal_generation_or_speech(pipeline):
    result=pipeline.run(np.zeros((100,100,3),np.uint8),10.7769,106.7009)
    assert result.reasoning.retrieved_chunks==[]
    assert result.audio_path is None
    pipeline.tts_engine.synthesize.assert_not_called()
    assert result.annotated_image.startswith('data:image/jpeg;base64,')

def test_tts_failure_preserves_visual_result(pipeline):
    pipeline.sign_detector.detect.return_value=[SignDetection('P.130',.9,(10,10,50,50), panel_role='primary', label_verified=True, context_verified=True, placement='roadside', lane_scope='all')]
    pipeline.tts_engine.synthesize.side_effect=RuntimeError('offline')
    result=pipeline.run(np.zeros((100,100,3),np.uint8),10.7769,106.7009)
    assert result.audio_path is None and result.annotated_image
    assert any('trực tuyến' in w for w in result.warnings)
    assert result.perception.signs[0].applicability=='applies'

def test_ignored_sign_is_not_spoken(pipeline):
    pipeline.sign_detector.detect.return_value=[SignDetection('P.130',.9,(10,10,50,50),allowed_vehicles=['motorcycle'],context_verified=True,label_verified=True)]
    result=pipeline.run(np.zeros((100,100,3),np.uint8),10.7769,106.7009)
    assert result.perception.signs[0].applicability=='ignored'
    pipeline.tts_engine.synthesize.assert_not_called()

def test_empty_input_is_rejected(pipeline):
    with pytest.raises(ValueError):pipeline.run(np.zeros((0,0,3),np.uint8),10,106)


def test_road_roi_excludes_dashboard_before_models_and_keeps_coordinates(pipeline):
    pipeline.sign_detector.detect.return_value = [SignDetection('P.130', .9, (10, 10, 30, 40))]
    result = pipeline.run(np.zeros((100,100,3),np.uint8),10,106,False,road_bottom=.65,annotate=False)
    assert pipeline.sign_detector.detect.call_args.args[0].shape == (65,100,3)
    assert pipeline.lane_detector.detect.call_args.args[0].shape == (100,100,3)
    assert result.perception.signs[0].bbox_xyxy == (10,10,30,40)
    assert result.annotated_image is None


@pytest.mark.parametrize('bottom', [0, -.1, 1.1, float('nan')])
def test_invalid_road_roi_is_rejected(pipeline, bottom):
    with pytest.raises(ValueError):
        pipeline.run(np.zeros((100,100,3),np.uint8),10,106,False,road_bottom=bottom)



def test_background_rejected_advertisement_never_reaches_context_or_speech(pipeline):
    fake = SignDetection('P.127', .99, (10, 10, 50, 50))
    fake.recognition_source = 'background_rejected'
    pipeline.sign_detector.detect.return_value=[fake]
    result=pipeline.run(np.zeros((100,100,3),np.uint8),10.7769,106.7009,True)
    assert result.perception.signs==[]
    assert result.reasoning.speech_text==''
    pipeline.tts_engine.synthesize.assert_not_called()



def test_camera_mask_excludes_detection_before_recognition(pipeline):
    pipeline.sign_detector.detect.return_value=[
        SignDetection('P.127',.99,(20,20,40,40))]
    result=pipeline.run(np.zeros((100,100,3),np.uint8),10,106,False,
                        road_bottom=.65,exclude_rects=[[.1,.1,.5,.5]],annotate=False)
    assert result.perception.signs==[]
    assert result.reasoning.speech_text==''
    assert pipeline.sign_recognizer.recognize.call_args.args[1]==[]
    assert pipeline.lane_detector.detect.call_args.args[0].shape==(100,100,3)
