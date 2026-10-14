"""Separate AI demo narration from human-verified legal instructions."""
import numpy as np
from src.interaction.ai_observation import ai_observation
from src.interaction.sign_speech import sign_speech
from src.perception.lane_visual_roi import optional_lane_visual_roi
from src.utils.types import SignDetection,LaneInfo

def sign(code='P.130',source='temporal_best_crop',confidence=.92,recognized=.9):
    return SignDetection(code,confidence,(10,10,70,70),
                         recognition_source=source,
                         recognition_confidence=recognized,
                         context_verified=False,label_verified=False,
                         panel_role='primary',applicability='uncertain')

def test_observation_voice_reports_actual_name_but_does_not_announce_an_order():
    s=sign()
    text=ai_observation([s],video=True)
    assert 'P.130' in text and 'Cấm dừng xe' in text
    assert 'AI dự đoán' in text and 'chưa xác minh' in text
    assert sign_speech([s])==''

def test_video_narration_requires_temporal_known_code_and_confidence():
    assert ai_observation([sign(source='single_frame')],video=True)==''
    assert ai_observation([sign(code='unknown')],video=True)==''
    assert ai_observation([sign(recognized=.3)],video=True)==''
    assert ai_observation([sign(confidence=.3)],video=True)==''
    assert ai_observation([sign(source='background_rejected')],video=True)==''
    assert ai_observation([sign(code='P.130',source='single_frame')],video=False)

def test_observation_does_not_guess_numeric_speed_or_supplement_scope():
    assert ai_observation([sign(code='P.127')],video=True).find('60')==-1
    s=sign();s.panel_role='supplementary'
    assert ai_observation([s],video=True)==''

def test_roi_visual_only_can_show_candidate_but_not_ego_lane_scope():
    original=LaneInfo([],backend='ufld_v2')
    alternative=LaneInfo([],
        polylines=[[(300,110),(280,175),(260,245),(240,340)]],
        ego_boundaries=[[(300,110),(280,175),(260,245),(240,340)],
                        [(700,110),(750,175),(780,245),(800,340)]],
        backend='ufld_v2')
    class Learned:
        def detect(self,image,filter_geometry=False):
            assert filter_geometry is True
            assert image.shape==(518,1280,3)
            return alternative
    class Detector:
        learned=Learned()
    value,used=optional_lane_visual_roi(Detector(),np.zeros((720,1280,3),dtype=np.uint8),original,enabled=True)
    assert used
    assert len(value.polylines)==1
    assert value.backend=='ufld_v2_roi_visual_only'
    assert value.ego_boundaries==[] and value.lane_center_offset_px is None

def test_roi_not_applied_if_full_image_has_lanes_or_disabled():
    orig=LaneInfo([],polylines=[[(20,20),(25,90),(30,150)]],backend='ufld_v2')
    class Detector:
        learned=None
    returned,used=optional_lane_visual_roi(Detector(),np.zeros((720,1280,3),dtype=np.uint8),orig,enabled=True)
    assert returned is orig and not used
    empty=LaneInfo([],backend='ufld_v2')
    returned,used=optional_lane_visual_roi(Detector(),np.zeros((720,1280,3),dtype=np.uint8),empty,enabled=False)
    assert returned is empty and not used
