"""Hybrid model must never replace UFLD if lines exist, or invent ego scope."""
from pathlib import Path
import numpy as np
from src.utils.types import LaneInfo
from src.perception.lane_multimodel import SecondaryLaneVisual

def test_ufld_baseline_preserved_when_nonempty():
    model=SecondaryLaneVisual(Path('definitely_not_present_yolopv2.pt'))
    original=LaneInfo([],polylines=[[(10,20),(20,80),(30,140)]],backend='ufld_v2')
    out,used=model.detect_if_needed(np.zeros((720,1280,3),dtype=np.uint8),original)
    assert out is original and not used
    assert model.candidate is None

def test_fail_closed_without_secondary_checkpoint():
    model=SecondaryLaneVisual(Path('definitely_not_present_yolopv2.pt'))
    old=LaneInfo([],backend='ufld_v2')
    out,used=model.detect_if_needed(np.zeros((720,1280,3),dtype=np.uint8),old)
    assert out is old and not used and model.load_failure

def test_secondary_candidate_accepted_only_visual_no_ego():
    model=SecondaryLaneVisual('unused.pt')
    model.path=Path(__file__)  # mocked model exists
    class Candidate:
        def detect(self,image):
            line=[(450,round(720*(.30+i*.03))) for i in range(12)]
            return LaneInfo([],polylines=[line],
                            ego_boundaries=[line,line],backend='yolopv2')
    model.candidate=Candidate()
    input_image=np.full((720,1280,3),210,np.uint8)
    out,used=model.detect_if_needed(input_image,LaneInfo([],backend='ufld_v2'))
    assert used
    assert out.backend=='yolopv2_visual_candidate'
    assert out.ego_boundaries==[]
    assert out.lane_center_offset_px is None
    assert out.rejection_reason is not None

def test_short_dashboard_artifact_is_rejected():
    model=SecondaryLaneVisual('unused.pt')
    model.path=Path(__file__)
    class Candidate:
        def detect(self,image):
            line=[(400+i*6,round(720*(.70+i*.009))) for i in range(12)]
            return LaneInfo([],polylines=[line],backend='yolopv2')
    model.candidate=Candidate()
    result,used=model.detect_if_needed(np.full((720,1280,3),180,np.uint8),
                                         LaneInfo([],backend='ufld_v2'))
    assert not used and result.backend=='ufld_v2'

def test_dark_segment_not_promoted_to_pseudo_road():
    model=SecondaryLaneVisual('unused.pt')
    model.path=Path(__file__)
    class Candidate:
        def detect(self,image):
            return LaneInfo([],polylines=[[(350,150+i*24) for i in range(12)]],backend='yolopv2')
    model.candidate=Candidate()
    result,used=model.detect_if_needed(np.full((720,1280,3),25,np.uint8),
                                         LaneInfo([],backend='ufld_v2'))
    assert not used

def test_no_background_video_requires_no_hardware():
    from src.pipeline.orchestrator import Pipeline
    from inspect import signature
    assert signature(Pipeline.run).parameters['lane_visual_roi'].default is False
