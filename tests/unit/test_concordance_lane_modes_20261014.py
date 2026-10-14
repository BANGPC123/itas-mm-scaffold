"""No cross-model vote can certify a sign's vehicle scope or legal speech."""
import numpy as np
import pytest
from src.utils.types import LaneInfo
from src.perception.lane_display_quality import visual_polylines
from src.perception.road_edge_veto import suppress_road_edge_contours
from src.perception.sign_concordance import label_agreement


class FakeRecognizer:
    def __init__(self,label,conf,source='single_frame'):
        self.label,self.conf,self.source=label,conf,source

    def recognize(self,image,signs,*,min_short_side):
        for s in signs:
            s.label=self.label
            s.recognition_confidence=self.conf
            s.recognition_source=self.source
        return signs


def _aux(conf=.95,sign_id=10):
    return {'bbox_xyxy':[20.,20.,100.,100.],'confidence':conf,
     'class_id':sign_id,'label_vi':'Cấm dừng và đỗ xe',
     'applicability':'uncertain','car_applicability_verified':False,'legal_warning_allowed':False}


def test_reject_advertisement_disagreement():
    out=label_agreement(FakeRecognizer('unknown',.99),np.zeros((200,200,3),np.uint8),[_aux()])
    assert out[0]['agreement_status']=='insufficient_evidence'
    assert not out[0]['legal_warning_allowed']


def test_high_agreement_remains_research_candidate():
    out=label_agreement(FakeRecognizer('P.130',.99),np.zeros((200,200,3),np.uint8),[_aux()])
    assert out[0]['agreement_status']=='cross_model_visual_agreement'
    assert out[0]['recognizer_label']=='P.130'
    assert out[0]['applicability']=='uncertain'
    assert out[0]['car_applicability_verified'] is False
    assert out[0]['legal_warning_allowed'] is False


@pytest.mark.parametrize('aux_conf,rec_conf',[(.7,.99),(.95,.7)])
def test_weak_votes_never_confirm(aux_conf,rec_conf):
    out=label_agreement(FakeRecognizer('P.130',rec_conf),
        np.zeros((200,200,3),np.uint8),[_aux(aux_conf)])
    assert out[0]['agreement_status']=='insufficient_evidence'


def test_unmapped_sign_cannot_confirm():
    out=label_agreement(FakeRecognizer('P.130',.99),
        np.zeros((200,200,3),np.uint8),[_aux(sign_id=12)])
    assert out[0]['agreement_status']=='insufficient_evidence'


def test_original_candidates_are_not_mutated():
    source=_aux()
    _=label_agreement(FakeRecognizer('P.130',.99),
        np.zeros((200,200,3),np.uint8),[source])
    assert 'agreement_status' not in source


def test_high_recall_lane_display_keeps_bounded_curves_without_creating_points():
    lane=LaneInfo([],polylines=[
        [(250+i*8,100+i*25) for i in range(13)],
        [(300+i*25,400+i*3) for i in range(15)],
    ],backend='ufld_v2')
    assert len(visual_polylines(lane,(720,1280,3),mode='precision'))==1
    assert visual_polylines(lane,(720,1280,3),mode='candidate')==lane.polylines


def test_high_recall_mode_filters_out_of_image_and_nan():
    lane=LaneInfo([],polylines=[[(0,0),(20,20),(40,40)],
        [(0,0),(60,5),(1300,4)],[(1,1),(2,2),(float('nan'),3)]])
    assert visual_polylines(lane,(720,1280,3),mode='candidate')==lane.polylines[:1]
    with pytest.raises(ValueError):
        visual_polylines(lane,(720,1280,3),mode='unreviewed')


def test_narrow_ground_mask_long_median_edge_is_suppressed():
    ground=np.zeros((300,400),dtype=np.float32)
    ground[40:290,100:132]=1.
    edge=[(103,y) for y in range(90,261,5)]
    internal=[(116,y) for y in range(90,261,5)]
    short=[(103,y) for y in range(100,135,5)]
    kept,removed=suppress_road_edge_contours([edge,internal,short],ground)
    assert removed==1
    assert kept==[internal,short]
