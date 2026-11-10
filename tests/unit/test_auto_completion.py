import numpy as np
import torch
from src.perception.clrnet_portable_nms import nms
from src.perception.supplementary_semantics import parse_supplementary
from scripts.build_auto_lane_corpus import consensus,mask_from_partial


def test_lane_nms_separates_nonoverlap_and_suppresses_close_offsets():
 boxes=torch.zeros((3,77));boxes[:,2]=0;boxes[:,4]=72
 boxes[0,5:]=100;boxes[1,5:]=110;boxes[2,5:]=250
 kept,number,_=nms(boxes,torch.tensor([.9,.8,.7]),50,4)
 assert kept.tolist()==[0,2] and number==2
 boxes[1,2]=1;boxes[1,4]=0
 kept,_,_=nms(boxes,torch.tensor([.9,.8,.7]),50,4)
 assert kept.tolist()==[0,1,2]


def test_ocr_conditions_remain_machine_candidates_with_exclusion():
 c=parse_supplementary(['TRỪ XE TẢI 06h00-22h00'],[.96])
 assert c['time_windows']==[['06:00','22:00']]
 assert c['vehicle_candidates']==['truck']
 assert c['condition_polarity']=='exclusion'
 assert not c['human_verified'] and not c['legal_effect_established']
 assert not parse_supplementary(['06:00-22:00'],[.4])['time_windows']
 assert parse_supplementary(['Gọi 0901234567'],[.99])['time_windows']==[]


def test_unknown_and_disagreement_never_become_lane_negatives():
 image=np.zeros((100,200,3),np.uint8)
 assert consensus(image,[],[])==[]
 assert np.all(mask_from_partial(image,[])==255)
 line={'points':[(50,50),(60,90)]}
 mask=mask_from_partial(image,[line])
 assert (mask==1).any() and (mask==0).any() and (mask==255).any()
 # Two agreeing networks without visual paint support are rejected.
 assert consensus(image,[line['points']],[line['points']])==[]
