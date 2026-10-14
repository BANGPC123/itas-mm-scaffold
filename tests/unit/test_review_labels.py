import importlib.util
from pathlib import Path
import copy
import pytest

spec=importlib.util.spec_from_file_location('review_labels',Path(__file__).resolve().parents[2]/'scripts/evaluate_reviewed_context.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)

def packet():
    return {'source_sha256':'abc','samples':[{'id':'1','images':['a.jpg'],'width':100,'height':100,'timestamp_seconds':1,
        'human_confirmed':False,'reviewer':'','reviewed_at':None,'visibility':'clear','vehicle':'car','ego_boundaries':[],'signs':[]}]}

def test_machine_proposals_cannot_be_scored_as_ground_truth():
    p=packet()
    with pytest.raises(ValueError,match='No human-confirmed'):review.validate(p,p)

def test_review_must_preserve_frame_identity_and_reviewer():
    draft=packet();p=copy.deepcopy(draft);s=p['samples'][0];s['human_confirmed']=True
    with pytest.raises(ValueError,match='reviewer'):review.validate(p,draft)
    s.update(reviewer='Test reviewer',reviewed_at='2026-10-09T00:00:00Z')
    assert len(review.validate(p,draft))==1
    s['timestamp_seconds']=2
    with pytest.raises(ValueError,match='identity'):review.validate(p,draft)
