import numpy as np
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('culane_metric',Path(__file__).resolve().parents[2]/'scripts/evaluate_culane.py')
metric=importlib.util.module_from_spec(spec)
spec.loader.exec_module(metric)
score,match_counts=metric.score,metric.match_counts


def test_exact_lane_and_disjoint_lane():
    left=[(200,590),(300,400),(400,250)]
    right=[(1500,590),(1400,400),(1300,250)]
    matrix=score([left,right],[left])
    assert matrix[0,0]==1
    assert matrix[0,1]==0
    assert match_counts(matrix,2,1)==(1,1,0)


def test_empty_cases_and_threshold_boundary():
    assert match_counts(np.empty((0,2)),2,0)==(0,2,0)
    assert match_counts(np.empty((2,0)),0,2)==(0,0,2)
    assert match_counts(np.array([[.5]]),1,1)==(0,1,1)
