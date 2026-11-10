import numpy as np
import torch
from src.perception.sign_open_set import FrozenReferenceGate


def test_unsupported_reference_veto_does_not_assign_a_new_class(tmp_path):
    path=tmp_path/'references.npz'
    np.savez(path,positive=np.array([[1.,0.,0.]],np.float32),
        negative=np.array([[0.,1.,0.]],np.float32),labels=np.array(['P.127']))
    gate=FrozenReferenceGate(path,'cpu')
    feature=torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,1.,0.]])
    assert gate.reject(feature,['P.127','P.127','unknown'])==[False,True,False]


def test_class_specific_negative_does_not_veto_a_different_supported_code(tmp_path):
    path=tmp_path/'references.npz'
    np.savez(path,positive=np.array([[1.,0.,0.],[1.,0.,0.]],np.float32),
        negative=np.array([[0.,1.,0.]],np.float32),labels=np.array(['P.127','P.131a']),
        negative_targets=np.array(['P.127']),negative_kinds=np.array(['irrelevant']))
    gate=FrozenReferenceGate(path,'cpu')
    feature=torch.tensor([[0.,1.,0.],[0.,1.,0.]])
    assert gate.verdict(feature,['P.127','P.131a'])==['background_rejected',None]
