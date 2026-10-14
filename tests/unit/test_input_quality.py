from unittest.mock import Mock
import numpy as np
import pytest
import torch
from src.perception.ufld_v2 import UFLDv2


@pytest.mark.parametrize('value', [0, 127, 255])
def test_uniform_frame_cannot_generate_lane_or_reach_network(value):
    model=UFLDv2.__new__(UFLDv2)
    model.device='cpu';model.dtype=torch.float32
    model.mean=torch.zeros(3,1,1);model.std=torch.ones(3,1,1)
    model.model=Mock(side_effect=AssertionError('Network must not run on a uniform frame'))
    result=model.detect(np.full((120,160,3),value,np.uint8))
    assert result.polylines==[] and result.ego_boundaries==[]
    assert result.rejection_reason=='insufficient_image_information'
    model.model.assert_not_called()
