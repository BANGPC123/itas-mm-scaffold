import numpy as np
import torch
from src.perception.sign_recognizer import SignRecognizer
from src.utils.types import SignDetection


def test_crops_are_batched_and_small_signs_abstain():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes=[]
        def forward(self,x):
            self.batch_sizes.append(len(x))
            return torch.tensor([[0.,5.]]).repeat(len(x),1)
    recognizer=SignRecognizer.__new__(SignRecognizer)
    recognizer.model=Model()
    recognizer.config={'min_short_side':32}
    recognizer.device='cpu'
    recognizer.size=128
    recognizer.threshold=.7
    recognizer.classes=['P.130','P.131a']
    signs=[SignDetection('traffic_sign',.9,box) for box in [(0,0,40,40),(50,50,100,100),(0,0,10,10)]]
    result=recognizer.recognize(np.zeros((120,120,3),np.uint8),signs)
    assert recognizer.model.batch_sizes==[2]
    assert [s.label for s in result]==['P.131a','P.131a','unknown']
    assert result[2].recognition_confidence is None
    video_signs=[SignDetection('traffic_sign',.9,box) for box in [(0,0,20,20),(0,0,10,10)]]
    result=recognizer.recognize(np.zeros((120,120,3),np.uint8),video_signs,min_short_side=16)
    assert [s.label for s in result]==['P.131a','unknown']


def test_explicit_background_class_is_distinct_from_low_confidence():
    class Model(torch.nn.Module):
        def forward(self,x):return torch.tensor([[0.,10.],[0.,0.]])
    recognizer=SignRecognizer.__new__(SignRecognizer)
    recognizer.model=Model();recognizer.config={'min_short_side':16}
    recognizer.device='cpu';recognizer.size=128;recognizer.threshold=.7
    recognizer.classes=['P.130','unknown']
    signs=[SignDetection('traffic_sign',.9,(0,0,40,40)) for _ in range(2)]
    result=recognizer.recognize(np.zeros((50,50,3),np.uint8),signs)
    assert [s.label for s in result]==['unknown','unknown']
    assert result[0].recognition_source=='background_rejected'
    assert result[1].recognition_source=='low_confidence'
