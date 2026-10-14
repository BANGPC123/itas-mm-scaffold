"""Frozen reference veto for unsupported video signs and advertising logos.

Positive references come only from released TRAIN annotations. Negative
references are explicitly AI-inspected development-video crops, never TEST GT.
This veto can abstain; it never assigns an unsupported new sign code.
"""
from pathlib import Path
import numpy as np


class FrozenReferenceGate:
    def __init__(self,path,device,margin=.015,min_similarity=.80):
        import torch
        bank=np.load(Path(path),allow_pickle=False)
        self.negative=torch.from_numpy(bank['negative']).to(device)
        self.positive=torch.from_numpy(bank['positive']).to(device)
        self.labels=bank['labels'];self.margin=margin;self.min_similarity=min_similarity
        self.targets=bank['negative_targets'] if 'negative_targets' in bank else np.full(len(self.negative),'*')
        self.kinds=bank['negative_kinds'] if 'negative_kinds' in bank else np.full(len(self.negative),'unsupported')

    def reject(self,embedding,labels):
        return [value is not None for value in self.verdict(embedding,labels)]

    def verdict(self,embedding,labels):
        import torch
        features=torch.nn.functional.normalize(embedding,dim=1)
        similarities=features@self.negative.T
        result=[]
        for i,label in enumerate(labels):
            selected=np.flatnonzero(self.labels==label)
            if not len(selected) or label=='unknown':result.append(None);continue
            negatives=np.flatnonzero((self.targets=='*')|(self.targets==label))
            if not len(negatives):result.append(None);continue
            best=negatives[int(similarities[i,negatives].argmax())]
            negative=similarities[i,best]
            positive=(features[i]@self.positive[selected].T).max()
            rejected=bool(negative>=self.min_similarity and negative>positive+self.margin)
            result.append(('background_rejected' if self.kinds[best]=='irrelevant' else 'outside_supported_reference') if rejected else None)
        return result
