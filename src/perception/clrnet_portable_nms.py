"""CPU lane-distance NMS following CLRNet's upstream CUDA overlap predicate.

Used only by the Windows research adapter; speed is not CUDA reproduction.
"""
import numpy as np
import torch


def nms(boxes, scores, overlap, top_k):
    if boxes.ndim != 2 or boxes.shape[1] != 77:
        raise ValueError('CLRNet NMS expects 72 offsets')
    rows=boxes.detach().float().cpu().numpy()
    order=np.argsort(-scores.detach().float().cpu().numpy(),kind='stable')
    keep=[]; parent=np.zeros(len(rows),dtype=np.int64); suppressed=set()
    for i in order:
        if int(i) in suppressed:continue
        keep.append(int(i));parent[i]=len(keep)
        a=rows[i];sa=int(a[2]*71+.5)
        ea=int(sa+a[4]-1+.5-int(a[4]-1<0))
        for j in order:
            if j==i or parent[j] or int(j) in suppressed:continue
            b=rows[j];sb=int(b[2]*71+.5)
            eb=int(sb+b[4]-1+.5-int(b[4]-1<0))
            start=max(sa,sb,0);end=min(ea,eb,71)
            if end>=start and np.abs(a[5+start:6+end]-b[5+start:6+end]).sum()<overlap*(end-start+1):
                suppressed.add(int(j));parent[j]=len(keep)
        if len(keep)>=top_k:break
    return (torch.tensor(keep,device=boxes.device,dtype=torch.long),len(keep),
            torch.tensor(parent,device=boxes.device,dtype=torch.long))
