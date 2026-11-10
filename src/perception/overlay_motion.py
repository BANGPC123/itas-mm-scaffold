"""Frame-to-frame image motion for exported paint/sign geometry.

Tracks only fresh inference geometry, validates reverse optical flow and
splits missing paint points. No extrapolation through occluders or frame cuts.
"""
import copy,cv2,numpy as np


class OverlayMotion:
    def __init__(self):
        self.gray=None;self.lines=[];self.boxes=[]

    def reset(self,frame,lines,boxes):
        self.gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.lines=copy.deepcopy(lines);self.boxes=copy.deepcopy(boxes)
        return self.lines,self.boxes

    def advance(self,frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);h,w=gray.shape
        if self.gray is None or gray.shape!=self.gray.shape:
            return self.reset(frame,[],[])
        points=[];line_spans=[];box_spans=[]
        for line in self.lines:
            line_spans.append((len(points),len(points)+len(line)));points.extend(line)
        for box in self.boxes:
            x1,y1,x2,y2=box
            box_spans.append((len(points),len(points)+5))
            points.extend([(x1,y1),(x2,y1),(x2,y2),(x1,y2),((x1+x2)/2,(y1+y2)/2)])
        if not points:
            self.gray=gray;return [],self.boxes
        old=np.asarray(points,np.float32).reshape(-1,1,2)
        new,status,error=cv2.calcOpticalFlowPyrLK(self.gray,gray,old,None,
            winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,15,.03))
        back,reverse,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,new,None,
            winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,15,.03))
        good=(status.ravel()>0)&(reverse.ravel()>0)&(error.ravel()<35)&(np.linalg.norm(back-old,axis=2).ravel()<1.8)
        new=new[:,0];old=old[:,0]
        good&=np.isfinite(new).all(1)&(new[:,0]>=0)&(new[:,0]<w)&(new[:,1]>=0)&(new[:,1]<h)
        good&=np.linalg.norm(new-old,axis=1)<max(w,h)*.09
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);lines=[]
        for lo,hi in line_spans:
            piece=[]
            for i in range(lo,hi):
                x,y=np.rint(new[i]).astype(int)
                supported=False
                if good[i] and 0<=x<w and 0<=y<h:
                    pixel=hsv[max(0,y-2):min(h,y+3),max(0,x-4):min(w,x+5)]
                    supported=bool(np.any(((pixel[:,:,1]<110)&(pixel[:,:,2]>130))|
                        ((pixel[:,:,0]>=10)&(pixel[:,:,0]<=40)&(pixel[:,:,1]>25)&(pixel[:,:,2]>90))))
                if supported:piece.append((int(x),int(y)))
                else:
                    if len(piece)>=2:lines.append(piece)
                    piece=[]
            if len(piece)>=2:lines.append(piece)
        boxes=[]
        for box,(lo,hi) in zip(self.boxes,box_spans):
            valid=good[lo:hi]
            if valid.sum()<3:boxes.append(None);continue
            delta=np.median((new-old)[lo:hi][valid],axis=0)
            x1,y1,x2,y2=box
            boxes.append([float(np.clip(x1+delta[0],0,w-1)),float(np.clip(y1+delta[1],0,h-1)),
                          float(np.clip(x2+delta[0],0,w-1)),float(np.clip(y2+delta[1],0,h-1))])
        self.gray=gray;self.lines=lines
        # Keep indices stable for label association; missing boxes are dropped
        # by the caller and replaced on the next fresh detector sample.
        self.boxes=[b if b is not None else [-10,-10,-9,-9] for b in boxes]
        return lines,boxes
