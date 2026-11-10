"""Order OCR fragments by printed line without inventing punctuation."""
import re,cv2,numpy as np


def printed_separator(image,left,right):
    if image is None:return False
    h,w=image.shape[:2];a,b=left[5],right[5]
    height=min(left[2],right[2]);yc=(left[0]+right[0])/2
    x1,x2=int(np.ceil(a[2]))+1,int(np.floor(b[0]))-1
    y1,y2=max(0,round(yc-height*.45)),min(h,round(yc+height*.45))
    if not 0<=x1<x2<=w or y2-y1<4:return False
    gray=cv2.cvtColor(image[y1:y2,x1:x2],cv2.COLOR_BGR2GRAY)
    if float(np.median(gray))<90 or np.ptp(gray)<40:return False
    _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    n,ids,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if bw>=max(4,height*.15) and 1<=bh<=height*.24 and bw/bh>=2.8 and area>=6:
            if x>0 and x+bw<gray.shape[1] and abs(y1+centers[i,1]-yc)<height*.28:return True
    return False


def printed_lines(texts,scores,boxes,image=None,recognize_line=None):
    if boxes is None or len(boxes)!=len(texts):return texts,scores
    entries=[]
    for text,score,box in zip(texts,scores,boxes):
        xy=np.asarray(box,float)
        if xy.shape!=(4,2) or not np.isfinite(xy).all():return texts,scores
        entries.append((float(xy[:,1].mean()),float(xy[:,0].min()),
                        max(1,float(np.ptp(xy[:,1]))),text,score,
                        (float(xy[:,0].min()),float(xy[:,1].min()),float(xy[:,0].max()),float(xy[:,1].max()))))
    groups=[]
    for entry in sorted(entries):
        candidates=[g for g in groups if abs(entry[0]-np.mean([v[0] for v in g]))<.55*min(entry[2],min(v[2] for v in g))]
        if len(candidates)==1:candidates[0].append(entry)
        else:groups.append([entry])
    groups.sort(key=lambda g:np.mean([v[0] for v in g]))
    lines=[];line_scores=[]
    for g in groups:
        ordered=sorted(g,key=lambda v:v[1]);words=[]
        for i,v in enumerate(ordered):
            if i and re.fullmatch(r'(?:[01]?\d|2[0-3]):[0-5]\d',v[3]) and re.fullmatch(r'(?:[01]?\d|2[0-3]):[0-5]\d',ordered[i-1][3]):
                if printed_separator(image,ordered[i-1],v):words.append('-')
            words.append(v[3])
        text=' '.join(words);confidence=float(min(v[4] for v in g))
        # Re-read an entire printed line when detection split its digits and
        # separator. The second recognition must retain both observed times.
        if recognize_line is not None and image is not None and len(g)>=2:
            from src.perception.plate_text_parser import parse_time_windows
            tokens=re.findall(r'(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?!\d)',text)
            if len(tokens)==2 and not parse_time_windows(text).complete:
                h,w=image.shape[:2]
                x1=max(0,int(min(v[5][0] for v in g))-3);x2=min(w,int(max(v[5][2] for v in g))+4)
                y1=max(0,int(min(v[5][1] for v in g))-3);y2=min(h,int(max(v[5][3] for v in g))+4)
                reread=recognize_line(image[y1:y2,x1:x2])
                if reread:
                    candidate,score=reread;parsed=parse_time_windows(candidate)
                    normalize=lambda times:[f'{int(t.split(":")[0]):02d}:{t.split(":")[1]}' for t in times]
                    values=re.findall(r'(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?!\d)',candidate)
                    if parsed.complete and normalize(values)==normalize(tokens) and score>=.85:
                        text=candidate;confidence=min(confidence,float(score))
        lines.append(text);line_scores.append(confidence)
    return lines,line_scores
