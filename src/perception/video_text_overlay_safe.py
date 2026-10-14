"""Place explanatory text away from visible detected signs and paint."""
import cv2,numpy as np
from src.perception.video_text_overlay import VideoTextOverlay
from src.perception.sign_catalog import display_name


class SafeVideoTextOverlay(VideoTextOverlay):
    def draw(self,frame,payload,lines,boxes,timestamp):
        h,w=frame.shape[:2];dark=(34,23,15)
        for p in lines:
            if len(p)>=2:cv2.polylines(frame,[np.asarray(p,np.int32)],False,(40,220,255),2)
        labels=[];visible=[]
        for sign,box in zip(payload['signs'],boxes):
            if box is None or min(box)<0:continue
            x1,y1,x2,y2=map(round,box);visible.append((x1,y1,x2,y2))
            color=(120,120,120) if sign['label']=='unknown' else (0,220,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            patch=self.patch(display_name(sign['label']),color)
            if y1<24 and x1>patch.shape[1]+8:
                label_x,label_y=x1-patch.shape[1]-4,max(0,y1)
            elif y1<24 and x2+patch.shape[1]+8<w:
                label_x,label_y=x2+4,max(0,y1)
            else:label_x,label_y=max(4,min(x1,w-patch.shape[1]-8)),y2+3 if y1<24 else max(0,y1-22)
            labels.append((patch,label_x,label_y))
        footer=self.footer(payload,w);footer_top=h-24*len(footer)-8 if footer else h
        header=f'ITAS-MM | {timestamp/60:.2f} phút | Video {self.fps:g} FPS | AI {self.analysis_hz:g} Hz | Ô tô con'
        view=payload['lane'].get('camera_view','unknown')
        message=('Camera quay ngang/vào cabin: không gán biển trong gương cho hướng đi.' if view=='side_or_cabin'
            else 'Chưa đủ bằng chứng hướng camera: không vẽ làn hoặc kết luận phạm vi biển.' if view=='unknown'
            else 'Vạch sơn quan sát được; không nối nét qua xe hoặc vùng thiếu bằng chứng.')
        patches=[self.patch(t,(255,255,255)) for t in [header,message]]
        width=min(w-20,max(p.shape[1] for p in patches));height=54
        candidates=[(10,4),(max(10,w-width-10),4),(10,max(4,footer_top-height-4)),(max(10,w-width-10),max(4,footer_top-height-4))]
        def obstruction(position):
            x,y=position
            overlap=sum(max(0,min(x+width,c)-max(x,a))*max(0,min(y+height,d)-max(y,b)) for a,b,c,d in visible)
            paint=sum(x<=px<x+width and y<=py<y+height for line in lines for px,py in line)
            return overlap+100*paint
        x,y=min(candidates,key=obstruction)
        # Only glyph patches have opaque backgrounds; the rest of the image
        # remains visible. A sign near the top edge is never covered by a band.
        for j,patch in enumerate(patches):self.stamp(frame,patch,x,y+26*j)
        for patch,x,y in labels:self.stamp(frame,patch,x,y)
        if footer:
            frame[footer_top:]=dark
            for i,line in enumerate(footer):self.stamp(frame,self.patch(line,(255,237,222)),10,footer_top+4+24*i)
        return frame
