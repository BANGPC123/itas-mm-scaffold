"""Cached Unicode text patches for native-FPS evidence rendering."""
from collections import OrderedDict
import cv2,numpy as np
from PIL import Image,ImageDraw
from src.perception.sign_catalog import display_name


class VideoTextOverlay:
    def __init__(self,font,fps,analysis_hz):
        self.font=font;self.fps=fps;self.analysis_hz=analysis_hz
        self.cache=OrderedDict();self.footer_cache=OrderedDict()

    def patch(self,text,color):
        key=(text,color)
        if key not in self.cache:
            box=self.font.getbbox(text);width=max(1,round(self.font.getlength(text))+4)
            canvas=Image.new('RGB',(width,max(1,box[3]-box[1]+4)),(15,23,34))
            ImageDraw.Draw(canvas).text((2,2-box[1]),text,font=self.font,fill=color[::-1])
            self.cache[key]=cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)
        self.cache.move_to_end(key)
        while len(self.cache)>256:self.cache.popitem(last=False)
        return self.cache[key]

    @staticmethod
    def stamp(frame,patch,x,y):
        h,w=frame.shape[:2];ph,pw=patch.shape[:2]
        a,b=max(0,x),max(0,y);c,d=min(w,x+pw),min(h,y+ph)
        if a<c and b<d:frame[b:d,a:c]=patch[b-y:d-y,a-x:c-x]

    def footer(self,payload,width):
        texts=tuple(dict.fromkeys(s['label']+': '+s['car_context']['text'] for s in payload['signs']
            if s['label']!='unknown' and s.get('car_context')))
        key=(texts,width)
        if key not in self.footer_cache:
            wrapped=[]
            for text in texts:
                line=''
                for word in text.split():
                    if line and self.font.getlength(line+' '+word)>width-24:wrapped.append(line);line=word
                    else:line=(line+' '+word).strip()
                if line:wrapped.append(line)
            self.footer_cache[key]=wrapped[:3]
        self.footer_cache.move_to_end(key)
        while len(self.footer_cache)>64:self.footer_cache.popitem(last=False)
        return self.footer_cache[key]

    def draw(self,frame,payload,lines,boxes,timestamp):
        h,w=frame.shape[:2];dark=(34,23,15)
        for p in lines:
            if len(p)>=2:cv2.polylines(frame,[np.asarray(p,np.int32)],False,(40,220,255),2)
        labels=[]
        for sign,box in zip(payload['signs'],boxes):
            if box is None or min(box)<0:continue
            x1,y1,x2,y2=map(round,box);color=(120,120,120) if sign['label']=='unknown' else (0,220,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            patch=self.patch(display_name(sign['label']),color)
            labels.append((patch,max(4,min(x1,w-patch.shape[1]-8)),max(68,y1-22)))
        frame[:57]=dark
        header=f'ITAS-MM | {timestamp/60:.2f} phút | Video {self.fps:g} FPS | AI {self.analysis_hz:g} Hz | Ô tô con'
        self.stamp(frame,self.patch(header,(255,255,255)),10,4)
        view=payload['lane'].get('camera_view','unknown')
        message=('Camera quay ngang/vào cabin: không gán biển trong gương cho hướng đi.' if view=='side_or_cabin'
            else 'Chưa đủ bằng chứng hướng camera: không vẽ làn hoặc kết luận phạm vi biển.' if view=='unknown'
            else 'Vạch sơn quan sát được; không nối nét qua xe hoặc vùng thiếu bằng chứng.')
        self.stamp(frame,self.patch(message,(255,255,255)),10,30)
        for patch,x,y in labels:self.stamp(frame,patch,x,y)
        footer=self.footer(payload,w)
        if footer:
            top=h-24*len(footer)-8;frame[top:]=dark
            for i,line in enumerate(footer):self.stamp(frame,self.patch(line,(255,237,222)),10,top+4+24*i)
        return frame
