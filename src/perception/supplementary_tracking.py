"""Bounded asynchronous plate OCR associated by stable primary track IDs."""
from copy import deepcopy
import threading,time
from src.perception.supplementary_motion import map_plate_rows


def read_video_with_motion(self,image,signs,*,stream_id,timestamp):
    primaries=[s for s in signs if not s.label.startswith('S') and s.label not in ('unknown','background')]
    primaries=primaries[:2]
    identity=tuple(sorted(((s.label,s.track_id) for s in primaries),key=lambda p:(p[0],-1 if p[1] is None else p[1])))
    key=stream_id;now=time.monotonic()
    with self.condition:
        if self.closed:return []
        previous=self.streams.get(key)
        compatible=previous is not None and bool(set(identity)&set(previous['identity']))
        reset=previous is None or timestamp<=previous['timestamp'] or previous['shape']!=image.shape or not compatible
        if reset:self.epoch+=1
        version=self.epoch if reset else previous['version']
        state={'timestamp':timestamp,'shape':image.shape,'identity':identity,'version':version,
               'submitted':previous['submitted'] if previous and not reset else -float('inf')}
        if reset:self.jobs.pop(key,None);self.completed.pop(key,None)
        self.streams[key]=state;self.streams.move_to_end(key)
        while len(self.streams)>8:
            old,_=self.streams.popitem(last=False);self.jobs.pop(old,None);self.completed.pop(old,None)
        result=[];done=self.completed.get(key)
        if done and done['version']==version and 0<=timestamp-done['timestamp']<=3:
            visible={(s.label,s.track_id):s.bbox_xyxy for s in primaries}
            for row in map_plate_rows(done['rows'],done['boxes'],visible,image.shape):
                result.append({**row,'cached':True,'observation_timestamp':done['timestamp'],
                               'ocr_age_ms':(timestamp-done['timestamp'])*1000})
        if primaries and now-state['submitted']>=1:
            self.jobs[key]=(image.copy(),deepcopy(primaries),timestamp,version)
            self.jobs.move_to_end(key);state['submitted']=now
            if self.worker is None:
                self.worker=threading.Thread(target=self._video_worker,daemon=True);self.worker.start()
            self.condition.notify()
        return result
