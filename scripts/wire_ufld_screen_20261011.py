from pathlib import Path
p=Path(r'D:\itas-mm\src\pipeline\orchestrator.py')
s=p.read_text(encoding='utf-8')
old="        tick=time.perf_counter();lane=self.lane_detector.detect(image)\n        if lane_visual_roi and lane.backend=='ufld_v2' and not lane.polylines:"
new="""        tick=time.perf_counter();lane=self.lane_detector.detect(image)
        if lane_visual_roi and lane.backend=='ufld_v2':
            from src.perception.ufld_display_screen import screen_ufld_for_demo
            lane,_=screen_ufld_for_demo(image,lane)
        if lane_visual_roi and lane.backend=='ufld_v2' and not lane.polylines:"""
assert old in s
s=s.replace(old,new,1)
p.write_text(s,encoding='utf-8')
print('SCREEN_BEFORE_SECONDARY_FALLBACK')
