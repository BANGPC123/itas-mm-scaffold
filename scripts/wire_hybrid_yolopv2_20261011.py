from pathlib import Path
p=Path(r'D:\itas-mm\src\pipeline\orchestrator.py')
s=p.read_text(encoding='utf-8')
old="""        if lane_visual_roi:
            from src.perception.lane_visual_roi import optional_lane_visual_roi
            lane,used=optional_lane_visual_roi(self.lane_detector,image,lane,enabled=True)
            if used:warnings.append('Đường làn ROI chỉ là gợi ý hiển thị thử nghiệm, không dùng làm bằng chứng làn áp dụng.')"""
new="""        if lane_visual_roi and lane.backend=='ufld_v2' and not lane.polylines:
            from src.perception.lane_multimodel import SecondaryLaneVisual
            if not hasattr(self,'secondary_lane_visual'):
                cfg_visual=load_config('perception').get('lane_visual_fallback',{})
                self.secondary_lane_visual=SecondaryLaneVisual(
                    cfg_visual.get('weights_path','models/lane/candidates/yolopv2_official_20261011/yolopv2.pt'),
                    device='cuda:0' if __import__('torch').cuda.is_available() else 'cpu',
                    roi=cfg_visual.get('roi',.8),threshold=cfg_visual.get('mask_threshold',.56),
                    min_span_ratio=cfg_visual.get('min_span_ratio',.20))
            lane,used=self.secondary_lane_visual.detect_if_needed(image,lane)
            if used:warnings.append('YOLOPv2: gợi ý đường làn chỉ để hiển thị, không dùng xác định phạm vi biển báo.')"""
assert old in s
s=s.replace(old,new,1)
p.write_text(s,encoding='utf-8')
p=Path(r'D:\itas-mm\configs\perception.yaml')
s=p.read_text(encoding='utf-8')
needle='lane_detector:\n'
assert needle in s
cfg="""lane_visual_fallback:
  # Optional dashboard-heavy MP4 VISUAL ONLY. Never use BDD mask for scope.
  model: yolopv2
  weights_path: models/lane/candidates/yolopv2_official_20261011/yolopv2.pt
  roi: 0.80
  mask_threshold: 0.56
  min_span_ratio: 0.20

"""
s=s.replace(needle,cfg+needle,1)
p.write_text(s,encoding='utf-8')
p=Path(r'D:\itas-mm\frontend\src\VideoPanel.jsx')
s=p.read_text(encoding='utf-8')
s=s.replace("perception.lane.backend==='ufld_v2_roi_visual_only'",
            "perception.lane.backend==='yolopv2_visual_candidate' || perception.lane.backend==='ufld_v2_roi_visual_only'")
s=s.replace("Gợi ý làn trên vùng đường (thử nghiệm, bật khi video có tap-lô lớn)",
            "YOLOPv2 bổ sung khi UFLD không thấy làn (đường vàng thử nghiệm)")
s=s.replace("Chế độ này chỉ thử UFLD-v2 trên phần đường phía trên khi model gốc không thấy làn.",
            "Chế độ này dùng YOLOPv2 segmentation trên vùng đường phía trên, chỉ khi UFLD-v2 gốc không thấy làn.")
p.write_text(s,encoding='utf-8')
print('HYBRID_BACKEND_FRONTEND_WIRED')
