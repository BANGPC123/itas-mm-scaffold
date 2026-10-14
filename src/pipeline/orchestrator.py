from __future__ import annotations
import base64
import time
import cv2
import numpy as np
from src.context.zone_classifier import ZoneClassifier
from src.context.sign_applicability import SignApplicability
from src.interaction.tts_engine import TtsEngine
from src.perception.lane_detector import LaneDetector
from src.perception.sign_detector import SignDetector
from src.perception.sign_recognizer import SignRecognizer
from src.perception.sign_catalog import NAMES_VI,display_name
from src.utils.config import load_config
from src.utils.types import PerceptionResult, PipelineResult, ReasoningResult

class Pipeline:
    def __init__(self):
        import torch
        torch.set_num_threads(4)
        cfg=load_config('perception')
        self.sign_detector=SignDetector(cfg)
        self.still_detector_size=cfg.get('sign_detector',{}).get('image_size',640)
        self.video_detector_size=cfg.get('sign_detector',{}).get('video_image_size',self.still_detector_size)
        self.sign_recognizer=SignRecognizer(cfg)
        self.video_min_short_side=cfg.get('sign_recognizer',{}).get('video_min_short_side',32)
        self.supplementary_reader=None
        self.auto_supplementary_enabled=cfg.get('supplementary_ocr',{}).get('enabled',False)
        self.lane_detector=LaneDetector(cfg)
        self.zone_classifier=ZoneClassifier(load_config('context'))
        self.applicability=SignApplicability(cfg.get('applicability'))
        self.current_vehicle=cfg.get('applicability',{}).get('current_vehicle','car')
        self._verified_zone_streams={}  # Sign-derived R420/R421 state; independent of mock GPS
        self.tts_engine=TtsEngine(load_config('interaction'))
        self.rag_chain=None
        from src.perception.temporal_signs import StreamEvidence
        self.stream_evidence=StreamEvidence()
        self.reasoning_cfg=load_config('reasoning')
        from src.reasoning.offline_legal_retrieval import OfflineLegalRetriever
        self.offline_legal=OfflineLegalRetriever()
        if self.reasoning_cfg.get('validated_corpus',False):
            from src.reasoning.ollama_client import OllamaClient
            from src.reasoning.rag_chain import RagChain
            from src.reasoning.vector_store import VectorStore
            client=OllamaClient(self.reasoning_cfg['ollama'])
            self.rag_chain=RagChain(self.reasoning_cfg,VectorStore(self.reasoning_cfg['rag'],client),client)

    def run(self,image,latitude,longitude,synthesize_audio=True,*,road_bottom=1.,annotate=True,stream_id=None,frame_timestamp=None,exclude_rects=None,lane_visual_roi=False,observed_at=None):
        if image is None or image.size==0:raise ValueError('Input image is empty')
        if not np.isfinite(road_bottom) or not .3 <= road_bottom <= 1:
            raise ValueError('road_bottom must be between 0.3 and 1')
        warnings=[];timings={};started=time.perf_counter()
        road = image[:max(1, int(image.shape[0]*road_bottom))]
        if road_bottom < 1:
            warnings.append('Vùng dưới ranh camera chỉ loại khỏi tìm biển báo; mô hình làn sử dụng toàn khung hình.')
        if hasattr(self.sign_detector,'image_size'):
            self.sign_detector.image_size=(getattr(self,'video_detector_size',960) if stream_id is not None else getattr(self,'still_detector_size',640))
        secondary_signs=None
        if stream_id is not None and getattr(self.sign_detector,'video_multiscale',False):
            signs,secondary_signs=self.sign_detector.detect_views(road)
        else:signs=self.sign_detector.detect(road)
        if exclude_rects:
            h,w=image.shape[:2]
            def excluded(sign):
                x1,y1,x2,y2=sign.bbox_xyxy
                x=(x1+x2)/(2*w);y=(y1+y2)/(2*h)
                return any(a<=x<=c and b<=y<=d for a,b,c,d in exclude_rects)
            signs=[s for s in signs if not excluded(s)]
            if secondary_signs is not None:secondary_signs=[s for s in secondary_signs if not excluded(s)]
            warnings.append('Một số vùng camera bị loại trừ; biển thật trong vùng cũng sẽ không được nhận dạng.')
        timings['detection']=(time.perf_counter()-started)*1000
        tick=time.perf_counter()
        if secondary_signs is not None:
            from src.perception.sign_multiscale import recognize_views
            signs=recognize_views(self.sign_recognizer,road,signs,secondary_signs,min_short_side=self.video_min_short_side)
        elif stream_id is None:signs=self.sign_recognizer.recognize(road,signs)
        else:signs=self.sign_recognizer.recognize(road,signs,min_short_side=self.video_min_short_side)
        signs=[s for s in signs if s.recognition_source!='background_rejected']
        if stream_id is not None:
            if frame_timestamp is None:raise ValueError('Video frame timestamp is required')
            signs=self.stream_evidence.get(stream_id,road.shape,road_bottom).update(road,signs,frame_timestamp)
        timings['recognition']=(time.perf_counter()-tick)*1000
        tick=time.perf_counter();lane=self.lane_detector.detect(image)
        if lane_visual_roi and lane.backend=='ufld_v2' and load_config('perception').get('lane_visual_fallback',{}).get('camera_aware',False):
            from src.perception.camera_lane import CameraLaneVisual
            if not hasattr(self,'camera_lane_visual'):
                self.camera_lane_visual=CameraLaneVisual(self.lane_detector.learned,
                    device='cuda:0' if __import__('torch').cuda.is_available() else 'cpu')
            lane=self.camera_lane_visual.detect(image,lane)
            warnings.append('Vạch sơn theo vùng mặt đường và kiểm tra che khuất; chưa xác minh phạm vi làn áp dụng.')
        if lane_visual_roi and lane.backend=='ufld_v2':
            from src.perception.ufld_display_screen import screen_ufld_for_demo
            lane,_=screen_ufld_for_demo(image,lane)
        if lane_visual_roi and lane.backend=='ufld_v2' and not lane.polylines:
            from src.perception.lane_multimodel import SecondaryLaneVisual
            if not hasattr(self,'secondary_lane_visual'):
                cfg_visual=load_config('perception').get('lane_visual_fallback',{})
                self.secondary_lane_visual=SecondaryLaneVisual(
                    cfg_visual.get('weights_path','models/lane/candidates/yolopv2_official_20261011/yolopv2.pt'),
                    device='cuda:0' if __import__('torch').cuda.is_available() else 'cpu',
                    roi=cfg_visual.get('roi',.8),threshold=cfg_visual.get('mask_threshold',.56),
                    min_span_ratio=cfg_visual.get('min_span_ratio',.20))
            lane,used=self.secondary_lane_visual.detect_if_needed(image,lane)
            if used:warnings.append('YOLOPv2: gợi ý đường làn chỉ để hiển thị, không dùng xác định phạm vi biển báo.')
        from src.perception.lane_consistency import credible_ego
        if lane.backend == 'ufld_v2' and not credible_ego(lane.ego_boundaries,image.shape):
            lane.ego_boundaries=[]
            lane.lane_center_offset_px=None
        from src.perception.lane_display_quality import visual_polylines
        lane.display_polylines=lane.polylines if lane.backend=='camera_road_paint_visual' else visual_polylines(lane,image.shape)
        if stream_id is not None and lane.backend!='camera_road_paint_visual':
            from src.perception.lane_temporal import LaneStreamPool
            if not hasattr(self,'lane_stream_pool'):self.lane_stream_pool=LaneStreamPool()
            from src.perception.lane_display_quality import visual_polylines
            safe_display=lane.polylines if lane.backend=='camera_road_paint_visual' else visual_polylines(lane,image.shape)
            lane.display_polylines=self.lane_stream_pool.get(stream_id,image.shape,(road_bottom,lane.backend)).update(safe_display,frame_timestamp,image.shape,image=image)
        if lane_visual_roi and lane.backend=='camera_road_paint_visual':
            cfg_surface=load_config('perception').get('lane_visual_fallback',{})
            if cfg_surface.get('surface_veto',False) and not getattr(getattr(self,'camera_lane_visual',None),'uses_road_surface_mask',False):
                from src.perception.road_surface_veto import RoadSurfaceVeto
                if not hasattr(self,'road_surface_veto'):
                    self.road_surface_veto=RoadSurfaceVeto(cfg_surface['surface_veto_path'],
                        device='cuda:0' if __import__('torch').cuda.is_available() else 'cpu')
                lane.display_polylines,removed=self.road_surface_veto.filter(image,lane.display_polylines or [])
                if removed:warnings.append('Đã loại nét hiển thị trùng vùng vỉa hè do mô hình phân đoạn phát hiện.')
        timings['lane']=(time.perf_counter()-tick)*1000
        if lane_visual_roi and getattr(lane,'camera_view','forward')=='side_or_cabin':
            for sign in signs:
                sign.label='unknown';sign.recognition_source='side_or_reflected_view'
            if stream_id in self.stream_evidence.sessions:
                self.stream_evidence.sessions[stream_id][2].tracks=[]
            warnings.append('Camera đang quay ngang/vào cabin; biển trong gương không dùng xác định hướng đi hiện tại.')
        signs=self.applicability.apply(signs,lane,image.shape,self.current_vehicle,observed_at=observed_at)
        verified_zone_status='unknown'
        if stream_id is not None:
            from src.context.verified_zone_transitions import ZoneSignState
            if stream_id not in self._verified_zone_streams and len(self._verified_zone_streams)>=32:
                self._verified_zone_streams.pop(next(iter(self._verified_zone_streams)))
            tracker=self._verified_zone_streams.setdefault(stream_id,ZoneSignState())
            verified_zone_status=tracker.update(signs,stream_id=stream_id,timestamp=frame_timestamp)
        for sign in signs:sign.name_vi=display_name(sign.label).split(' — ',1)[-1]
        candidates=[]
        if getattr(self,'auto_supplementary_enabled',False):
            tick_ocr=time.perf_counter()
            if self.supplementary_reader is None:
                from src.perception.supplementary_reader import SupplementaryReader
                self.supplementary_reader=SupplementaryReader()
            if stream_id is not None:
                candidates=self.supplementary_reader.read_video(image,signs,stream_id=stream_id,timestamp=frame_timestamp)
            else:
                candidates=self.supplementary_reader.read(image,signs)
            timings['supplementary_ocr']=(time.perf_counter()-tick_ocr)*1000
        from src.context.car_context import assess_car,car_summary
        for sign in signs:
            sign.car_context=assess_car(sign,image,candidates,video=stream_id is not None,
                observed_at=observed_at,camera_view=getattr(lane,'camera_view','unknown') if lane_visual_roi else 'forward')
        perception=PerceptionResult(signs,lane,candidates)
        zone=self.zone_classifier.classify(latitude,longitude)
        if self.sign_detector._model is None:warnings.append('Bộ phát hiện biển báo chưa có trọng số đã huấn luyện.')
        if self.sign_recognizer.model is None:warnings.append('Bộ nhận dạng biển báo chưa sẵn sàng.')
        if lane.backend=='unavailable':warnings.append('Mô hình làn đường chưa sẵn sàng.')
        if lane.rejection_reason:warnings.append('Chưa đủ bằng chứng để xác định làn: '+lane.rejection_reason)
        warnings.append('Tọa độ GPS và vùng đường hiện là dữ liệu mô phỏng.')
        tick=time.perf_counter()
        applicable=[s for s in signs if s.applicability=='applies' and s.label!='unknown']
        if self.rag_chain and applicable:
            try:
                reasoning=self.rag_chain.answer('Phát hiện: '+ '; '.join(s.label for s in applicable),zone.zone_label)
            except Exception:
                warnings.append('Không kết nối được mô hình RAG. Dùng tra cứu PDF offline.')
                reasoning=self.offline_legal.for_signs(signs,zone.zone_label)
        else:
            if not hasattr(self,'offline_legal'):
                from src.reasoning.offline_legal_retrieval import OfflineLegalRetriever
                self.offline_legal=OfflineLegalRetriever()
            reasoning=self.offline_legal.for_signs(signs,zone.zone_label)
            if not getattr(self,'reasoning_cfg',{}).get('validated_corpus',False):
                warnings.append('Tra cứu văn bản offline dùng cho đồ án; hiệu lực điều khoản và mức phạt chưa kiểm chứng đầy đủ.')
        timings['reasoning']=(time.perf_counter()-tick)*1000
        from src.reasoning.sign_reference import SignReference
        if not hasattr(self, 'sign_reference'):
            self.sign_reference = SignReference()
        if not reasoning.references:
            reasoning.references=self.sign_reference.lookup([s.label for s in signs if s.label!='unknown'])
        from src.interaction.sign_speech import sign_speech
        reasoning.speech_text=sign_speech(signs)
        from src.interaction.ai_observation import ai_observation
        reasoning.observation_text=ai_observation(signs,video=stream_id is not None)
        reasoning.observation_speech_text=reasoning.observation_text
        reasoning.car_summary=car_summary(signs)
        audio=None;tick=time.perf_counter()
        if synthesize_audio and (reasoning.speech_text or reasoning.observation_speech_text):
            try:audio=self.tts_engine.synthesize(reasoning.speech_text or reasoning.observation_speech_text)
            except Exception:warnings.append('Giọng đọc trực tuyến tạm không khả dụng; nội dung vẫn hiển thị.')
        timings['tts']=(time.perf_counter()-tick)*1000
        annotated=None
        if annotate:
            overlay=image.copy()
            for points in (lane.display_polylines if lane.display_polylines is not None else lane.polylines):cv2.polylines(overlay,[np.array(points,np.int32)],False,(70,200,70),2)
            for sign in signs:
                x1,y1,x2,y2=map(int,sign.bbox_xyxy)
                color=(140,140,140) if sign.applicability=='ignored' else (0,190,255)
                cv2.rectangle(overlay,(x1,y1),(x2,y2),color,2)
                cv2.putText(overlay,sign.label,(x1,max(16,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
            ok,encoded=cv2.imencode('.jpg',overlay,[cv2.IMWRITE_JPEG_QUALITY,85])
            annotated='data:image/jpeg;base64,'+base64.b64encode(encoded).decode() if ok else None
        timings['total']=(time.perf_counter()-started)*1000
        return PipelineResult(perception,zone,reasoning,audio,warnings,timings,annotated,verified_zone_status=verified_zone_status)

    @staticmethod
    def _build_query_text(perception_result):
        return 'Phát hiện: '+ '; '.join(s.label for s in perception_result.signs if s.applicability=='applies')
