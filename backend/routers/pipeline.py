from dataclasses import asdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from io import BytesIO
import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from src.pipeline.orchestrator import Pipeline
from src.utils.config import load_config
from pydantic import BaseModel,Field,model_validator
from typing import Literal
from datetime import datetime
from src.utils.types import SignDetection,LaneInfo
from src.context.sign_applicability import SignApplicability
from backend.schemas import PipelineRunResponse

router=APIRouter(prefix='/pipeline',tags=['pipeline'])
lock=threading.Lock()
speech_lock=threading.Lock()

@lru_cache(maxsize=1)
def get_inference_worker():
    # CUDA/accelerator first-use state belongs to a thread. Keep model warmup
    # and every frame on the same worker even when a speech request takes an
    # ASGI worker thread. The nonblocking inference lock still rejects backlog.
    return ThreadPoolExecutor(max_workers=1,thread_name_prefix='itas-perception')

class SpeechRequest(BaseModel):
    text:str=Field(min_length=1,max_length=1200)

@lru_cache(maxsize=1)
def get_speech_engine():
    from src.interaction.tts_engine import TtsEngine
    return TtsEngine(load_config('interaction'))

@router.post('/speech')
def speech(request:SpeechRequest):
    # Speech synthesis must never hold the inference lock.
    if not speech_lock.acquire(blocking=False):raise HTTPException(429,'Đang tạo giọng đọc')
    try:
        path=get_speech_engine().synthesize(request.text)
        return {'audio_url':'/pipeline/audio/'+Path(path).name}
    except Exception as exc:
        raise HTTPException(503,'Bộ tạo giọng đọc tạm không khả dụng') from exc
    finally:speech_lock.release()

@router.get('/retraining')
def retraining_results():
    import json
    root=Path(__file__).resolve().parents[2]/'output/retraining_20261009'
    if not (root/'deployment.json').is_file():return {'available':False}
    def read(name):
        path=root/name
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
    product=root.parent/'product_20261010/summary.json'
    return {'available':True,'deployment':read('deployment.json'),
            'product_video':json.loads(product.read_text(encoding='utf-8')) if product.is_file() else None,
            'detector':read('detector_test.json'),'recognizer':read('recognizer_test.json'),
            'lane':read('lane/results.json'),'video':read('video_full/summary.json'),
            'latency':read('video_latency/results.json'),
            'video_url':'/pipeline/retrained-video' if (root/'video_full/test_video_annotated.mp4').is_file() else None}

@router.get('/retrained-video')
def retrained_video():
    root=Path(__file__).resolve().parents[2]
    path=root/'output/product_20261010/test_video_annotated.mp4'
    if not (path.parent/'summary.json').is_file():
        path=root/'output/retraining_20261009/video_full/test_video_annotated.mp4'
    if not path.is_file():raise HTTPException(404,'Video kiểm thử chưa hoàn thành')
    return FileResponse(path,media_type='video/mp4')


# Search immutable official-law PDF excerpts without requiring Ollama.
class LegalSearchRequest(BaseModel):
    query: str = Field(min_length=2,max_length=180)
    top_k: int = Field(default=3,ge=1,le=6)

@lru_cache(maxsize=1)
def get_offline_legal_index():
    from src.reasoning.offline_legal_retrieval import OfflineLegalRetriever
    return OfflineLegalRetriever()

@router.post('/legal-search')
def legal_search(request:LegalSearchRequest):
    retriever=get_offline_legal_index()
    hits=retriever.search_grounded(request.query,request.top_k)
    return {
        'query':request.query,'hits':hits,'retrieved_count':len(hits),
        'indexed_documents':retriever.manifest.get('source_stats',[]),
        'structured_sources':list(retriever.structured_index.sources.values()) if retriever.structured_index else [],
        'retrieval_backend':'task6_bm25' if retriever.structured_index else 'pdf_lexical',
        'research_only':True,'penalty_amount_verified':False,
        'warning':'Tài liệu gốc chính thức, nhưng chưa xác minh tất cả sửa đổi hoặc phạm vi áp dụng cụ thể.'
    }


# Two exact, pre-rendered capstone artifacts; no arbitrary filesystem path.
CAPSTONE_VIDEOS={
    'lane':'output/capstone_demo_20261011/video_with_narration.mp4',
    'signs':'output/capstone_sign_gallery_named_tts_20261011/video_named_with_tts.mp4',
    'user-hybrid':'output/video_tests/user_test_video_lane_dedup_checked_20261011/hybrid_lane_75s.mp4',
    'user-complete':'output/video_recheck_full_20261014/test_video_annotated.mp4',
    'user-fusion':'output/video_fusion_20261014/test_video_annotated.mp4',
    'independent-car':'output/video_quality_20261013/independent_car_v2/independent_car_annotated.mp4'
}

VIDEO_COMPLETION_DIR=Path(__file__).resolve().parents[2]/'output/video_recheck_full_20261014'
VIDEO_FUSION_DIR=Path(__file__).resolve().parents[2]/'output/video_fusion_20261014'
INDEPENDENT_VIDEO_DIR=Path(__file__).resolve().parents[2]/'output/video_quality_20261013/independent_car_v2'

@router.get('/video-completion-status')
def video_completion_status():
    import json
    def read(path):
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
    summary=read(VIDEO_COMPLETION_DIR/'summary.json')
    verification=read(VIDEO_COMPLETION_DIR/'verification.json')
    video=VIDEO_COMPLETION_DIR/'test_video_annotated.mp4'
    ready=bool(summary and verification and summary.get('complete_source_scan')
        and not summary.get('failures') and not summary.get('tts_failures')
        and verification.get('passed') and video.is_file()
        and verification.get('video_size')==video.stat().st_size
        and verification.get('video_mtime_ns')==video.stat().st_mtime_ns)
    root=Path(__file__).resolve().parents[2]
    sign_test=read(root/'output/video_quality_20261013/sign_TEST_frozen/report.json')
    independent=read(INDEPENDENT_VIDEO_DIR/'report.json')
    auxiliary=INDEPENDENT_VIDEO_DIR/'independent_car_annotated.mp4'
    independent_ready=bool(independent and independent.get('passed') and auxiliary.is_file()
        and independent.get('video_size')==auxiliary.stat().st_size
        and independent.get('video_mtime_ns')==auxiliary.stat().st_mtime_ns)
    return {'complete':ready,'summary':summary,'verification':verification,
        'sign_test':sign_test,'selected_sign_method':'rescue_12','independent_video_accuracy_available':False,
        'independent_video':{'complete':independent_ready,'summary':independent,
            'video_url':'/pipeline/capstone-video/independent-car' if independent_ready else None},
        'video_url':'/pipeline/capstone-video/user-complete' if ready else None,
        'report_url':'/pipeline/video-completion-report' if ready else None}

@router.get('/video-fusion-status')
def video_fusion_status():
    import json
    p=VIDEO_FUSION_DIR
    def get(name):
        path=p/name
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
    summary=get('summary.json')
    verification=get('verification.json')
    movie=p/'test_video_annotated.mp4'
    ready=bool(summary and verification and movie.is_file() and
        summary.get('complete_source_scan') and not summary.get('failures') and
        verification.get('passed') and verification.get('video_size')==movie.stat().st_size
        and verification.get('video_mtime_ns')==movie.stat().st_mtime_ns)
    return {'complete':ready,'summary':summary if ready else None,
            'video_url':'/pipeline/capstone-video/user-fusion' if ready else None,
            'ml_checkpoint_changed':False,'auxiliary_yolo11_video_included':False,
            'lane_veto_updated':True,'human_lane_gt_certified':False}

@router.get('/video-completion-report')
def video_completion_report():
    report=Path(__file__).resolve().parents[2]/'docs/ITAS_MM_INDEPENDENT_RELEASE_20261014.md'
    if not report.is_file():raise HTTPException(404,'Báo cáo kiểm thử chưa được xuất')
    return FileResponse(report,media_type='text/markdown; charset=utf-8',filename=report.name)

@router.get('/capstone-video/{kind}')
def capstone_video(kind:str):
    path=CAPSTONE_VIDEOS.get(kind)
    if not path:raise HTTPException(404,'No such capstone video')
    if kind=='user-fusion' and not video_fusion_status()['complete']:
        raise HTTPException(404,'Video fusion chưa vượt kiểm tra giải mã và checksum')
    if kind=='user-complete' and not video_completion_status()['complete']:
        raise HTTPException(404,'Video đầy đủ đang chờ kiểm tra cuối')
    if kind=='independent-car' and not video_completion_status()['independent_video']['complete']:
        raise HTTPException(404,'Video hành trình độc lập đang chờ kiểm tra')
    full=Path(__file__).resolve().parents[2]/path
    if not full.is_file():raise HTTPException(404,'Demo video not generated')
    return FileResponse(full,media_type='video/mp4',filename=full.name,
        content_disposition_type='inline')

@router.get('/capstone-status')
def capstone_status():
    import json
    root=Path(__file__).resolve().parents[2]
    p=root/'output/capstone_demo_20261011/OFFLINE_CAPSTONE_ACCEPTANCE.json'
    if not p.is_file():
        return {'offline_demo_ready':False,'reason':'acceptance_report_missing'}
    doc=json.loads(p.read_text(encoding='utf-8'))
    return {'offline_demo_ready':doc['offline_mp4_visual_demo_ready'],
        'research_goals_fully_validated':doc['full_proposal_research_validation_complete'],
        'camera_hardware_required':False,'real_gps_required':False,
        'videos':{key:'/pipeline/capstone-video/'+key for key in CAPSTONE_VIDEOS},
        'remaining_gaps':doc['still_unfinished']}

@router.get('/auxiliary-sign-status')
def auxiliary_sign_status():
    """Non-regulatory YOLO11 checkpoint evidence; safe even when weights are absent."""
    import hashlib,json
    root=Path(__file__).resolve().parents[2]
    cfg=load_config('perception').get('auxiliary_sign_review',{})
    weights=root/cfg.get('weights_path','__missing__')
    labels=root/cfg.get('labels_path','__missing__')
    info=json.loads(labels.read_text(encoding='utf-8')) if labels.is_file() else {}
    sha=hashlib.sha256(weights.read_bytes()).hexdigest() if weights.is_file() else None
    report=root/'output/import_yolo11_comment_20261014/val_localization.json'
    val=json.loads(report.read_text(encoding='utf-8')) if report.is_file() else None
    return {'enabled_for_opt_in_preview':bool(cfg.get('enabled',False)),
        'checkpoint_available':weights.is_file(),
        'checkpoint_sha256_matches_manifest':bool(sha and sha==info.get('source_sha256')),
        'classes':info.get('num_classes',0),'training_domain_not_revalidated':True,
        'review_only':True,'changes_primary_sign_detector':False,
        'affects_car_scope_or_legal_speech':False,
        'validation':val,'field_accuracy_certified':False}

@router.get('/sign-catalog')
def sign_catalog():
    from src.perception.sign_catalog import NAMES_VI
    return {'names':NAMES_VI,'detector':'YOLO11n (1 class: traffic_sign)',
            'recognizer':'MobileNetV3-Small (30 sign classes; background rejection when supported by checkpoint)','lane':'UFLD-v2 ResNet18',
            'edition':'Display names from project QCVN 41:2024 copy; applicability requires context'}

@router.get('/evaluation')
def evaluation():
    import json
    paths={
        'detector':'output/evaluation/detector_results.json',
        'recognizer':'output/evaluation/recognizer/results.json',
        'product':'output/evaluation/product/sign_pipeline_results.json',
    }
    if not all(Path(path).is_file() for path in paths.values()):return {'available':False}
    data={name:json.loads(Path(path).read_text()) for name,path in paths.items()}
    return {'available':True,'split':'TEST','snapshot':'original_baseline_before_20261009','current_runtime_accuracy_verified':False,
            'detector_map50_95':data['detector']['test']['metrics/mAP50-95(B)'],
            'recognizer_accuracy':data['recognizer']['test_all']['accuracy'],
            'recognizer_macro_f1':data['recognizer']['test_all']['macro_f1'],
            'correct_code_recall':data['product']['correct_code_recall_all_released'],
            'report_url':'/pipeline/report' if Path('output/pdf/DSP391m_FA26_Group2_Report3.pdf').is_file() else None}

@router.get('/report')
def model_report():
    path=Path('output/pdf/DSP391m_FA26_Group2_Report3.pdf')
    if not path.is_file():raise HTTPException(404,'Báo cáo chưa được xuất.')
    return FileResponse(path,media_type='application/pdf',filename=path.name,content_disposition_type='inline')

@lru_cache(maxsize=1)
def get_pipeline():return Pipeline()

@router.get('/status')
def status():
    import json
    cfg=load_config('perception')
    gate_file=Path(__file__).resolve().parents[2]/'output/evaluation/ITAS_MM_RELEASE_GATE_20261011.json'
    gate=json.loads(gate_file.read_text(encoding='utf-8')) if gate_file.is_file() else None
    return {'sign_detector':Path(cfg['sign_detector']['weights_path']).is_file(),
            'sign_recognizer':Path(cfg['sign_recognizer']['weights_path']).is_file(),
            'lane_detector':Path(cfg['lane_detector']['weights_path']).is_file(),
            'lane_backend':cfg['lane_detector'].get('backend'),
            'legal_corpus_validated':load_config('reasoning').get('validated_corpus',False),
            'task6_offline_retrieval_ready':(Path(__file__).resolve().parents[2]/'data/regulations/corpus.json').is_file(),
            'reviewed_scope_features':['vehicle_panel_intersection','S.508a/b_time_windows','P.131b/c_local_day_parity'],
            'offline_pdf_retrieval_ready':(Path('data/processed/legal_reference/official_corpus_20261011/manifest.json').is_file()),
            'field_validated':False,
            'release_readiness':{'status':gate['status'],'model_release_approved':gate['model_release_approved'],
                                 'audit_snapshot_utc':gate['generated_at_utc']} if gate else
                                {'status':'NOT_AUDITED','model_release_approved':False},
            'mode':'research_preview',
            'speech_policy':'reviewed_primary_sign_and_verified_scope_only',
            'message':'Các dự đoán chưa được chứng nhận cho điều khiển hoặc cảnh báo lái xe ngoài thực địa.'}

@router.post('/run',response_model=PipelineRunResponse)
def run_pipeline(image:UploadFile=File(...),latitude:float=Form(...,ge=-90,le=90),longitude:float=Form(...,ge=-180,le=180),synthesize_audio:bool=Form(True),road_bottom:float=Form(1.,ge=.3,le=1.),annotate:bool=Form(True),lane_visual_roi:bool=Form(False),lane_display_mode:Literal['precision','candidate']=Form('precision'),auxiliary_sign_review:bool=Form(False),stream_id:str|None=Form(None,max_length=64,pattern=r'^[A-Za-z0-9_-]+$'),frame_timestamp:float|None=Form(None,ge=0,allow_inf_nan=False),exclusion_zones:str=Form('[]',max_length=1200),observed_at:datetime|None=Form(None)):
    if observed_at is not None and (observed_at.tzinfo is None or observed_at.utcoffset() is None):
        raise HTTPException(422,'Ngày giờ quan sát cần có múi giờ; không dùng giờ phát lại làm giờ ghi hình.')
    if (stream_id is None)!=(frame_timestamp is None):raise HTTPException(422,'Video requires both stream_id and frame_timestamp')
    import json
    try:
        rects=json.loads(exclusion_zones)
        if not isinstance(rects,list) or len(rects)>8:raise ValueError()
        for r in rects:
            if not isinstance(r,list) or len(r)!=4 or not all(type(x) in (int,float) and np.isfinite(x) for x in r):raise ValueError()
            a,b,c,d=r
            if not(0<=a<c<=1 and 0<=b<d<=1):raise ValueError()
    except (ValueError,TypeError):
        raise HTTPException(422,'Invalid normalized camera exclusion zones')
    contents=image.file.read(10*1024*1024+1)
    if len(contents)>10*1024*1024:raise HTTPException(413,'Ảnh vượt quá 10 MB.')
    from PIL import Image,UnidentifiedImageError
    try:
        with Image.open(BytesIO(contents)) as header:
            if header.width*header.height>8_000_000:raise HTTPException(413,'Ảnh vượt quá 8 megapixel.')
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc:
        raise HTTPException(400,'Không đọc được ảnh.') from exc
    decoded=cv2.imdecode(np.frombuffer(contents,dtype=np.uint8),cv2.IMREAD_COLOR)
    if decoded is None:raise HTTPException(400,'Không đọc được ảnh.')
    if decoded.shape[0]*decoded.shape[1]>8_000_000:raise HTTPException(413,'Ảnh vượt quá 8 megapixel.')
    if not lock.acquire(blocking=False):raise HTTPException(429,'Đang xử lý khung hình trước. Vui lòng thử lại.')
    try:
        temporal={'stream_id':stream_id,'frame_timestamp':frame_timestamp} if stream_id is not None else {}
        if rects:temporal['exclude_rects']=rects
        if lane_visual_roi:temporal['lane_visual_roi']=True
        if lane_display_mode!='precision':temporal['lane_display_mode']=lane_display_mode
        if auxiliary_sign_review:temporal['auxiliary_sign_review']=True
        if observed_at is not None:temporal['observed_at']=observed_at
        result=get_inference_worker().submit(get_pipeline().run,decoded,latitude,longitude,synthesize_audio,
            road_bottom=road_bottom,annotate=annotate,**temporal).result()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception('Inference failed')
        raise HTTPException(503,'Chưa thể xử lý ảnh. Kiểm tra trạng thái mô hình và nhật ký máy chủ.') from exc
    finally:lock.release()
    response=asdict(result)
    # Task 6 RAG retains rich evidence internally; the desktop response
    # contract exposes plain excerpts plus source references separately.
    response['reasoning']['retrieved_chunks']=[c['text'] if isinstance(c,dict) else c for c in response['reasoning']['retrieved_chunks']]
    response['frame_shape']=list(decoded.shape)
    response['perception']['lane_line_count']=len(result.perception.lane.polylines)
    response['perception']['lane_center_offset_px']=result.perception.lane.lane_center_offset_px
    response['reasoning']['retrieved_chunk_count']=len(result.reasoning.retrieved_chunks)
    response['audio_url']='/pipeline/audio/'+Path(result.audio_path).name if result.audio_path else None
    response.pop('audio_path',None)
    return response

class ReviewedSign(BaseModel):
    label:str=Field(max_length=80)
    confidence:float=Field(ge=0,le=1,allow_inf_nan=False)
    recognition_confidence:float|None=Field(default=None,ge=0,le=1,allow_inf_nan=False)
    bbox_xyxy:tuple[float,float,float,float]
    panel_role:Literal['unknown','primary','supplementary']='unknown'
    allowed_vehicles:list[Literal['car','motorcycle','motorbike','truck','bus','van','bicycle']]=Field(default_factory=list,max_length=7)
    time_windows:list[tuple[str,str]]=Field(default_factory=list,max_length=8)
    overhead:bool|None=None
    placement:Literal['unknown','roadside','overhead']='unknown'
    lane_scope:Literal['unknown','all','ego','other']='unknown'
    label_verified:bool=False
    @model_validator(mode='after')
    def valid_box(self):
        x1,y1,x2,y2=self.bbox_xyxy
        if not all(np.isfinite(v) for v in self.bbox_xyxy) or x1<0 or y1<0 or x2<=x1 or y2<=y1:raise ValueError('Invalid bounding box')
        if self.time_windows:
            from src.context.time_scope import validate_windows
            if self.panel_role!='supplementary' or self.label not in ('S.508a','S.508b'):
                raise ValueError('Time windows require a supported time panel')
            validate_windows(self.time_windows)
        return self

class InterpretationRequest(BaseModel):
    signs:list[ReviewedSign]=Field(max_length=200)
    ego_boundaries:list[list[tuple[int,int]]]=Field(default_factory=list,max_length=2)
    width:int=Field(gt=0,le=10000)
    height:int=Field(gt=0,le=10000)
    vehicle:Literal['car','motorcycle','motorbike','truck','bus','van','bicycle']='car'
    observed_at:datetime|None=None
    @model_validator(mode='after')
    def validate_observation_time(self):
        if self.observed_at is not None and (self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None):
            raise ValueError('Observation time requires an explicit timezone offset')
        return self

@router.post('/interpret')
def interpret(request:InterpretationRequest):
    for sign in request.signs:
        if sign.bbox_xyxy[2]>request.width or sign.bbox_xyxy[3]>request.height:raise HTTPException(422,'Box outside image')
    if any(len(points)>200 or any(x<0 or y<0 or x>=request.width or y>=request.height for x,y in points) for points in request.ego_boundaries):raise HTTPException(422,'Invalid lane geometry')
    from src.perception.sign_catalog import NAMES_VI
    if any(s.label_verified and s.panel_role != 'supplementary' and s.label not in NAMES_VI for s in request.signs):
        raise HTTPException(422,'Verified primary sign code must be in supported Vietnamese sign catalog')
    if any(s.label_verified and s.panel_role == 'supplementary' and not (s.time_windows if s.label in ('S.508a','S.508b') else s.allowed_vehicles) for s in request.signs):
        raise HTTPException(422,'Supplementary panels need explicitly reviewed structured conditions')
    signs=[SignDetection(**s.model_dump(),context_verified=True) for s in request.signs]
    lane=LaneInfo([],ego_boundaries=request.ego_boundaries)
    result=SignApplicability(load_config('perception').get('applicability')).apply(signs,lane,(request.height,request.width,3),request.vehicle,observed_at=request.observed_at)
    from src.perception.sign_catalog import NAMES_VI
    for sign in result:sign.name_vi=NAMES_VI.get(sign.label)
    from src.interaction.sign_speech import sign_speech
    grounded=get_offline_legal_index().for_signs(result,'simulated_GPS_context')
    return {'signs':[asdict(s) for s in result],
            'speech_text':sign_speech(result),
            'guidance_text':grounded.guidance_text,
            'references':grounded.references,
            'retrieved_chunks':grounded.retrieved_chunks,
            'context_source':'human_review','legal_advice_generated':False,
            'penalty_verified':False}

@router.get('/audio/{filename}')
def audio(filename:str):
    if len(filename)!=68 or not filename.endswith('.mp3') or any(c not in '0123456789abcdef' for c in filename[:-4]):raise HTTPException(404)
    path=Path(load_config('interaction')['tts']['output_dir'])/filename
    if not path.is_file():raise HTTPException(404)
    return FileResponse(path,media_type='audio/mpeg')

@router.get('/automatic-research-status')
def automatic_research_status():
    """Published local experiment results; never promote weak labels to GT."""
    import json
    root=Path(__file__).resolve().parents[2]
    def read(relative):
        p=root/relative
        return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else None
    labels=read('data/processed/lane_auto_consensus_20261011/report.json')
    training=read('output/retraining_auto_20261011/report.json')
    comparison=read('output/evaluation/auto_lane_comparison_20261011/report.json')
    system=read('output/evaluation/auto_system_20261011_v2/report.json')
    retrieval=read('output/evaluation/auto_legal_20261011/retrieval.json')
    return {'automated_experiments_complete':all([labels,training,comparison,system,retrieval]),
            'independent_field_accuracy_validated':False,'labels':labels,
            'trained_models':list(training['models']) if training else [],
            'comparison':[{'model':k,'pixel_f1':v['pixel_f1']} for k,v in comparison['models'].items()] if comparison else [],
            'comparison_dataset':comparison['dataset'] if comparison else None,
            'system':{'frames':system['frames'],'failures':len(system['failures']),
                      'latency':system['latency'],'serial_inference_throughput_fps':system['serial_inference_throughput_fps']} if system else None,
            'recall_at_4':retrieval['recall_at_4'] if retrieval else None,
            'legal_currency_verified':False,'production_model_changed':False}
