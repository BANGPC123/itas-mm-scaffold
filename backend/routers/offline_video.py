"""Opt-in, local-file offline video analysis; no webcam or realtime deadlines.

One job uses the same warmed-up inference worker as /pipeline/run, so models
are not duplicated on the 8 GB GPU. Output remains local and must pass decode
verification before being served.
"""
from __future__ import annotations
from pathlib import Path
from uuid import uuid4
import threading,json,hashlib
from datetime import datetime
import cv2
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from backend.routers.pipeline import get_pipeline,get_inference_worker,lock

router=APIRouter(prefix='/pipeline/offline-video',tags=['offline-video'])
ROOT=Path(__file__).resolve().parents[2]
DIR=ROOT/'output/offline_video_jobs'
jobs={}
registry=threading.RLock()
MAX_UPLOAD_BYTES=850*1024*1024
MAX_VIDEO_FRAMES=100000


def _run_job(job_id, source, output, analysis_hz, start_sec, max_sec, auxiliary_review, recorded_at):
    with registry:
        jobs[job_id]['status']='waiting_for_inference'
    # Hold the normal inference lock, avoiding model races between the
    # asynchronous renderer and the interactive frame-processing API.
    with lock:
        with registry:
            jobs[job_id]['status']='running'
        try:
            from scripts.render_offline_video_max_20261014 import render
            result=render(source,output,analysis_hz=analysis_hz,
                start_second=start_sec,max_seconds=max_sec,
                include_auxiliary=auxiliary_review,
                recorded_at=recorded_at,pipeline=get_pipeline())
            with registry:
                jobs[job_id]['status']='complete'
                jobs[job_id]['frames']=result['output_frames']
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception('Offline video analysis failed')
            with registry:
                jobs[job_id]['status']='failed'
                jobs[job_id]['error']=type(exc).__name__
                jobs[job_id]['message']='Render thất bại; xem nhật ký máy chủ. File nguồn vẫn giữ cục bộ.'


def _read_job(job_id):
    with registry:
        row=jobs.get(job_id)
        if row is None:raise HTTPException(404,'Không tìm thấy công việc xử lý video')
        state={k:v for k,v in row.items() if k not in ('source','out','future')}
    out=DIR/job_id/'result'
    progress=out/'progress.json'
    if progress.is_file():
        try:state['progress']=json.loads(progress.read_text(encoding='utf8'))
        except (json.JSONDecodeError,OSError):pass
    if state['status']=='complete':
        video=out/'test_video_offline_max.mp4'
        summary=out/'summary.json'
        if not video.is_file() or not summary.is_file():
            state['status']='failed'
            return state
        report=json.loads(summary.read_text(encoding='utf8'))
        if not report.get('verified_full_decode'):
            state['status']='failed'
            return state
        state['progress']={'encoded_source_frames':report['output_frames'],
                           'expected':report['output_frames'],
                           'samples':report['counts']['samples']}
        state['video_url']=f'/pipeline/offline-video/jobs/{job_id}/video'
        state['report_url']=f'/pipeline/offline-video/jobs/{job_id}/report'
        state['quality']={'analysis_hz':report['analysis_hz'],
           'sign_detection_passes':'960/640/1280','temporal_confirmation':True,
           'human_labeled_accuracy_available':False}
    return state


@router.post('/start',status_code=202)
def start_video_job(file:UploadFile=File(...),
                    analysis_hz:float=Form(6.,ge=1.,le=10.),
                    start_second:float=Form(0.,ge=0.),
                    max_seconds:float|None=Form(None,ge=1.,le=7200.),
                    auxiliary_review:bool=Form(True),
                    recorded_at:datetime|None=Form(None)):
    if recorded_at is not None and (recorded_at.tzinfo is None or recorded_at.utcoffset() is None):
        raise HTTPException(422,'Ngày giờ video phải có múi giờ.')
    # Deliberately only MP4 files: no arbitrary remote URLs or local paths.
    if not (file.filename or '').lower().endswith('.mp4'):
        raise HTTPException(415,'Hiện chỉ hỗ trợ MP4.')
    with registry:
        if any(job['status'] not in ('complete','failed') for job in jobs.values()):
            raise HTTPException(409,'Đang có một video được phân tích offline.')
    if lock.locked():raise HTTPException(409,'Inference đang bận; thử lại sau.')
    job_id=uuid4().hex
    folder=DIR/job_id;folder.mkdir(parents=True,exist_ok=False)
    source=folder/'source.mp4'
    count=0;sha=hashlib.sha256()
    try:
        with source.open('xb') as handle:
            while True:
                chunk=file.file.read(1024*1024)
                if not chunk:break
                count+=len(chunk)
                if count>MAX_UPLOAD_BYTES:
                    raise HTTPException(413,'Video vượt giới hạn 850 MB.')
                sha.update(chunk);handle.write(chunk)
        cap=cv2.VideoCapture(str(source))
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w,h=int(cap.get(3)),int(cap.get(4))
        cap.release()
        if not (5<=fps<=120 and 0<n<=MAX_VIDEO_FRAMES and
                256<=w<=3840 and 144<=h<=2160):
            raise HTTPException(422,'Video không hợp lệ hoặc vượt giới hạn 100.000 frame/4K.')
        if int(start_second*fps)>=n:
            raise HTTPException(422,'Thời điểm bắt đầu vượt thời lượng video.')
    except Exception:
        source.unlink(missing_ok=True)
        try:folder.rmdir()
        except OSError:pass
        raise
    with registry:
        jobs[job_id]={'job_id':job_id,'status':'queued','source':str(source),
                      'out':str(folder/'result'),'upload_bytes':count,
                      'source_sha256':sha.hexdigest()}
        future=get_inference_worker().submit(_run_job,job_id,source,
            folder/'result',analysis_hz,start_second,max_seconds,auxiliary_review,recorded_at)
        jobs[job_id]['future']=future
    return {'job_id':job_id,'status':'queued','source_bytes':count,
            'status_url':f'/pipeline/offline-video/jobs/{job_id}'}


@router.get('/jobs/{job_id}')
def video_job_status(job_id:str):
    return _read_job(job_id)


@router.get('/jobs/{job_id}/video')
def download_video(job_id:str):
    state=_read_job(job_id)
    if state['status']!='complete':raise HTTPException(409,'Video chưa được xác minh.')
    return FileResponse(DIR/job_id/'result/test_video_offline_max.mp4',
        media_type='video/mp4',filename='itas_mm_offline_max.mp4')


@router.get('/jobs/{job_id}/report')
def download_report(job_id:str):
    state=_read_job(job_id)
    if state['status']!='complete':raise HTTPException(409,'Báo cáo chưa sẵn sàng.')
    return FileResponse(DIR/job_id/'result/summary.json',
        media_type='application/json',filename='itas_mm_offline_quality_report.json')


@router.get('/demo-status')
def offline_demo_status():
    directory=ROOT/'output/offline_video_max_preview_20261014'
    summary=directory/'summary.json'
    video=directory/'test_video_offline_max.mp4'
    if not summary.is_file() or not video.is_file():return {'complete':False}
    info=json.loads(summary.read_text(encoding='utf8'))
    ready=bool(info.get('verified_full_decode') and
        info.get('output_frames')==info.get('decoded_frames')==900)
    return {'complete':ready,'frames':info.get('output_frames') if ready else None,
        'analysis_hz':info.get('analysis_hz') if ready else None,
        'duration_seconds':30 if ready else None,
        'video_url':'/pipeline/offline-video/demo-video' if ready else None,
        'multi_model_legal_approval':False}


@router.get('/demo-video')
def offline_demo_video():
    if not offline_demo_status().get('complete'):
        raise HTTPException(404,'Video mẫu chưa vượt kiểm tra giải mã')
    return FileResponse(ROOT/'output/offline_video_max_preview_20261014/test_video_offline_max.mp4',
        media_type='video/mp4')
