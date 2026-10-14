"""Prepare the local product before declaring the HTTP server ready."""
from pathlib import Path
import json
import time


def warmup_product(pipeline, speech, *, root=None):
    import cv2
    from src.perception.sign_catalog import NAMES_VI
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    # Paint fitting solves tiny linear systems. Avoid a many-thread BLAS pool
    # competing with CUDA launches and the isolated OCR worker on a laptop.
    from threadpoolctl import threadpool_limits
    pipeline._cpu_blas_limit=threadpool_limits(limits=1,user_api='blas')
    pipeline.tts_engine.close()
    pipeline.tts_engine = speech
    speech.warmup()
    voice_started = time.perf_counter()
    for code, name in NAMES_VI.items():
        speech.synthesize(f'AI dự đoán có thể là biển {code}: {name}. '
                          'Đây chỉ là nhận diện tham khảo, chưa xác minh điều kiện áp dụng.')
    report = {'speech_provider': speech.provider, 'speech_cache_seconds': time.perf_counter() - voice_started,
              'cpu_blas_threads':1}
    # A separate stream warms the camera models; no user tracking state survives.
    summary = root / 'output/video_completion_20261011/surface_final_current/summary.json'
    if summary.is_file():
        source = Path(json.loads(summary.read_text(encoding='utf-8'))['source'])
        cap = cv2.VideoCapture(str(source))
        enabled = pipeline.auto_supplementary_enabled
        pipeline.auto_supplementary_enabled = False
        try:
            for timestamp in (0, 540, 900):
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Cannot decode the model warmup video')
                # Warm the same JPEG transport used by /pipeline/run. JPEG
                # slightly changes paint evidence and can activate a different
                # lazy model branch from the uncompressed source frame.
                encoded, blob = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not encoded:raise RuntimeError('Cannot encode the warmup frame')
                frame = cv2.imdecode(blob, cv2.IMREAD_COLOR)
                pipeline.run(frame, 10.7769, 106.7009, False, road_bottom=.65,
                             annotate=False, stream_id='server-warmup',
                             frame_timestamp=timestamp, lane_visual_roi=True)
            report['camera_warmup_frames'] = 3
        finally:
            cap.release()
            pipeline.auto_supplementary_enabled = enabled
            pipeline.stream_evidence.sessions.clear()
            pipeline._verified_zone_streams.clear()
    else:
        import numpy as np
        pipeline.run(np.full((720, 1280, 3), 100, dtype=np.uint8),
                     10.7769, 106.7009, False, annotate=False)
        report['camera_warmup_frames'] = 0
    if getattr(pipeline,'auto_supplementary_enabled',False):
        import numpy as np
        if getattr(pipeline,'supplementary_reader',None) is None:
            from src.perception.supplementary_reader import SupplementaryReader
            pipeline.supplementary_reader=SupplementaryReader()
        tick=time.perf_counter()
        # Start the isolated OCR models before /health becomes ready. No
        # traffic-sign or tracking state is introduced by this blank sample.
        pipeline.supplementary_reader._ocr(np.full((60,180,3),240,np.uint8))
        report['ocr_warmup_seconds']=time.perf_counter()-tick
        report['ocr_ready_before_http']=True
    report['startup_seconds'] = time.perf_counter() - started
    target = root / 'output/product_startup_status.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
