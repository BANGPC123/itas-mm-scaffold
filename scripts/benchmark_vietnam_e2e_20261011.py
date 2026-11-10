"""Warm end-to-end pipeline benchmark on TWO NEW Vietnam-sourced videos.

Actual inference includes YOLO, MobileNet, UFLD, scope grouping and speech
decision; network TTS is deliberately disabled because no context GT is
verified. This is a PERFORMANCE check, not F1/false-alert ground truth.
"""
from pathlib import Path
import json,time,cv2,numpy as np,torch,psutil
from src.pipeline.orchestrator import Pipeline
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/evaluation/vietnam_e2e_benchmark_20261011'
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);cv2.setNumThreads(1)
sources=['commons_hanoi_20261011','commons_saigon_govap_20261011']
samples=[]
for source in sources:
    pack=ROOT/'output/review'/source
    data=json.loads((pack/'review_manifest.json').read_text(encoding='utf-8'))
    step=max(1,len(data['frames'])//10)
    for f in data['frames'][::step][:10]:
        image=cv2.imread(str(pack/f['image']))
        if image is None:raise RuntimeError('Missing '+f['image'])
        image=cv2.resize(image,(1280,720),interpolation=cv2.INTER_AREA)
        samples.append((source,f['id'],float(f['timestamp_seconds']),image))
t0=time.perf_counter();p=Pipeline();load=time.perf_counter()-t0
p.run(samples[0][3],10.7769,106.7009,False,road_bottom=.65,annotate=False)
rows=[]
for source,frame,timestamp,image in samples:
    before=time.perf_counter()
    result=p.run(image,10.7769,106.7009,False,road_bottom=.65,
                 annotate=False,stream_id=source,frame_timestamp=timestamp)
    duration=(time.perf_counter()-before)*1000
    if result.audio_path is not None:
        raise AssertionError('No audio allowed without approved verified applicability')
    rows.append({'source_group':source,'frame':frame,'timestamp_seconds':timestamp,
        'wall_ms':round(duration,2),'pipeline_reported_ms':round(result.timings_ms.get('total',0),2),
        'component_ms':result.timings_ms,'signs':len(result.perception.signs),
        'lane_lines':len(result.perception.lane.polylines),
        'ego_boundaries':len(result.perception.lane.ego_boundaries),
        'applicable_signs':sum(s.applicability=='applies' for s in result.perception.signs),
        'speech_text_nonempty':bool(result.reasoning.speech_text),
        'tts_network_called':False})
    print('FRAME_BENCH',source,frame,round(duration,1),flush=True)
ms=np.array([r['wall_ms'] for r in rows])
report={'scope':'WARM_INFERENCE_PIPELINE_ONLY_2_PUBLIC_SOURCES',
    'frames':len(rows),'resolution':'1280x720','model_load_seconds':round(load,2),
    'mean_wall_ms':round(float(ms.mean()),2),
    'median_wall_ms':round(float(np.median(ms)),2),
    'p95_wall_ms':round(float(np.percentile(ms,95)),2),
    'effective_serial_throughput_fps':round(float(1000/ms.mean()),2),
    'p95_frame_equiv_fps':round(float(1000/np.percentile(ms,95)),2),
    'cuda_mem_allocated_mb':round(torch.cuda.memory_allocated()/1024**2,2) if torch.cuda.is_available() else None,
    'process_rss_mb':round(psutil.Process().memory_info().rss/1024**2,2),
    'observed_applicable_signs':sum(r['applicable_signs'] for r in rows),
    'audio_calls':0,'gps_is_mocked':True,
    'ground_truth_metrics_available':False,
    'external_field_F1_status':'NOT_EVALUABLE_WITHOUT_REVIEWER_GROUND_TRUTH',
    'rows':rows}
(OUT/'benchmark.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('BENCHMARK_DONE',json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False),flush=True)
