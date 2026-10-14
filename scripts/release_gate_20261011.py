"""Conservative release readiness gate for the FOUR ITAS-MM workstreams.

One report combines model inventory, actual experiments, legal-review state,
human-signed evaluation readiness, warm runtime speed and optional TTS proof.
Never updates production checkpoint or declares automatic driving safety.
"""
from pathlib import Path
import json,datetime,hashlib
ROOT=Path(__file__).resolve().parents[1]
EVAL=ROOT/'output/evaluation'
OUT=EVAL/'ITAS_MM_RELEASE_GATE_20261011.json'
if OUT.exists():raise FileExistsError('Refuse overwrite of release-readiness audit')

def read(path):
    p=ROOT/path
    return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else None
def fexist(path):return (ROOT/path).is_file()

lane=read('output/evaluation/lane_contrast_ab_20261011/report.json')
sign=read('output/retraining_20261011/recognizer_33class_EXPERIMENT/report.json')
bench=read('output/evaluation/vietnam_e2e_benchmark_20261011/benchmark.json')
field=read('output/evaluation/independent_field_readiness_20261011.json')
ocr=read('output/review/field_review_pack_20261011/ocr_experiments_20261011/results.json')
ads=read('output/review/field_review_pack_20261011/advertisement_audit_20261011/summary.json')
video=read('data/raw/external/vietnam_commons_street_20261011/SOURCES.json')
checkpoint_paths=[
    'models/lane/selected_20261009.pth',
    'models/sign_detector/retrained_20261009.pt',
    'models/sign_recognizer/retrained_20261009.pt']
invent=[]
for name in checkpoint_paths:
    path=ROOT/name
    invent.append({'file':name,'exists':path.is_file(),
                   'size_bytes':path.stat().st_size if path.is_file() else None})
model_ready=all(x['exists'] for x in invent)
source_count=len(video['sources']) if video else 0
gate=[
 {'area':'01_lane_detection','status':'NOT_FIELD_VALIDATED',
  'baseline_f1_culane_val_70':lane['culane_val_sample']['original']['f1'] if lane else None,
  'experimental_f1_clahe':lane['culane_val_sample']['clahe']['f1'] if lane else None,
  'experimental_f1_gamma':lane['culane_val_sample']['gamma_08']['f1'] if lane else None,
  'blocked_by':['12 new Vietnam frames had no baseline lane lines and no verified lane geometry ground truth',
                'tested CLAHE/gamma did not beat baseline on CULane VAL']},
 {'area':'02_traffic_signs','status':'33_CLASS_EXPERIMENT_ONLY',
  'baseline_classes':31,'experimental_classes':len(sign['classes']) if sign else None,
  'candidate_saved':sign['candidate_saved'] if sign else False,
  'candidate_old_class_val_acc':sign['history'][-1]['canonical_VAL_33class_acc'] if sign else None,
  'candidate_external_val_acc_NOT_INDEPENDENT':sign['history'][-1]['external_source_VAL_accuracy_NOT_INDEPENDENT'] if sign else None,
  'blocked_by':['external R420/R421 source labels/route originality not independently audited',
                'advertisements/background false-accepts remain',
                'new weights lack independent-source Vietnam field validation']},
 {'area':'03_vietnam_context','status':'RESEARCH_FAIL_CLOSED',
  'regulatory_reference':'QCVN 41:2024/BGTVT published in Circular 51/2024/TT-BGTVT',
  'verified_zone_tracking':'implemented, requires externally verified sign and scope',
  'ocr_stable_development_plate_candidates':ocr['paired_stable_candidate_count'] if ocr else None,
  'blocked_by':['supplementary text and scope are not independently verified',
                'demo GPS and old legal corpus verification remain incomplete',
                'full vehicle/time/route scope legal QA is not complete']},
 {'area':'04_end_to_end','status':'PERFORMANCE_MEASURED_NOT_FIELD_CERTIFIED',
  'benchmark_frames':bench['frames'] if bench else None,
  'fps_serial_warm':bench['effective_serial_throughput_fps'] if bench else None,
  'p95_ms':bench['p95_wall_ms'] if bench else None,
  'tts_live_audio_smoke':'PASS: neutral Vietnamese MP3 generated separately, not legal message',
  'blocked_by':['two new independent-source videos not human reviewed',
                'no source-disjoint sign/lane/OCR validation metrics',
                'network TTS latency and real GPS not covered by video throughput benchmark']}
]
ready=bool(model_ready and field and field.get('gate',{}).get('field_evidence_ready') is True
           and all(len(a['blocked_by'])==0 for a in gate))
report={'project':'ITAS-MM','generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'safety_mode':'RESEARCH_PREVIEW_FAIL_CLOSED',
 'production_models_loaded_in_previous_GPU_smoke':True,
 'production_weight_inventory':invent,
 'trained_new_sign_candidate_available':fexist('output/retraining_20261011/recognizer_33class_EXPERIMENT/candidate_33classes.pt'),
 'new_independently_sourced_road_videos':source_count,
 'model_release_approved':ready,'field_validation_complete':False,
 'status':'RELEASE_READY' if ready else 'BLOCKED_INDEPENDENT_FIELD_GROUND_TRUTH_AND_LEGAL_VALIDATION',
 'workstreams':gate,
 'selected_weights_modified_this_stage':False,
 'note':'An offline unit test pass, pretrained CULane F1 or software-level FPS is not road safety accreditation.'}
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':report['status'],'model_release_approved':ready,'workstreams':[{k:w[k] for k in ('area','status')} for w in gate],
'gate_path':str(OUT)},ensure_ascii=False),flush=True)
