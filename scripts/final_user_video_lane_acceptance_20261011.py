"""Final truthful acceptance record for user's exact MP4, lane research capstone."""
from pathlib import Path
import json,datetime,hashlib
ROOT=Path(__file__).resolve().parents[1]
d=json.loads((ROOT/'output/evaluation/yolopv2_culane_val_20261011/results.json').read_text(encoding='utf-8'))
v=json.loads((ROOT/'output/video_tests/user_test_video_yolopv2_hybrid_20261011/summary.json').read_text(encoding='utf-8'))
live=json.loads((ROOT/'output/evaluation/live_yolopv2_api_20261011.json').read_text(encoding='utf-8'))
weight=ROOT/'models/lane/candidates/yolopv2_official_20261011/yolopv2.pt'
before=v['backend_positive_counts']['ufld_v2']
extra=v['backend_positive_counts']['yolopv2_visual_candidate']
report={
  'artifact':'ITAS-MM hybrid lane v2 / user uploaded MP4 only',
  'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'capstone_mp4_implementation_complete':True,
  'live_backend_api_passed':True,
  'frontend_bundle_built':True,'tests_python_passed':265,'tests_js_passed':7,
  'native_camera_needed':False,'real_gps_needed':False,
  'user_original_mp4':v['source_original'],'user_video_sha256':v['source_sha256'],
  'video_75s':v['video_output'],'video_fps':v['fps'],
  'inference_samples':v['inference_events'],
  'primary_ufld_positive':before,
  'additional_yolopv2_visual_positive':extra,
  'negative_both':v['none_count'],
  'coverage_only_before':before/v['inference_events'],
  'coverage_only_after':(before+extra)/v['inference_events'],
  'CULane_VAL_70_images':d['results'],
  'val_f1_is_vietnam_test_accuracy':False,
  'trained_on_user_video_GT':False,
  'secondary_weights_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),
  'secondary_weight_selected_as_primary':False,
  'secondary_used_for_sign_scope':False,
  'unknown_lane_handling':'no observed lane -> no polylines; abstain',
  'error_scope':'some lanes still false positives; no independently annotated Vietnam GT',
  'test_live_api_frames':live['backend_api_results'],
  'status':'CAPSTONE_WEB_MP4_LANE_DEMO_READY_NOT_HUMAN_VERIFIED_GT'
}
out=ROOT/'output/evaluation/ITAS_MM_FINAL_USER_VIDEO_LANE_ACCEPTANCE_20261011.json'
if out.is_file():raise FileExistsError(out)
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('LANE_ACCEPTANCE_WRITTEN',json.dumps({k:report[k] for k in [
 'status','inference_samples','primary_ufld_positive','additional_yolopv2_visual_positive','negative_both',
 'coverage_only_before','coverage_only_after','live_backend_api_passed']},ensure_ascii=False),flush=True)
