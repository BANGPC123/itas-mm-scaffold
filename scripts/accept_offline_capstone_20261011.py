"""Capstone offline acceptance, independent of automotive field deployment.

Check runnable artifacts and exact remaining proposal research gaps. Not an
independent road-driving accuracy or penalty correctness certificate.
"""
from pathlib import Path
import datetime,json,cv2,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/capstone_demo_20261011/OFFLINE_CAPSTONE_ACCEPTANCE.json'
if OUT.is_file():raise FileExistsError('Refuse overwrite of existing acceptance report')
def exists(x):
    p=ROOT/x
    return p.is_file() and p.stat().st_size>0
weights={k:exists(v) for k,v in {
    'YOLO11n':'models/sign_detector/retrained_20261009.pt',
    'MobileNetV3':'models/sign_recognizer/retrained_20261009.pt',
    'UFLD-v2':'models/lane/selected_20261009.pth'}.items()}
pdf_meta=json.loads((ROOT/'data/processed/legal_reference/official_corpus_20261011/manifest.json').read_text(encoding='utf-8'))
docs={r['document']:r['pages_with_extracted_text'] for r in pdf_meta['source_stats']}
videos={}
for key,path in {'lane_demo':'output/capstone_demo_20261011/video_with_narration.mp4',
                 'sign_gallery':'output/capstone_sign_gallery_20261011/sign_gallery_with_narration.mp4'}.items():
    cap=cv2.VideoCapture(str(ROOT/path))
    videos[key]={'file':path,'playable':cap.isOpened(),
        'frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps':round(cap.get(cv2.CAP_PROP_FPS),2)}
    cap.release()
ready=all(weights.values()) and all(v['playable'] and v['frames']>=120 for v in videos.values()) and (
    exists('frontend/dist/index.html') and
    exists('data/processed/legal_reference/official_corpus_20261011/chunks.jsonl') and
    pdf_meta['chunks']>=500)
report={'type':'University capstone Web/Desktop research demo',
    'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'camera_hardware_needed':False,'real_gps_needed':False,
    'simulated_gps':True,'offline_mp4_visual_demo_ready':ready,
    'selected_weights':weights,'video_artifacts':videos,
    'source_pdf_count':pdf_meta['source_count'],'retrievable_corpus_chunks':pdf_meta['chunks'],
    'pdf_pages_extracted_per_document':docs,
    'official_pdf_originals_preserved':True,'legal_latest_amendments_verified':False,
    'penalties_auto_readable_from_decree_168':False,
    'full_proposal_research_validation_complete':False,
    'still_unfinished':['Scanned 168/2024/NĐ-CP has no searchable text; penalty mapping cannot be truthfully verified automatically.',
        'SAHI/contrastive model improvements lack validated ablations on held-out sign frames.',
        'Human-verified accuracy for compound/small signs and Vietnamese video lane output is not yet measured.',
        'No Ollama service; current PDF-grounded retrieval returns extracts and citations without open-ended LLM generation.'],
    'safety_scope':'Demonstration from MP4 only; should not be used as road guidance',
    'status':'OFFLINE_CAPSTONE_DEMO_READY_WITH_RESEARCH_LIMITATIONS' if ready else 'OFFLINE_CAPSTONE_DEMO_BLOCKED'}
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('ACCEPTANCE_RESULT',json.dumps(report,ensure_ascii=False),flush=True)
