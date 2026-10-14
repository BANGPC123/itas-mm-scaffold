"""Research sign recognition montage from already existing dev snapshots.

NOT continuous driving video; no claim of stable frame-to-frame prediction,
signed human ground truth, speed law, or penalty. Demonstrates sign + PDF
retrieval using the actual models, while keeping narration explicitly neutral.
"""
from pathlib import Path
import cv2,json,numpy as np,hashlib,subprocess
from src.pipeline.orchestrator import Pipeline
ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'output/review/field_review_pack_20261011'
OUT=ROOT/'output/capstone_sign_gallery_20261011'
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'summary.json').exists():raise FileExistsError('Do not overwrite prior montage')
manifest=json.loads((PACK/'review_manifest.json').read_text(encoding='utf-8'))
p=Pipeline()
fps=12;hold=1.1
out=OUT/'sign_gallery.mp4'
writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,720))
if not writer.isOpened():raise RuntimeError('MP4 writer not opened')
rows=[];n=0
try:
    for i,f in enumerate(manifest['frames']):
        img=cv2.imread(str(PACK/f['image']))
        if img is None or hashlib.sha256((PACK/f['image']).read_bytes()).hexdigest()!=f['image_sha256']:
            raise RuntimeError('Snapshot source changed')
        if img.shape[:2]!=(720,1280):
            img=cv2.resize(img,(1280,720),interpolation=cv2.INTER_AREA)
        result=p.run(img,10.7769,106.7009,False,road_bottom=.65,annotate=False)
        face=img.copy()
        for line in result.perception.lane.polylines:
            if len(line)>1:cv2.polylines(face,[np.asarray(line,np.int32)],False,(60,215,65),2)
        for s in result.perception.signs:
            a,b,c,d=map(int,s.bbox_xyxy)
            cv2.rectangle(face,(a,b),(c,d),(10,210,255),3)
            cv2.putText(face,s.label,(a,max(100,b-8)),cv2.FONT_HERSHEY_SIMPLEX,.83,
                        (10,210,255),2,cv2.LINE_AA)
        # Reader-visible metadata; no legal instructions inferred.
        cv2.rectangle(face,(0,0),(1280,82),(15,24,36),-1)
        cv2.putText(face,'ITAS-MM | Sign detection / lane / simulated GPS / offline legal retrieval',
            (16,28),cv2.FONT_HERSHEY_SIMPLEX,.71,(240,245,255),2)
        cv2.putText(face,f'DEVELOPMENT SNAPSHOTS, NOT A CONTINUOUS VIDEO   |  scene {i+1:02d}/24   |  legal refs: {len(result.reasoning.references)}',
            (16,58),cv2.FONT_HERSHEY_SIMPLEX,.64,(150,235,235),2)
        cv2.rectangle(face,(0,672),(1280,719),(17,23,32),-1)
        cv2.putText(face,'AI OUTPUT IS UNVERIFIED. No penalty or mandatory road action generated.',
                    (16,702),cv2.FONT_HERSHEY_SIMPLEX,.62,(210,215,230),2)
        for _ in range(round(fps*hold)):writer.write(face);n+=1
        rows.append({'frame':f['id'],'source_image':f['image'],'sign_codes':[s.label for s in result.perception.signs],
            'unknown_count':sum(s.label=='unknown' for s in result.perception.signs),
            'lane_polylines':len(result.perception.lane.polylines),
            'pdf_sources':[{'code':a.get('code'),'source':a.get('source_url'),'page':a.get('page')} for a in result.reasoning.references],
            'ground_truth_reviewed':False,'gps_is_simulated':True})
        print('SIGN_GALLERY',i+1,'signs',len(result.perception.signs),'refs',len(result.reasoning.references),flush=True)
finally:writer.release()
with (OUT/'events.jsonl').open('w',encoding='utf-8') as file:
    for e in rows:file.write(json.dumps(e,ensure_ascii=False)+'\n')
# Mux existing neutral demo narration (no new TTS network request).
narrated=None
try:
    import imageio_ffmpeg
    mp3=ROOT/'data/processed/tts_output/e1f8785b8764110bac55c2ffa6ca72017836759b8522a299d1135d28cab097e3.mp3'
    if mp3.is_file():
        narrated=OUT/'sign_gallery_with_narration.mp4'
        completed=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error',
            '-i',str(out),'-i',str(mp3),'-filter_complex','[1:a]apad[a]',
            '-map','0:v','-map','[a]','-c:v','copy','-c:a','aac','-t',str(n/fps),
            str(narrated)],capture_output=True,text=True,timeout=80)
        if completed.returncode or not narrated.is_file():narrated=None
except Exception:narrated=None
summary={'status':'PREVIEW_MONTAGE_COMPLETE','source':'24 nonconsecutive development snapshots',
    'is_continuous_road_video':False,'original_source_group':'test_video_development',
    'frames':n,'fps':fps,'scenes':len(rows),
    'observed_signs_total':sum(len(r['sign_codes']) for r in rows),
    'observed_known_code_total':sum(sum(x!='unknown' for x in r['sign_codes']) for r in rows),
    'observed_legal_citations':sum(len(r['pdf_sources']) for r in rows),
    'output_video':str(out),'output_narrated':str(narrated) if narrated else None,
    'video_verified_GT':False,'legal_fines_verified':False,'gps_simulated':True,
    'selected_weights_changed':False}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print('SIGN_GALLERY_READY',json.dumps(summary,ensure_ascii=False),flush=True)
