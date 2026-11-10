"""Live API invariance/reproduction for lane & sign fusion."""
import json,cv2,requests
from pathlib import Path
root=Path(r'D:\itas-mm')
info=json.loads((root/'output/ai_visual_gt_20261014/selection.json').read_text(encoding='utf8'))
cases=[]
image=cv2.resize(cv2.imread(info['lane_0020']['image']),(960,540))
cases.append(('lane_0020',image))
cap=cv2.VideoCapture(r'C:\Users\KHANH BANG\Downloads\test_video.mp4')
for sec in (149.333,299.2,660.7,899.3):
 cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000);okay,im=cap.read()
 if not okay:raise RuntimeError('Cannot decode source '+str(sec))
 cases.append((str(sec),im))
cap.release()
rows=[]
for key,frame in cases:
 ok,b=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,95])
 assert ok
 modes=[]
 for secondary in (False,True):
  response=requests.post('http://127.0.0.1:8000/pipeline/run',
    files={'image':('frame.jpg',b.tobytes(),'image/jpeg')},
    data={'latitude':'10.7769','longitude':'106.7009','synthesize_audio':'false',
          'annotate':'false','road_bottom':'1','lane_visual_roi':'true',
          'auxiliary_sign_review':str(secondary).lower(),'lane_display_mode':'precision'},
    timeout=180)
  response.raise_for_status()
  result=response.json()
  signs=result['perception']['signs']
  modes.append({'main':[(s['label'],s['applicability'],s['car_context'].get('status')) for s in signs],
                'audio':result['reasoning']['speech_text'],
                'car_summary':result['reasoning']['car_summary'],
                'lanes':len(result['perception']['lane']['display_polylines'] or []),
                'extra':result['perception'].get('auxiliary_sign_candidates',[])})
 a,z=modes
 for field in ('main','audio','car_summary','lanes'):
  assert a[field]==z[field],(key,field)
 for extra in z['extra']:
  assert not extra['legal_warning_allowed'] and not extra['car_applicability_verified']
 rows.append({'frame':key,'main':a['main'],'lanes':a['lanes'],
  'secondary_candidates':[{'label':x['label_vi'],'agreement_status':x.get('agreement_status'),
    'recognized_label':x.get('recognizer_label')} for x in z['extra']],
  'main_unchanged':True})
 print('LIVE_FUSION',key,'lanes',a['lanes'],'main',[x[0] for x in a['main']],
    'secondary',rows[-1]['secondary_candidates'],flush=True)
assert rows[0]['lanes']==0,rows[0]
out=root/'output/evaluation/fusion_online_20261014.json'
out.write_text(json.dumps({'passed':True,'cases':rows,'independent_human_GT':False},ensure_ascii=False,indent=2),encoding='utf8')
print('FULL_API_SMOKE_PASS',out,flush=True)
