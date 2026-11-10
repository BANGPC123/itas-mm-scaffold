"""Compare sign-background refusal on separate AI-reviewed DEVELOPMENT crops.

Not blind human ground truth: video shares source with 4 training hard negatives.
Do not promote new checkpoint using this audit alone.
"""
from pathlib import Path
import cv2,math,torch,json
from torchvision.models import mobilenet_v3_small
from src.perception.sign_recognizer import crop_tensor
root=Path(__file__).resolve().parents[1]
review=root/'output/review/test_video_ai_review'
changes=json.loads((review/'changes.json').read_text(encoding='utf-8'))
removed=[v for v in changes if v['action']=='removed']
crop_data=[]
for v in removed:
    idx=int(v['frame'].split('_')[-1])
    img=cv2.imread(str(review/'images'/f'{idx:03}_0.jpg'))
    x1,y1,x2,y2=v['before']['bbox_xyxy']
    h,w=img.shape[:2]
    imgcrop=img[max(0,math.floor(y1)):min(h,math.ceil(y2)),max(0,math.floor(x1)):min(w,math.ceil(x2))]
    if imgcrop.size==0:continue
    crop_data.append((v['frame'],v['before']['label'],v['reason'],imgcrop))
result={'ground_truth':'AI visual development review, not human-verified',
        'source_video_independent':False,'n_removed':len(removed)}
for name,model_path in [
    ('deployed',root/'models/sign_recognizer/retrained_20261009.pt'),
    ('experimental',root/'output/retraining_20261011/recognizer_hardneg/candidate.pt')]:
    ckpt=torch.load(model_path,weights_only=True,map_location='cpu')
    net=mobilenet_v3_small(num_classes=len(ckpt['classes']))
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    output=[]
    with torch.inference_mode():
        for frame,before,reason,crop in crop_data:
            if min(crop.shape[:2])<16:
                output.append({'frame':frame,'previous_code':before,'reason':reason,'predicted':'unknown','too_small':True})
                continue
            p=net(crop_tensor(crop).unsqueeze(0)).softmax(1)[0]
            conf,cls=p.max(0)
            label=ckpt['classes'][int(cls)] if float(conf)>=.63 else 'unknown'
            output.append({'frame':frame,'previous_code':before,'reason':reason,
                          'predicted':label,'confidence':round(float(conf),5),'too_small':False})
    result[name]={'n':len(output),'known_false_accepts':sum(x['predicted']!='unknown' for x in output),
        'unknown_count':sum(x['predicted']=='unknown' for x in output),'details':output}
out=root/'output/evaluation/hardneg_ai_review_20261011.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'n':len(crop_data),'baseline_false_accepts':result['deployed']['known_false_accepts'],
   'candidate_false_accepts':result['experimental']['known_false_accepts'],
   'source_video_independent':False},ensure_ascii=False),flush=True)
