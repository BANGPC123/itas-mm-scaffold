from pathlib import Path
import cv2,json
from src.pipeline.orchestrator import Pipeline
root=Path(__file__).resolve().parents[1]
source=root/'output/review/test_video_ai_review/images/020_0.jpg'
image=cv2.imread(str(source))
assert image is not None
pipe=Pipeline()
mask=[[320/image.shape[1],387/image.shape[0],420/image.shape[1],.65]]
res={}
for name,rect in [('without_exclusion',None),('with_camera_exclusion',mask)]:
    result=pipe.run(image,10.7769,106.7009,False,road_bottom=.65,annotate=False,exclude_rects=rect)
    res[name]={'signs':[{'label':s.label,'box':[round(float(v),1) for v in s.bbox_xyxy]} for s in result.perception.signs],
               'lane_count':len(result.perception.lane.polylines),
               'audio':bool(result.reasoning.speech_text)}
out=root/'output/evaluation/gauge_mask_real_snapshot_20261011.json'
out.write_text(json.dumps({'source':str(source),'mask':mask,'development_sample':True,
 'result':res,'not_field_validation':True},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False),flush=True)
