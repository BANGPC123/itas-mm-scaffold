"""Generate a local interactive review surface for AI-labeled frames.

Review decisions are human-supplied ONLY after an actual person opens the HTML
and explicitly approves/corrects each frame. Generation does not sign records.
"""
from pathlib import Path
import json
import cv2

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/'output/ai_visual_gt_20261014'

def main():
    labels=json.loads((DIR/'ai_visual_labels.json').read_text(encoding='utf8'))
    index=json.loads((DIR/'selection.json').read_text(encoding='utf8'))
    images=DIR/'source_images'
    images.mkdir(exist_ok=True)
    for name,meta in index.items():
        output=images/(name+'.jpg')
        if not output.exists():
            im=cv2.imread(meta['image'])
            if im is None:raise RuntimeError(name)
            w,h=meta['label_image_size']
            im=cv2.resize(im,(w,h),interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(output),im,[cv2.IMWRITE_JPEG_QUALITY,94]):
                raise RuntimeError('Unable to render '+name)
    manifest={'annotation_provenance':labels['annotation_kind'],
              'human_verified':False,
              'source_images_staged_locally':True,
              'records':{key:dict(index[key],**a) for key,a in labels['annotations'].items()}}
    for key,row in manifest['records'].items():
        row['image']='source_images/'+key+'.jpg'
    payload=json.dumps(manifest,ensure_ascii=False).replace('</','<\\/')
    html=r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>ITAS-MM • Xác nhận nhãn làn đường</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:system-ui,Arial,sans-serif;background:#10151f;color:#e8edf4;margin:0;padding:18px}
header{max-width:1250px;margin:0 auto 16px}h1{font-size:23px;margin:2px 0 10px}
.warn{background:#392b15;border:1px solid #886224;border-radius:7px;padding:12px;color:#ffe1a1}
main{display:grid;grid-template-columns:minmax(500px,960px) minmax(250px,350px);gap:18px;max-width:1350px;margin:auto}
#canvas{width:100%;height:auto;border:1px solid #46516a;cursor:crosshair;background:#222;display:block}
aside{border:1px solid #374055;padding:14px;border-radius:9px;background:#1d2534}
button,select,input,textarea{background:#263550;color:white;border:1px solid #586b88;border-radius:6px;padding:8px;font:inherit}
button{cursor:pointer}button:hover{background:#354761}label{display:block;margin:10px 0 4px}
button.primary{background:#21665b}button.warnbtn{background:#763c2b}
.row{display:flex;gap:7px;flex-wrap:wrap;margin:9px 0}
input,select,textarea{max-width:100%;box-sizing:border-box;width:100%}
small{opacity:.8}#frameNote{min-height:80px}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0;font-size:13px}
.green{color:#72f878}.red{color:#ff7878}.pink{color:#ff7ced}
#frameList{max-height:170px;overflow:auto;font-size:12px;margin-top:10px}
#frameList button{font-size:12px;padding:4px;margin:2px}
@media(max-width:870px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>ITAS-MM — Rà soát 36 frame có nhãn AI</h1>
<div class="warn">⚠ Đây là nhãn do ChatGPT xem ảnh và chọn tọa độ. <b>Chưa phải human ground truth.</b>
Chỉ ghi nhận xác nhận khi người kiểm tra thực sự xem từng ảnh, sửa lỗi và nhấn Xác nhận. Không dùng nhãn này cho chứng nhận lái xe.</div></header>
<main><section><div class="row"><button id="prev">← Trước</button><button id="next">Tiếp →</button>
<strong id="title">...</strong></div><canvas id="canvas" width="960" height="540"></canvas>
<div class="legend"><span class="green">━━ Vạch dọc nhìn thấy</span><span class="red">━━ Vạch mép đường</span><span class="pink">▣ Biển báo</span><span>◼ Vùng quảng cáo</span></div>
<div class="row"><label><input type="checkbox" id="showLabels" checked> Hiện tọa độ nhãn</label></div>
<div id="frameList"></div></section>
<aside><div><small id="progress">0 / 36 đã xác nhận</small></div>
<label for="reviewer">Tên người kiểm tra thực tế</label><input id="reviewer" placeholder="Họ tên người rà ảnh (không điền 'AI')">
<label for="status">Trạng thái vạch</label><select id="status">
<option value="annotated">Có vạch dọc nhìn thấy</option>
<option value="no_visible_lane">Quan sát đủ đường nhưng không thấy vạch</option>
<option value="uncertain">Không đủ bằng chứng</option>
<option value="not_assessable">Camera không hướng ra mặt đường</option></select>
<label for="frameNote">Ghi chú</label><textarea id="frameNote" rows="4"></textarea>
<label for="markType">Loại nét khi chỉnh sửa</label><select id="markType">
<option value="lane">Vạch dọc</option><option value="edge">Mép đường / không phải làn</option></select>
<small>Nhấp 2 điểm trên ảnh để thêm một đoạn thẳng; tiếp tục để thêm nét mới. Nhãn cũ vẫn có thể xóa.</small>
<div class="row"><button id="undo">Lùi nét mới nhất</button><button id="clear">Xóa tất cả nét của loại chọn</button></div>
<div class="row"><button class="primary" id="approve">✓ Xác nhận frame này</button><button class="warnbtn" id="reject">Cần rà lại</button></div>
<div class="row"><button id="export">Xuất JSON có quyết định kiểm tra</button></div>
<small>Chỉ file JSON do người dùng xuất và ký tên mới là phản hồi người kiểm tra; không tự cập nhật bộ GT đang lưu.</small>
</aside></main>
<script id="dataset" type="application/json">__PAYLOAD__</script>
<script>
const model=JSON.parse(document.getElementById('dataset').textContent);
const ids=Object.keys(model.records), data=structuredClone(model.records);
const state={}, canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
let index=0, pending=null, lastAdd=null;const image=new Image();
const $=id=>document.getElementById(id);
function draw(){
  const id=ids[index],r=data[id];canvas.width=r.label_image_size[0];canvas.height=r.label_image_size[1];
  if(image.complete&&image.naturalWidth){ctx.drawImage(image,0,0,canvas.width,canvas.height)}
  if(!$('showLabels').checked)return;
  function lines(values,color,width=4){ctx.lineWidth=width;ctx.strokeStyle=color;
    for(const pts of values||[]){if(pts.length<2)continue;ctx.beginPath();ctx.moveTo(...pts[0]);
      for(const p of pts.slice(1))ctx.lineTo(...p);ctx.stroke();}}
  lines(r.road_edge_markings,'#ff4848');lines(r.lanes,'#18f665');
  ctx.lineWidth=2;ctx.strokeStyle='#ef52ff';
  for(const item of r.signs||[]){const [a,b,c,d]=item.bbox;ctx.strokeRect(a,b,c-a,d-b)}
  ctx.strokeStyle='#21b2ff';
  for(const item of r.non_sign_objects||[]){const [a,b,c,d]=item.bbox;ctx.strokeRect(a,b,c-a,d-b)}
  if(pending){ctx.beginPath();ctx.fillStyle='#ffffff';ctx.arc(pending[0],pending[1],5,0,Math.PI*2);ctx.fill()}
}
function render(){
 const id=ids[index],r=data[id];pending=null;lastAdd=null;
 image.onload=draw;image.src=r.image;
 $('title').textContent=id+' · '+r.source_video_seconds+'s · '+(index+1)+'/'+ids.length;
 $('status').value=r.lane_status;
 $('frameNote').value=r.notes||'';
 $('progress').textContent=Object.values(state).filter(x=>x.decision==='approved').length+' / '+ids.length+' đã xác nhận';
 const list=$('frameList');list.textContent='';
 ids.forEach((id,i)=>{const b=document.createElement('button');b.textContent=(state[id]?.decision==='approved'?'✓ ':'')+id;
   b.onclick=()=>{index=i;render()};list.append(b)});
}
function saveFields(){
 const r=data[ids[index]];r.lane_status=$('status').value;r.notes=$('frameNote').value;
}
$('next').onclick=()=>{saveFields();index=(index+1)%ids.length;render()};
$('prev').onclick=()=>{saveFields();index=(index+ids.length-1)%ids.length;render()};
$('showLabels').onchange=draw;
canvas.addEventListener('click',event=>{
 const bb=canvas.getBoundingClientRect();const pt=[Math.round((event.clientX-bb.x)*canvas.width/bb.width),
   Math.round((event.clientY-bb.y)*canvas.height/bb.height)];
 if(!pending){pending=pt;draw();return}
 const key=$('markType').value==='lane'?'lanes':'road_edge_markings';
 (data[ids[index]][key]||=([])).push([pending,pt]);
 lastAdd=key;pending=null;state[ids[index]]={decision:'edited_unreviewed'};draw()
});
$('undo').onclick=()=>{const key=lastAdd||($('markType').value==='lane'?'lanes':'road_edge_markings');
 data[ids[index]][key]?.pop();pending=null;draw()};
$('clear').onclick=()=>{if(!confirm('Xóa toàn bộ đường của loại đã chọn trên frame này?'))return;
 const key=$('markType').value==='lane'?'lanes':'road_edge_markings';data[ids[index]][key]=[];
 state[ids[index]]={decision:'edited_unreviewed'};draw()};
$('approve').onclick=()=>{saveFields();const reviewer=$('reviewer').value.trim();
 if(!reviewer){alert('Cần tên người thật đang rà ảnh. AI không được tự xác nhận.');return}
 state[ids[index]]={decision:'approved',reviewer,reviewed_at_utc:new Date().toISOString()};
 render()};
$('reject').onclick=()=>{saveFields();state[ids[index]]={decision:'needs_revision'};render()};
$('export').onclick=()=>{saveFields();const reviewer=$('reviewer').value.trim();
 const approved=ids.filter(id=>state[id]?.decision==='approved').length;
 const result={schema_version:'itasmm-human-review-response-v1',
  reviewer_supplied_name:reviewer,frame_count:ids.length,explicitly_approved_frames:approved,
  independent_human_verification_complete:approved===ids.length&&!!reviewer,
  source_provenance:'ChatGPT AI draft; user corrections and decisions',annotations:data,review_decisions:state};
 const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);
 a.download='itasmm_human_review_response.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),5000)
};
$('status').onchange=()=>{saveFields();state[ids[index]]={decision:'edited_unreviewed'}};
$('frameNote').onchange=saveFields;render();
</script></body></html>'''
    output=DIR/'review_ai_visual_36.html'
    output.write_text(html.replace('__PAYLOAD__',payload),encoding='utf8')
    print('REVIEW_UI',output,'size_MB',round(output.stat().st_size/1e6,2),'frames',len(index))
if __name__=='__main__':main()
