"""Create an OFFLINE human-review pack. Never label AI suggestions as human GT."""
from pathlib import Path
import hashlib,json,shutil
root=Path(__file__).resolve().parents[1]
src=root/'output/review/test_video_ai_review'
dst=root/'output/review/field_review_pack_20261011'
(dst/'frames').mkdir(exist_ok=True,parents=True)
ref=json.loads((src/'ai_reviewed.json').read_text(encoding='utf-8'))
frames=[]
changes=json.loads((src/'changes.json').read_text(encoding='utf-8'))
removed_by_frame={}
for item in changes:
    if item.get('action')=='removed':
        removed_by_frame.setdefault(item['frame'],[]).append(item)
for row in ref['samples']:
    idx=int(row['id'].split('_')[-1])
    orig=src/'images'/f'{idx:03d}_0.jpg'
    if not orig.is_file():raise RuntimeError('Missing '+str(orig))
    final=dst/'frames'/orig.name
    shutil.copy2(orig,final)
    candidates=[]
    for s in row['signs']:
        candidates.append({'id':s['id'],'bbox_xyxy':[round(float(x),2) for x in s['bbox_xyxy']],
            'label':s['label'],'panel_role':s.get('panel_role','unknown'),
            'ai_suggested_label':s['label'],'judgment':'unreviewed',
            'source':'ai_candidate'})
    for j,item in enumerate(removed_by_frame.get(row['id'],[])):
        box=[round(float(x),2) for x in item['before']['bbox_xyxy']]
        if not (0<=box[0]<box[2]<=row['width'] and 0<=box[1]<box[3]<=row['height']):
            continue
        if any(sum(abs(a-b) for a,b in zip(s['bbox_xyxy'],box))<8 for s in candidates):
            continue
        candidates.append({'id':f'ai_removed_{j}','bbox_xyxy':box,'label':item['before']['label'],
            'panel_role':'unknown','ai_suggested_label':item['before']['label'],
            'judgment':'unreviewed','source':'ai_removed_candidate',
            'ai_removal_reason':item.get('reason','')})
    frames.append({'id':row['id'],'source_group':'test_video_development',
        'timestamp_seconds':row['timestamp_seconds'],'image':'frames/'+orig.name,
        'image_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
        'width':row['width'],'height':row['height'],'reviewer':'','reviewed_at':None,
        'human_confirmed':False,'all_visible_signs_annotated':False,'lane_status':'unreviewed',
        'signs':candidates,'lanes':[],'notes':''})
manifest={'schema_version':'itasmm-field-review-v1','provenance':'AI-proposed labels; human_confirmed=false until independently reviewed','frames':frames}
(dst/'review_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
html=r'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>ITAS-MM · Human Ground Truth Review</title><meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{box-sizing:border-box}body{margin:0;background:#101720;color:#e9f0f7;font:15px system-ui,Segoe UI,sans-serif}header{padding:12px 20px;background:#1b2737;display:flex;align-items:center;gap:12px;flex-wrap:wrap}main{display:grid;grid-template-columns:minmax(350px,1fr) 390px;gap:12px;padding:12px}section{padding:12px;background:#1b2737;border-radius:9px}canvas{width:100%;height:auto;border:1px solid #7187a6;cursor:crosshair;touch-action:none}button,select,input,textarea{color:#f5f9ff;background:#293b50;border:1px solid #7b90a7;border-radius:5px;padding:7px;font:inherit}button{cursor:pointer}button:disabled{opacity:.4}label{display:inline-block;margin:4px}.note{color:#c5d6e8;font-size:13px}.warn{color:#ffd396}.controls{display:flex;flex-wrap:wrap;gap:6px;align-items:center}aside{max-height:86vh;overflow-y:auto}fieldset{border:1px solid #59677a;border-radius:7px;margin:10px 0;padding:10px}textarea{width:100%}.sign-row{padding:8px;border-bottom:1px solid #38485c}strong{color:#86cbff}@media(max-width:1050px){main{grid-template-columns:1fr}aside{max-height:none}}.selected{background:#334b65}#message{padding:5px 10px;color:#ffd79c}
</style></head><body><header><strong>ITAS-MM — GÁN NHÃN NGƯỜI DUYỆT</strong><span class="warn">Nguồn VIDEO PHÁT TRIỂN — không phải independent TEST</span><label>Người duyệt <input id="reviewer" placeholder="Họ tên / mã người duyệt"></label><button id="prev">◀ Trước</button><span id="counter">0/0</span><button id="next">Sau ▶</button><button id="export">Xuất file JSON</button><label>Nhập lại <input id="import" type="file" accept=".json" style="max-width:190px"></label></header>
<main><section><div class="controls"><strong id="scene">Ảnh</strong><label>Chế độ <select id="mode"><option value="view">Xem/chọn box</option><option value="box">Vẽ box biển còn thiếu (kéo chuột)</option><option value="lane">Vẽ polyline làn (click từng điểm)</option></select></label><label>Làn <select id="laneRole"><option value="ego_left">Biên trái làn xe</option><option value="ego_right">Biên phải làn xe</option><option value="other">Làn khác</option></select></label><button id="finish">Kết thúc đường làn</button><button id="undo">Hoàn tác điểm</button></div><p class="note">Box đỏ = ứng viên AI chưa kiểm tra. Box xanh = biển thật do người duyệt xác nhận. Box cam = vật thể không phải biển. Click để chọn. Dùng vạch xanh để vẽ polyline; chỉ đánh dấu làn có thể nhìn thấy, không suy đoán phần khuất.</p><canvas id="canvas" width="1280" height="720"></canvas><div id="message" role="status"></div></section>
<aside><section><h3>1. Kiểm tra biển báo</h3><div id="signs"></div><h3>2. Biên làn và tính đầy đủ</h3><label>Tình trạng <select id="laneStatus"><option value="unreviewed">Chưa kiểm tra</option><option value="annotated">Đã vẽ làn</option><option value="no_visible_lane">Không thấy vạch làn</option><option value="uncertain">Không đủ rõ</option></select></label><div id="lanes"></div><label><input type="checkbox" id="exhaustive"> Tôi đã đánh dấu TẤT CẢ biển thật có thể nhìn rõ trong ảnh (đủ điều kiện đo recall).</label><label>Ghi chú/điều kiện <textarea id="notes" rows="3"></textarea></label><p><button id="review">Tôi xác nhận đã tự kiểm tra ảnh này</button> <span id="reviewStatus"></span></p><p class="note">Không tự động đánh dấu các nhãn AI là đúng. Đối với box không phải biển, chọn <strong>not_sign</strong>. Nếu chưa rõ, chọn <strong>uncertain</strong>. Để đánh giá detector recall cần tự bổ sung cả biển YOLO bỏ sót, sau đó mới tích ô đầy đủ.</p></section></aside></main>
<script>
const manifest=__MANIFEST__;
const original=JSON.parse(JSON.stringify(manifest));
let frames=JSON.parse(JSON.stringify(manifest.frames)),idx=0,selected=-1,begin=null,points=[];
const c=document.getElementById('canvas'),ctx=c.getContext('2d'),img=new Image(),$=id=>document.getElementById(id);
const esc=x=>String(x??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
function frame(){return frames[idx]}
function xy(ev){const r=c.getBoundingClientRect();return [Math.max(0,Math.min(c.width,Math.round((ev.clientX-r.left)*c.width/r.width))),Math.max(0,Math.min(c.height,Math.round((ev.clientY-r.top)*c.height/r.height)))]}
function say(t){$('message').textContent=t}
function redraw(){
 if(img.complete&&img.naturalWidth)ctx.drawImage(img,0,0,c.width,c.height);else{ctx.fillStyle='#15202b';ctx.fillRect(0,0,c.width,c.height)}
 frame().signs.forEach((s,i)=>{const [x1,y1,x2,y2]=s.bbox_xyxy;ctx.strokeStyle=s.judgment==='sign'?'#54eb92':s.judgment==='not_sign'?'#ffb14b':s.judgment==='uncertain'?'#f6de7c':'#ff5364';ctx.lineWidth=i===selected?4:2;ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.fillStyle=ctx.strokeStyle;ctx.font='18px system-ui';ctx.fillText(s.label||'unknown',x1,Math.max(19,y1-4))});
 frame().lanes.forEach(l=>{ctx.strokeStyle=l.role==='ego_left'?'#46dffe':l.role==='ego_right'?'#6efc8c':'#eeda75';ctx.lineWidth=4;ctx.beginPath();l.points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke()});
 if(points.length){ctx.strokeStyle='#ffd283';ctx.lineWidth=3;ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke()}
}
function option(value,items){return items.map(v=>'<option value="'+v+'" '+(v===value?'selected':'')+'>'+v+'</option>').join('')}
function render(){
 const f=frame();$('counter').textContent=(idx+1)+'/'+frames.length;$('scene').textContent=f.id+' · '+f.timestamp_seconds+'s';$('reviewer').value=f.reviewer||$('reviewer').value;$('laneStatus').value=f.lane_status;$('notes').value=f.notes||'';$('exhaustive').checked=f.all_visible_signs_annotated===true;
 $('reviewStatus').textContent=f.human_confirmed?'✓ Có xác nhận từ người duyệt':'Chưa được xác nhận';
 $('signs').innerHTML=f.signs.map((s,i)=>'<div class="sign-row '+(i===selected?'selected':'')+'"><button data-choose="'+i+'">#'+(i+1)+'</button> <label>Đánh giá <select data-index="'+i+'" data-field="judgment">'+option(s.judgment,['unreviewed','sign','not_sign','uncertain'])+'</select></label><label>Mã biển (nếu đọc được) <input data-index="'+i+'" data-field="label" value="'+esc(s.label)+'" size="10"></label><label>Vai trò <select data-index="'+i+'" data-field="panel_role">'+option(s.panel_role,['unknown','primary','supplementary'])+'</select></label><button data-remove="'+i+'">Bỏ box</button><small class="note"> Gợi ý AI: '+esc(s.ai_suggested_label||'mới')+'</small></div>').join('')||'<p>Không có box ứng viên — hãy kiểm tra xem có biển bị bỏ sót không.</p>';
 $('lanes').innerHTML=f.lanes.map((l,i)=>'<div>'+esc(l.role)+' ('+l.points.length+' điểm) <button data-dellane="'+i+'">Xóa đường</button></div>').join('')||'<p class="note">Chưa vẽ đường làn</p>';
 img.onload=()=>{c.width=img.naturalWidth;c.height=img.naturalHeight;redraw()};
 img.src=f.image;redraw();
}
function dirty(){frame().human_confirmed=false;frame().reviewed_at=null;$('reviewStatus').textContent='Chưa được xác nhận';redraw()}
$('prev').onclick=()=>{idx=Math.max(0,idx-1);selected=-1;points=[];render()};
$('next').onclick=()=>{idx=Math.min(frames.length-1,idx+1);selected=-1;points=[];render()};
$('signs').onchange=ev=>{const i=Number(ev.target.dataset.index),field=ev.target.dataset.field;if(!field)return;frame().signs[i][field]=ev.target.value;dirty()};
$('signs').onclick=ev=>{if(ev.target.dataset.choose!==undefined){selected=Number(ev.target.dataset.choose);render()}if(ev.target.dataset.remove!==undefined){frame().signs.splice(Number(ev.target.dataset.remove),1);selected=-1;dirty();render()}};
$('lanes').onclick=ev=>{if(ev.target.dataset.dellane!==undefined){frame().lanes.splice(Number(ev.target.dataset.dellane),1);dirty();render()}};
$('laneStatus').onchange=ev=>{frame().lane_status=ev.target.value;if(ev.target.value==='no_visible_lane')frame().lanes=[];dirty();render()};
$('notes').onchange=ev=>{frame().notes=ev.target.value;dirty()};
$('exhaustive').onchange=ev=>{frame().all_visible_signs_annotated=ev.target.checked;dirty()};
c.onpointerdown=ev=>{if($('mode').value==='box'){begin=xy(ev);c.setPointerCapture(ev.pointerId)}};
c.onpointerup=ev=>{if($('mode').value!=='box'||!begin)return;const [x,y]=xy(ev),[a,b]=begin;begin=null;const box=[Math.min(a,x),Math.min(b,y),Math.max(a,x),Math.max(b,y)];if(box[2]-box[0]<8||box[3]-box[1]<8)return;say('Box mới: cần chọn judgment và mã biển nếu xác nhận');frame().signs.push({id:'manual_'+Date.now()+'_'+frame().signs.length,bbox_xyxy:box,label:'unknown',ai_suggested_label:'',panel_role:'unknown',judgment:'unreviewed',source:'human_drawn'});dirty();render()};
c.onclick=ev=>{const mode=$('mode').value,[x,y]=xy(ev);if(mode==='lane'){points.push([x,y]);redraw()}else if(mode==='view'){selected=frame().signs.findIndex(s=>{const [a,b,d,e]=s.bbox_xyxy;return a<=x&&x<=d&&b<=y&&y<=e});render()}};
$('finish').onclick=()=>{if(points.length<2){say('Cần ít nhất 2 điểm cho polyline');return}frame().lanes.push({role:$('laneRole').value,points:points.slice()});frame().lane_status='annotated';points=[];dirty();render()};
$('undo').onclick=()=>{points.pop();redraw()};
$('review').onclick=()=>{const f=frame();if(!$('reviewer').value.trim()){say('Nhập tên người duyệt trước');return}if(f.signs.some(s=>s.judgment==='unreviewed')){say('Vẫn còn box chưa được phân loại');return}if(f.lane_status==='unreviewed'){say('Chọn tình trạng vạch làn');return}if(f.all_visible_signs_annotated&&f.signs.some(s=>s.judgment==='uncertain')){say('Chỉ đánh dấu đủ biển nếu không còn box uncertain');return}f.reviewer=$('reviewer').value.trim();f.human_confirmed=true;f.reviewed_at=new Date().toISOString();say('Đã ghi xác nhận vào bản nháp; nhớ Xuất file JSON');render()};
$('export').onclick=()=>{const data={schema_version:'itasmm-field-review-v1',frames};const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='itasmm_human_review_'+new Date().toISOString().slice(0,10)+'.json';a.click();URL.revokeObjectURL(url)};
$('import').onchange=async ev=>{const file=ev.target.files[0];if(!file)return;try{const parsed=JSON.parse(await file.text());if(parsed.schema_version!=='itasmm-field-review-v1'||parsed.frames.length!==original.frames.length)throw Error('Khác phiên bản/độ dài');for(let i=0;i<parsed.frames.length;i++){if(parsed.frames[i].id!==original.frames[i].id||parsed.frames[i].image_sha256!==original.frames[i].image_sha256)throw Error('Không trùng nguồn ảnh')}frames=parsed.frames;idx=0;selected=-1;points=[];render();say('Đã nhập bản nháp, kiểm tra lại trước xuất')}catch(e){say('Không thể nhập JSON: '+e.message)}};
render();
</script></body></html>'''
html=html.replace('__MANIFEST__',json.dumps(manifest,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
(dst/'index.html').write_text(html,encoding='utf-8')
(dst/'INSTRUCTIONS.txt').write_text('Mở index.html bằng trình duyệt (file://). Xem từng ảnh, ghi nhận tất cả ứng viên biển và làn, chọn xác nhận thủ công, xuất JSON rồi chạy script validate_human_review_20261011.py. Tệp này thuộc video phát triển, KHÔNG phải holdout độc lập.\n',encoding='utf-8')
print(json.dumps({'frames':len(frames),'images_mb':round(sum((dst/f['image']).stat().st_size for f in frames)/1048576,2),'index':str(dst/'index.html')},ensure_ascii=False))
