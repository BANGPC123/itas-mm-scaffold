"""Offline self-contained human review UI. AI proposals NEVER auto-confirm."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'data/lane_vietnam_review_20261011'
manifest=json.loads((PACK/'manifest.json').read_text(encoding='utf-8'))
drafts={}
f=PACK/'ai_drafts.jsonl'
if f.exists():drafts={r['id']:r for r in [json.loads(x) for x in f.read_text(encoding='utf-8').splitlines() if x.strip()]}
data={'schema':manifest['schema'],'source_sha256':manifest['source_sha256'],'frames':manifest['frames']}
page=r"""<!doctype html><html lang="vi"><meta charset="utf-8"><title>ITAS-MM Lane Review 600</title>
<style>
:root{font-family:system-ui,sans-serif;color-scheme:dark;background:#0c1522;color:#e7edf5}
body{max-width:1600px;margin:auto;padding:12px}button,select,input,textarea{font:inherit;padding:8px;margin:3px;background:#243247;color:white;border:1px solid #54708c;border-radius:5px}
button{cursor:pointer}nav,section{border:1px solid #354458;border-radius:9px;background:#172334;padding:10px;margin:8px 0}
main{display:grid;grid-template-columns: minmax(470px,2fr) minmax(330px,1fr);gap:12px}
canvas{width:100%;max-height:72vh;background:#050a10;cursor:crosshair}
small{color:#b6c8d8}.warn{color:#ffcf86}input[type=checkbox]{width:20px;height:20px}label{display:inline-block;margin:4px}
@media(max-width:850px){main{display:block}}
</style>
<h2>ITAS-MM — Chỉnh nhãn làn từ 600 ảnh test_video.mp4</h2>
<p class="warn">ĐƯỜNG AI VÀ MASK AI KHÔNG PHẢI GROUND TRUTH. Chỉ xác nhận frame sau khi đã tự quan sát và sửa hết đường sai.</p>
<nav><button id="prev">← Trước</button><input id="index" type="number" min="1" max="600" style="width:80px"><button id="goto">Đến</button>
<button id="next">Tiếp →</button> <span id="counter"></span> <b id="progress"></b>
<button id="export">Xuất JSON kết quả</button> <label>Nhập JSON <input type="file" id="import" accept=".json"></label></nav>
<main><section><canvas id="canvas" width="960" height="540"></canvas>
<p><label><input type="checkbox" id="showAi" checked>Hiện polyline gợi ý AI</label>
<label><input type="checkbox" id="showMask">Hiện mask AI</label></p><small id="info"></small></section>
<section><h3>Vẽ đường bằng chuột</h3>
<p>Nhấn chuột chọn nhiều điểm trên vạch rồi bấm <b>Lưu đường</b>. Các nét màu xanh là nhãn người duyệt; AI hiện nét đứt vàng / xanh lam.</p>
<select id="kind"><option value="lane_paint">Vạch sơn phân làn</option><option value="road_edge">Mép đường / dải phân cách (không phải lane paint)</option><option value="occluded">Vùng bị che — bỏ qua khi train</option></select>
<p><button id="finish">Lưu đường</button><button id="undoPt">Lùi điểm</button><button id="clearPts">Hủy nét</button><button id="deleteLast">Xóa đường vừa lưu</button></p>
<div id="labels"></div>
<label><input type="checkbox" id="noLane">Không có vạch sơn phân làn nhìn thấy</label>
<p><label><input type="checkbox" id="complete">Đã kiểm tra HẾT các vạch sơn và ranh giới trong ảnh</label></p>
<p>Người duyệt <input id="reviewer" placeholder="Họ tên người duyệt"></p>
<p>Ghi chú<br><textarea id="note" rows="2" style="width:95%"></textarea></p>
<button id="approve">Tôi xác nhận đã tự kiểm tra ảnh</button><button id="revoke">Bỏ xác nhận</button>
<p id="status" class="warn"></p>
</section></main>
<p><small>Nhấn Xuất JSON định kỳ để sao lưu; ảnh được chia theo block thời gian, TEST_HOLDOUT_SAME_VIDEO không phải independent TEST.</small></p>
<script>
const M=__MANIFEST__,AI=__PROPOSALS__, $=id=>document.getElementById(id);
const STORE='itasmm_lane_review_'+M.source_sha256.slice(0,16);
let frames=JSON.parse(JSON.stringify(M.frames)),at=0,pts=[],photo=new Image(),mask=null;
function store(){let f=frames[at];f.reviewer=$('reviewer').value.trim();f.review_note=$('note').value;f.human_no_visible_lane=$('noLane').checked;f.human_exhaustive=$('complete').checked;
try{localStorage.setItem(STORE,JSON.stringify(frames))}catch(e){$('status').textContent='Không lưu tự động được; hãy xuất JSON.'}}
function stroke(ctx,line,color,width=3,dashes=[]){if(!line||line.length<2)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dashes);ctx.beginPath();line.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke();ctx.setLineDash([])}
function render(){if(!photo.complete||!photo.naturalWidth)return;
let f=frames[at],c=$('canvas'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(photo,0,0,c.width,c.height);
if($('showMask').checked&&mask){ctx.globalAlpha=.20;ctx.drawImage(mask,0,0,c.width,c.height);ctx.globalAlpha=1}
if($('showAi').checked)(AI[f.id]?.ai_proposals||[]).forEach(x=>stroke(ctx,x.points,x.source.startsWith('UFLD')?'#4cdaff':'#ffcf68',2,[7,6]));
f.human_lanes.forEach(l=>stroke(ctx,l.points,l.kind==='lane_paint'?'#2cfc76':l.kind==='road_edge'?'#ff6868':'#fa5deb',4));
stroke(ctx,pts,'white',3);pts.forEach(p=>{ctx.fillStyle='white';ctx.fillRect(p[0]-3,p[1]-3,6,6)});
$('labels').textContent='Nhãn thật: '+f.human_lanes.map((l,i)=>(i+1)+'. '+l.kind+' ('+l.points.length+' điểm)').join(' | ');
}
function progress(){$('progress').textContent=' | Người đã xác nhận: '+frames.filter(f=>f.human_confirmed).length+'/600'}
function load(){let f=frames[at];pts=[];mask=null;$('counter').textContent=(at+1)+'/600 '+f.split+' ('+f.id+')';$('index').value=at+1;
$('reviewer').value=f.reviewer||'';$('note').value=f.review_note||'';
$('noLane').checked=!!f.human_no_visible_lane;$('complete').checked=!!f.human_exhaustive;
$('info').textContent='t='+f.source_video_seconds+'s | '+f.temporal_group+' | SHA='+f.sha256.slice(0,16);
$('status').textContent=f.human_confirmed?'ĐÃ xác nhận bởi người duyệt':'Chưa xác nhận';
photo=new Image();photo.onload=()=>{let c=$('canvas');c.width=f.width;c.height=f.height;render()};photo.src=f.image;
const source=AI[f.id];if(source?.mask){let m=new Image();m.onload=()=>{mask=m;render()};m.src=source.mask}
progress()}
function goto(n){store();at=Math.max(0,Math.min(frames.length-1,n));load()}
$('prev').onclick=()=>goto(at-1);$('next').onclick=()=>goto(at+1);$('goto').onclick=()=>goto(Number($('index').value)-1);
$('canvas').onpointerdown=e=>{if($('noLane').checked&&$('kind').value==='lane_paint')return;const r=e.currentTarget.getBoundingClientRect();pts.push([Math.round((e.clientX-r.left)*frames[at].width/r.width),Math.round((e.clientY-r.top)*frames[at].height/r.height)]);render()};
$('finish').onclick=()=>{if(pts.length<2){$('status').textContent='Cần ít nhất 2 điểm';return}
frames[at].human_lanes.push({kind:$('kind').value,points:pts,source:'human_drawn'});pts=[];frames[at].human_confirmed=false;store();render()};
$('undoPt').onclick=()=>{pts.pop();render()};$('clearPts').onclick=()=>{pts=[];render()};
$('deleteLast').onclick=()=>{frames[at].human_lanes.pop();frames[at].human_confirmed=false;store();render()};
$('noLane').onchange=()=>{if($('noLane').checked){frames[at].human_lanes=frames[at].human_lanes.filter(l=>l.kind!=='lane_paint');pts=[];$('complete').checked=true}frames[at].human_confirmed=false;store();render()};
$('complete').onchange=()=>{frames[at].human_confirmed=false;store()};
$('note').oninput=store;$('reviewer').oninput=store;$('showAi').onchange=render;$('showMask').onchange=render;
$('approve').onclick=()=>{store();let f=frames[at];if(!f.reviewer){$('status').textContent='Nhập tên người duyệt';return}
if(!f.human_exhaustive){$('status').textContent='Hãy xác nhận đã đánh dấu toàn bộ vạch';return}
if(f.human_no_visible_lane&&f.human_lanes.some(l=>l.kind==='lane_paint')){$('status').textContent='Không thể vừa không có vạch sơn vừa có nhãn lane_paint';return}
if(!f.human_no_visible_lane&&!f.human_lanes.some(l=>l.kind==='lane_paint')){$('status').textContent='Chưa có vạch sơn nào: hãy chọn Không có vạch hoặc vẽ vạch sơn';return}
f.human_confirmed=true;f.reviewed_at=new Date().toISOString();store();load()};
$('revoke').onclick=()=>{frames[at].human_confirmed=false;frames[at].reviewed_at=null;store();load()};
$('export').onclick=()=>{store();let payload={schema:M.schema,source_sha256:M.source_sha256,
exported_at:new Date().toISOString(),frames};const b=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}),url=URL.createObjectURL(b),a=document.createElement('a');a.href=url;a.download='itasmm_lane_review_export.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1500)};
$('import').onchange=async e=>{try{let d=JSON.parse(await e.target.files[0].text());if(d.source_sha256!==M.source_sha256||d.frames?.length!==frames.length)throw Error('Sai nguồn');
d.frames.forEach((f,i)=>{if(f.sha256!==M.frames[i].sha256||f.id!==M.frames[i].id||f.split!==M.frames[i].split)throw Error('Sai SHA frame '+i)});
frames=d.frames;at=0;load();store()}catch(e){$('status').textContent='Không nhập được: '+e.message}};
try{let d=JSON.parse(localStorage.getItem(STORE));if(Array.isArray(d)&&d.length===600&&d[0].sha256===frames[0].sha256)frames=d}catch(e){}
load();
</script></html>"""
page=page.replace('__MANIFEST__',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
page=page.replace('__PROPOSALS__',json.dumps(drafts,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
(PACK/'review_lane_600.html').write_text(page,encoding='utf-8')
print('LANE_REVIEW_UI_WRITTEN',len(page.encode('utf-8')),'bytes','AI_DRAFT_FRAMES',len(drafts),flush=True)
