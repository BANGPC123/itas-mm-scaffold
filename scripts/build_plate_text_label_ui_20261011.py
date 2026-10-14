"""A separate, no-autoverification OCR text labeling interface.

Makes human-confirmed plate transcription possible. Source video dev origin
never changes, and all AI OCR predictions remain unverified suggestions.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
pack=ROOT/'output/review/field_review_pack_20261011'
base=pack/'index_AI_OCR_va_loc_quang_cao.html'
dest=pack/'index_FIELD_OCR_text_labels.html'
if dest.exists():raise FileExistsError('Refuse overwrite existing reviewer draft')
html=base.read_text(encoding='utf-8')
old='<h3>2. Biên làn và tính đầy đủ</h3>'
new='''<h3>2. Xác minh chữ biển phụ</h3>
<p class="note">Bản OCR tự động chỉ là gợi ý. Nhập chính xác toàn bộ dòng chữ thấy được; chỉ tick khi tự kiểm tra được. Không dùng nhãn này làm chỉ dẫn giao thông hay TTS.</p>
<div id="plateTextRows"></div><h3>3. Biên làn và tính đầy đủ</h3>'''
assert old in html
html=html.replace(old,new,1)
needle=" $('lanes').innerHTML=f.lanes.map"
# Inline JS source: render() starts on a line with one leading space.
assert needle in html
inject=r'''
 $('plateTextRows').innerHTML=f.signs.filter(s=>s.panel_role==='supplementary'&&s.judgment==='sign')
  .map(s=>'<div class="sign-row"><b>'+esc(s.id)+'</b> | OCR chưa xác nhận: '+esc((s.ai_ocr_time_candidate||[]).join('; ')||'không rõ')
    +'<label>Văn bản được nhìn và xác nhận <input type="text" data-plate="'+esc(s.id)+'" data-field="plate_text" value="'+esc(s.plate_text||'')+'" placeholder="Ví dụ: 6:00-22:00" size="24"></label>'
    +'<label><input type="checkbox" data-plate="'+esc(s.id)+'" data-field="plate_text_verified" '+(s.plate_text_verified?'checked':'')+'> Tôi đã kiểm tra nguyên văn biển phụ</label></div>').join('')||'<p class="note">Khung hiện tại không có biển phụ đã phân loại.</p>';
'''
html=html.replace(needle,inject+needle,1)
hook=r'''
$('plateTextRows').onchange=ev=>{
 const id=ev.target.dataset.plate,field=ev.target.dataset.field;
 if(!id||!field)return;
 const s=frame().signs.find(x=>x.id===id);
 if(!s)return;
 if(field==='plate_text'){s.plate_text=ev.target.value.trim();s.plate_text_verified=false}
 else if(field==='plate_text_verified'){
   if(ev.target.checked&&!String(s.plate_text||'').trim()){
     say('Nhập nội dung biển phụ trước khi xác nhận chữ');ev.target.checked=false;return;
   }
   s.plate_text_verified=ev.target.checked;
 }
 dirty();render();
};
'''
needle2="$('review').onclick="
assert needle2 in html
html=html.replace(needle2,hook+'\n'+needle2,1)
html=html.replace("if(f.signs.some(s=>s.judgment==='unreviewed'))",
                  "if(f.signs.some(s=>s.plate_text_verified&&!String(s.plate_text||'').trim())){say('Thiếu nội dung biển phụ đã xác nhận');return}if(f.signs.some(s=>s.judgment==='unreviewed'))",1)
dest.write_text(html,encoding='utf-8')
print(dest)
