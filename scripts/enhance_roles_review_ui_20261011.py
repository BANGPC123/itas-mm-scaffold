"""Add view-only AI QA navigation and parent arrows to role-review UI."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'output/review/field_review_pack_20261011/index_AI_bien_chinh_bien_phu.html'
html=path.read_text(encoding='utf-8')
link_script="""frame().signs.filter(s=>s.ai_panel_parent_candidate_id).forEach(s=>{
  const p=frame().signs.find(o=>o.id===s.ai_panel_parent_candidate_id);
  if(!p)return;
  const a=p.bbox_xyxy,b=s.bbox_xyxy;
  ctx.save();ctx.setLineDash([7,5]);ctx.strokeStyle='#dd83ff';ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo((a[0]+a[2])/2,a[3]);ctx.lineTo((b[0]+b[2])/2,b[1]);ctx.stroke();ctx.restore();
});"""
assert 'if(points.length){' in html
html=html.replace('if(points.length){',link_script+'\n if(points.length){',1)
assert '</header>' in html
html=html.replace('</header>','<button type="button" id="next_ai_review">AI QA →</button></header>',1)
js_hook="""$('next_ai_review').onclick=()=>{
 for(let k=1;k<=frames.length;k++){
   const next=(idx+k)%frames.length,f=frames[next];
   if(f.signs.some(s=>s.source==='ai_autobox_20261011'||s.ai_role_needs_revalidation||s.ai_auto_bbox_changed)){
     idx=next;selected=-1;points=[];render();say('Khung có thay đổi AI cần kiểm tra');return;
   }
 }
};"""
assert '\nrender();\n</script>' in html
html=html.replace('\nrender();\n</script>','\n'+js_hook+'\nrender();\n</script>',1)
path.write_text(html,encoding='utf-8')
print(path)
