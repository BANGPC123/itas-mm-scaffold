from pathlib import Path
import json
p=Path(r'D:\itas-mm\output\review\field_review_pack_20261011\index_AI_tu_bat_box.html')
src=p.read_text(encoding='utf-8')
assert '<script>' in src and '</script>' in src
js=src.split('<script>',1)[1].split('</script>',1)[0]
text=js.split('const manifest=',1)[1].split(';\nconst original=',1)[0]
d=json.loads(text)
assert len(d['frames'])==24
assert sum(len(f['signs']) for f in d['frames'])==87
assert sum((p.parent/f['image']).is_file() for f in d['frames'])==24
assert sum(s.get('source')=='ai_autobox_20261011' for f in d['frames'] for s in f['signs'])==4
test=p.with_suffix('.check.js')
test.write_text(js,encoding='utf-8')
print('24 FRAMES, 87 BOXES, 4 NEW, ALL SOURCE IMAGES PRESENT')
