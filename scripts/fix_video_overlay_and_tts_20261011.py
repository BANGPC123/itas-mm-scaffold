from pathlib import Path
root=Path(r'D:\itas-mm')
p=root/'frontend/src/VideoPanel.jsx'
s=p.read_text(encoding='utf-8')
repls=[
 ("const [speak,setSpeak]=useState(false)", "const [speak,setSpeak]=useState(true)"),
 ("const fontSize=Math.max(18,14*frame_shape[1]/Math.max(1,canvas.getBoundingClientRect().width));",
  "const fontSize=Math.max(17,Math.min(30,14*frame_shape[1]/Math.max(1,canvas.getBoundingClientRect().width)));"),
 ("const text=result.reasoning.speech_text||'';setObservation(text||",
  "const text=result.reasoning.speech_text||result.reasoning.observation_speech_text||'';\n          setObservation(result.reasoning.observation_text||result.reasoning.speech_text||"),
 ("latency,signs:result.perception.signs.length,fresh,",
  "latency,signs:result.perception.signs.length,lanes:result.perception.lane.polylines.length,fresh,"),
 ("Đọc cảnh báo đã được xác minh (Edge-TTS)",
  "Đọc tên biển AI dự đoán (tham khảo, Edge-TTS); cảnh báo chỉ khi đã xác minh"),
 ("speaker.current=createVideoSpeech({","speaker.current=createVideoSpeech({maxAge:8,")]
for old,new in repls:
    if old not in s:raise ValueError('Missing UI pattern: '+old)
    s=s.replace(old,new,1)
start=s.index("const label=sign.label==='unknown'?")
end=s.index(";\n            const labelWidth=",start)
s=s[:start]+"""const label=sign.label==='unknown'
              ? 'Ứng viên biển — chưa xác định'
              : sign.applicability==='applies'
                ? 'Đã xác minh: '+sign.label+' — '+(sign.name_vi||'')
                : 'AI dự đoán: '+sign.label+' — '+(sign.name_vi||'')"""+s[end:]
needle="{running&&stats&&!stats.fresh&&<p>"
assert needle in s
s=s.replace(needle,
 """{running&&stats&&<p role="status">UFLD-v2: {stats.lanes>0?stats.lanes+' đường làn dự đoán':'chưa tìm được đường làn ở khung hình này'}; chưa phải ground truth.</p>}
    """+needle,1)
needle='    <audio ref={audio} controls aria-label="Giọng đọc biển báo"/>'
assert needle in s
s=s.replace(needle,"""    <button type="button" disabled={running}
      onClick={async()=>{setSpeechError('');try{
        const response=await fetch(API+'/pipeline/speech',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({text:'Đây là giọng đọc thử của hệ thống ITAS MM.'})});
        if(!response.ok)throw new Error('Dịch vụ Edge-TTS tạm không khả dụng.');
        audio.current.src=API+(await response.json()).audio_url;
        await audio.current.play();
      }catch(e){setSpeechError(e.message||'Không phát được giọng thử.')}}}>Kiểm tra giọng đọc tiếng Việt</button>
"""+needle,1)
p.write_text(s,encoding='utf-8')
p=root/'frontend/src/videoRuntime.js'
s=p.read_text(encoding='utf-8')
assert "ageSeconds = .3" in s
s=s.replace("ageSeconds = .3","ageSeconds = 1.2",1)
p.write_text(s,encoding='utf-8')
print('VIDEO_UI_OBSERVATION_VOICE_AND_LONGER_OVERLAY_UPDATED')
