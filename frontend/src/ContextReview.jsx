import {useEffect,useState} from 'react';
const API=import.meta.env.VITE_API_BASE||(import.meta.env.DEV?'http://localhost:8000':'');
const choices=[['car','Ô tô con'],['motorcycle','Xe máy'],['truck','Xe tải'],['bus','Xe buýt'],['van','Xe van'],['bicycle','Xe đạp']];
const starting=s=>({
  label:s.label,confidence:s.confidence,recognition_confidence:s.recognition_confidence,
  bbox_xyxy:[...s.bbox_xyxy],panel_role:s.panel_role||'unknown',
  allowed_vehicles:[...(s.allowed_vehicles||[])],overhead:s.overhead??null,
  placement:s.placement||'unknown',lane_scope:s.lane_scope||'unknown',
  label_verified:false,time_windows:s.time_windows||[],time_window_text:''
});
export default function ContextReview({result,onReview}) {
  const [signs,setSigns]=useState([]);
  const [vehicle,setVehicle]=useState('car');
  const [observedAt,setObservedAt]=useState('');
  const [busy,setBusy]=useState(false),[error,setError]=useState('');
  const [reviewed,setReviewed]=useState([]);
  const [speech,setSpeech]=useState(''),[voiceUrl,setVoiceUrl]=useState('');
  useEffect(()=>{
    setSigns(result.perception.signs.map(starting));
    setReviewed([]);setSpeech('');setVoiceUrl('');setError('');
  },[result.annotated_image]);
  if(!signs.length)return null;
  const change=(i,key,value)=>setSigns(s=>s.map((x,j)=>i===j?{...x,[key]:value}:x));
  const addVehiclePlate=(code='S.505a')=>{
    const main=signs.find(s=>s.panel_role==='primary'&&s.label!=='unknown');
    if(!main){setError('Hãy xác định biển chính nhìn thấy trước khi thêm biển phụ.');return}
    const [x1,y1,x2,y2]=main.bbox_xyxy;
    const H=result.frame_shape[0],W=result.frame_shape[1];
    const start=Math.min(H-24,Math.ceil(y2)+5);
    if(start<=0||start+20>=H){setError('Không đủ chỗ cho bbox biển phụ; hãy kiểm tra lại hình ảnh.');return}
    const a=Math.max(0,Math.round(x1+4)),c=Math.min(W-1,Math.round(x2-4));
    setSigns(previous=>[...previous,{
      label:code,confidence:1,recognition_confidence:null,
      bbox_xyxy:[a,start,Math.max(a+2,c),start+20],
      panel_role:'supplementary',allowed_vehicles:[],overhead:false,
      placement:'roadside',lane_scope:'unknown',label_verified:false,time_windows:[],time_window_text:''
    }]);
    setError('Bbox mới chỉ là vị trí gợi ý. Sửa tọa độ theo biển phụ THẬT trong ảnh, chọn loại xe và xác nhận khi chắc chắn.');
  };
  const submit=async e=>{
    e.preventDefault();setBusy(true);setError('');
    try{
      const response=await fetch(API+'/pipeline/interpret',{
        method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({signs:signs.map(({time_window_text,...s})=>({...s,time_windows:time_window_text?time_window_text.split(';').map(pair=>pair.trim().split('-').map(x=>x.trim())):s.time_windows})),observed_at:observedAt||null,ego_boundaries:result.perception.lane.ego_boundaries,
          width:result.frame_shape[1],height:result.frame_shape[0],vehicle})
      });
      if(!response.ok){
        const problem=await response.json().catch(()=>({detail:'Dữ liệu ngữ cảnh chưa hợp lệ.'}));
        throw Error(typeof problem.detail==='string'?problem.detail:'Chưa áp dụng được ngữ cảnh');
      }
      const doc=await response.json();setReviewed(doc.signs);
      setSpeech(doc.speech_text||'');setVoiceUrl('');onReview(doc);
    }catch(err){setError(err.message)}
    finally{setBusy(false)}
  };
  return <form className="panel" onSubmit={submit}>
    <h2>Xác nhận biển chính – biển phụ – loại xe</h2>
    <p>Nguyên tắc: biển chính có phạm vi áp dụng gốc; biển phụ S.505a chỉ giới hạn vào loại xe vẽ trên biển. Chỉ đánh dấu đã kiểm tra khi biển THỰC SỰ hiện trên ảnh. Biển phụ không đọc được phải giữ trạng thái chưa xác định.</p>
    <label>Xe đang mô phỏng: <select value={vehicle} onChange={e=>setVehicle(e.target.value)}>
      {choices.map(([id,name])=><option key={id} value={id}>{name}</option>)}
    </select></label>
    <label>Thời điểm quan sát (có múi giờ, cho biển ngày chẵn/lẻ hoặc giờ)
      <input value={observedAt} onChange={e=>setObservedAt(e.target.value)} placeholder="2026-10-11T08:30:00+07:00"/>
    </label>
    <button type="button" onClick={()=>addVehiclePlate()}>+ Thêm biển phụ S.505a bị AI bỏ sót</button>
    <button type="button" onClick={()=>addVehiclePlate('S.508a')}>+ Thêm biển phụ thời gian S.508a</button>
    {signs.map((s,i)=><fieldset key={i} style={{marginTop:12,marginBottom:12}}>
      <legend>{i+1}. {s.label} — {s.panel_role==='supplementary'?'Biển phụ':'Ứng viên biển chính'}</legend>
      <label>Mã biển <input value={s.label} maxLength={80} onChange={e=>change(i,'label',e.target.value)}/></label>
      <label>Vai trò <select value={s.panel_role} onChange={e=>change(i,'panel_role',e.target.value)}>
        <option value="unknown">Chưa biết</option><option value="primary">Biển chính</option>
        <option value="supplementary">Biển phụ</option>
      </select></label>
      {s.panel_role==='supplementary'&&<button type="button" onClick={()=>
        setSigns(old=>old.filter((_,j)=>j!==i))}>Xóa biển phụ này</button>}
      {s.panel_role==='supplementary'&&s.label==='S.505a'&&
        <label>Hình loại xe thực tế trên biển phụ <select value={s.allowed_vehicles[0]||''}
          onChange={e=>change(i,'allowed_vehicles',e.target.value?[e.target.value]:[])}>
          <option value="">Chưa đọc rõ (không suy đoán)</option>
          {choices.map(([id,name])=><option key={id} value={id}>{name}</option>)}
        </select></label>}
      {s.panel_role==='supplementary'&&['S.508a','S.508b'].includes(s.label)&&
        <label>Khung giờ đọc rõ trên biển (HH:MM-HH:MM, ngăn bằng dấu ;)
          <input value={s.time_window_text} onChange={e=>change(i,'time_window_text',e.target.value)} placeholder="07:00-09:00;16:00-18:00"/>
        </label>}
      <div style={{display:'flex',flexWrap:'wrap',gap:8}}>
        {['x1','y1','x2','y2'].map((coord,j)=><label key={coord}>{coord}
          <input type="number" style={{width:90}} min={0}
            max={j%2?result.frame_shape[0]:result.frame_shape[1]}
            value={s.bbox_xyxy[j]} onChange={e=>{
              const copy=[...s.bbox_xyxy];copy[j]=Number(e.target.value);
              change(i,'bbox_xyxy',copy);
            }}/></label>)}
      </div>
      {s.panel_role!=='supplementary'&&<>
        <label>Vị trí <select value={s.placement} onChange={e=>change(i,'placement',e.target.value)}>
          <option value="unknown">Chưa rõ</option><option value="roadside">Bên đường cùng chiều</option>
          <option value="overhead">Trên làn</option></select></label>
        <label>Phạm vi <select value={s.lane_scope} onChange={e=>change(i,'lane_scope',e.target.value)}>
          <option value="unknown">Chưa rõ</option><option value="all">Phạm vi chung cùng chiều</option>
          <option value="ego">Làn đang đi</option><option value="other">Làn khác</option>
        </select></label>
      </>}
      <label><input type="checkbox" checked={!!s.label_verified}
        onChange={e=>change(i,'label_verified',e.target.checked)}/>
        Tôi đã nhìn ảnh và xác nhận {s.panel_role==='supplementary'?'mã và điều kiện ghi trên biển phụ':'mã biển và vị trí chính xác'}
      </label>
    </fieldset>)}
    <button type="submit" disabled={busy}>{busy?'Đang đánh giá…':'Kiểm tra phạm vi áp dụng theo loại xe'}</button>
    {reviewed.length>0&&<section style={{marginTop:12}}>
      <h3>Kết quả áp dụng cho {choices.find(c=>c[0]===vehicle)?.[1]}</h3>
      {reviewed.filter(s=>s.panel_role==='primary').map((s,i)=>
        <p key={i}><b>{s.label}: {s.name_vi||''}</b> — <strong>
          {s.applicability==='applies'?'ÁP DỤNG':s.applicability==='ignored'?'KHÔNG ÁP DỤNG':'CHƯA XÁC ĐỊNH'}
        </strong> ({s.applicability_reason})</p>)}
    </section>}
    {speech&&<section>
      <p><b>Giọng đọc khi đã xác minh:</b> {speech}</p>
      <button type="button" onClick={async()=>{
        setBusy(true);try{
          const r=await fetch(API+'/pipeline/speech',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:speech})});
          if(!r.ok)throw Error('Edge-TTS không khả dụng');
          setVoiceUrl(API+(await r.json()).audio_url);
        }catch(err){setError(err.message)}finally{setBusy(false)}
      }}>Tạo giọng đọc</button>
      {voiceUrl&&<audio controls autoPlay src={voiceUrl}/>}
    </section>}
    {error&&<p role="alert">{error}</p>}
  </form>;
}
