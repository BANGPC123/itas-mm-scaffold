import {useEffect,useRef,useState} from 'react';
import {analyzeLatest, freshResult} from './videoRuntime';
import {createVideoSpeech} from './videoSpeech';
const API=import.meta.env.VITE_API_BASE||(import.meta.env.DEV?'http://localhost:8000':'');

export default function VideoPanel({onRun}) {
  const video=useRef(null), overlay=useRef(null), active=useRef(false), stream=useRef(null);
  const url=useRef(null), generation=useRef(0), request=useRef(null), latest=useRef(null);
  const audio=useRef(null), speaker=useRef(null), maskStart=useRef(null);
  const [speak,setSpeak]=useState(true), [speechError,setSpeechError]=useState('');
  const [observation,setObservation]=useState('');
  const [running,setRunning]=useState(false), [error,setError]=useState('');
  const [roadBottom,setRoadBottom]=useState(.65), [stats,setStats]=useState(null);
  const [laneVisualRoi,setLaneVisualRoi]=useState(true);
  const [laneDisplayMode,setLaneDisplayMode]=useState('precision');
  const [reviewAuxiliary,setReviewAuxiliary]=useState(false);
  const [recordingStart,setRecordingStart]=useState('');
  const [masking,setMasking]=useState(false), [exclusions,setExclusions]=useState([]);
  const [lat,setLat]=useState('10.7769'), [lon,setLon]=useState('106.7009');
  const getMaskPoint=e=>{const r=e.currentTarget.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];};
  const beginMask=e=>{if(!masking||running)return;maskStart.current=getMaskPoint(e);e.currentTarget.setPointerCapture(e.pointerId);};
  const endMask=e=>{if(!masking||running||!maskStart.current)return;const [sx,sy]=maskStart.current,[ex,ey]=getMaskPoint(e);maskStart.current=null;const rect=[Math.min(sx,ex),Math.min(sy,ey),Math.max(sx,ex),Math.max(sy,ey)];if(rect[2]-rect[0]>.01&&rect[3]-rect[1]>.01)setExclusions(old=>old.length>=8?old:[...old,rect]);};
  const invalidate=()=>{generation.current++;active.current=false;request.current?.abort();latest.current=null;speaker.current?.reset();audio.current?.pause();};
  const stop=()=>{invalidate();setRunning(false);};
  const release=()=>{
    invalidate();
    stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;
    if(url.current)URL.revokeObjectURL(url.current);url.current=null;
  };
  useEffect(()=>()=>release(),[]);
  useEffect(()=>{
    let id, lastPaintKey;
    const paint=()=>{
      const v=video.current, canvas=overlay.current;
      if(v && canvas && v.videoWidth){
        const sample=latest.current;
        const visible=sample && !v.seeking && freshResult(sample.timestamp,v.currentTime);
        const paintKey=`${v.videoWidth}:${v.videoHeight}:${visible?sample.timestamp:'empty'}`;
        if(paintKey===lastPaintKey){id=requestAnimationFrame(paint);return;}
        lastPaintKey=paintKey;
        if(canvas.width!==v.videoWidth || canvas.height!==v.videoHeight){canvas.width=v.videoWidth;canvas.height=v.videoHeight;}
        const ctx=canvas.getContext('2d');
        ctx.clearRect(0,0,canvas.width,canvas.height);
        ctx.strokeStyle='#ffffff';ctx.lineWidth=2;ctx.setLineDash([10,8]);
        ctx.beginPath();ctx.moveTo(0,canvas.height*roadBottom);ctx.lineTo(canvas.width,canvas.height*roadBottom);ctx.stroke();ctx.setLineDash([]);
        if(visible){
          const {perception,frame_shape}=sample.result;
          const roiVisualOnly=perception.lane.backend==='yolopv2_visual_candidate' || perception.lane.backend==='ufld_v2_roi_visual_only' || perception.lane.backend==='camera_road_paint_visual' || (laneDisplayMode==='candidate'&&perception.lane.backend==='ufld_v2');
          ctx.save();ctx.scale(canvas.width/frame_shape[1],canvas.height/frame_shape[0]);
          ctx.save();
          if(laneVisualRoi && perception.lane.backend!=='camera_road_paint_visual') {
            ctx.beginPath();ctx.rect(0,0,frame_shape[1],frame_shape[0]*.72);ctx.clip();
          }
          ctx.strokeStyle=roiVisualOnly?'#ffd166':'#85d784';
          ctx.lineWidth=roiVisualOnly?3:2;
          ctx.setLineDash(roiVisualOnly?[10,7]:[]);
          for(const points of (perception.lane.display_polylines??perception.lane.polylines)){
            if(points.length<2)continue;
            ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();
          }
          ctx.restore();
          const fontSize=Math.max(17,Math.min(30,14*frame_shape[1]/Math.max(1,canvas.getBoundingClientRect().width)));
          ctx.font=`${fontSize}px sans-serif`;
          for(const sign of perception.signs){
            const [x1,y1,x2,y2]=sign.bbox_xyxy;
            ctx.strokeStyle=sign.applicability==='ignored'?'#aaa':'#ffbe00';
            ctx.strokeRect(x1,y1,x2-x1,y2-y1);
            const label=sign.label==='unknown'
              ? 'Ứng viên biển — chưa xác định'
              : sign.applicability==='applies'
                ? 'Đã xác minh: '+sign.label+' — '+(sign.name_vi||'')
                : 'AI dự đoán: '+sign.label+' — '+(sign.name_vi||'');
            const labelWidth=Math.min(frame_shape[1]-8,ctx.measureText(label).width);
            const labelX=Math.max(4,Math.min(x1,frame_shape[1]-labelWidth-4));
            const labelY=Math.max(fontSize+4,y1-6);
            ctx.fillStyle='#141414';ctx.fillRect(labelX-3,labelY-fontSize-2,labelWidth+6,fontSize+8);
            ctx.fillStyle=ctx.strokeStyle;ctx.fillText(label,labelX,labelY,frame_shape[1]-8);
            if(sign.label!=='unknown' && sign.car_context?.text){
              const carLabel=sign.car_context.text;
              const carWidth=Math.min(frame_shape[1]-8,ctx.measureText(carLabel).width);
              const carY=labelY+fontSize+7;
              ctx.fillStyle='#141414';ctx.fillRect(labelX-3,carY-fontSize-2,carWidth+6,fontSize+7);
              ctx.fillStyle='#e6f1ff';ctx.fillText(carLabel,labelX,carY,frame_shape[1]-8);
            }
          }
          for(const candidate of perception.auxiliary_sign_candidates||[]){
            const [x1,y1,x2,y2]=candidate.bbox_xyxy;
            const agreed=candidate.agreement_status==='cross_model_visual_agreement';
            ctx.strokeStyle=agreed?'#70deeb':'#d88bff';ctx.lineWidth=2;ctx.setLineDash([5,5]);
            ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.setLineDash([]);
            ctx.font='16px sans-serif';ctx.fillStyle='#f2cfff';
            ctx.fillText((agreed?'Hai AI cùng mã (chưa xác minh luật): ':'YOLO11 (cần rà): ')+candidate.label_vi,x1,Math.max(18,y1-5),Math.max(20,frame_shape[1]-x1-6));
          }
          ctx.restore();
        }
        for(const [x1,y1,x2,y2] of exclusions){
          ctx.fillStyle='rgba(245,70,50,.23)';ctx.strokeStyle='#f54832';ctx.lineWidth=3;
          const a=x1*canvas.width,b=y1*canvas.height,w=(x2-x1)*canvas.width,h=(y2-y1)*canvas.height;
          ctx.fillRect(a,b,w,h);ctx.strokeRect(a,b,w,h);
        }
      }else if(canvas && lastPaintKey!==undefined){
        canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
        lastPaintKey=undefined;
      }
      id=requestAnimationFrame(paint);
    };
    id=requestAnimationFrame(paint);
    return()=>cancelAnimationFrame(id);
  },[roadBottom,exclusions,laneDisplayMode]);
  const camera=async()=>{
    release();setExclusions([]);setMasking(false);setRunning(false);setError('');setStats(null);
    try{
      stream.current=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment',width:{ideal:1280},height:{ideal:720}},audio:false});
      video.current.srcObject=stream.current;await video.current.play();
    }catch{setError('Không mở được camera. Hãy kiểm tra quyền truy cập hoặc chọn video.');}
  };
  const choose=e=>{
    const file=e.target.files?.[0];if(!file)return;
    release();setRecordingStart('');setExclusions([]);setMasking(false);setRunning(false);setError('');setStats(null);video.current.srcObject=null;
    url.current=URL.createObjectURL(file);video.current.src=url.current;
  };
  const start=async()=>{
    const latitude=Number(lat),longitude=Number(lon);
    if(!lat.trim()||!lon.trim()||!Number.isFinite(latitude)||!Number.isFinite(longitude)||Math.abs(latitude)>90||Math.abs(longitude)>180){setError('Tọa độ GPS chưa hợp lệ.');return;}
    if(!video.current?.videoWidth){setError('Chọn video hoặc mở camera trước.');return;}
    const own=++generation.current, v=video.current, canvas=document.createElement('canvas');
    const streamId=crypto.randomUUID();
    const scale=Math.min(1,1920/v.videoWidth);
    canvas.width=Math.round(v.videoWidth*scale);canvas.height=Math.round(v.videoHeight*scale);
    const context=canvas.getContext('2d');
    request.current=new AbortController();
    const signal=request.current.signal;
    setSpeechError('');setObservation('');
    speaker.current=createVideoSpeech({maxAge:8,
      synthesize:async text=>{
        const response=await fetch(`${API}/pipeline/speech`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text}),signal});
        if(!response.ok)throw new Error('TTS tạm không khả dụng; nội dung vẫn hiển thị.');
        return `${API}${(await response.json()).audio_url}`;
      },
      play:url=>new Promise((resolve,reject)=>{
        const a=audio.current;a.src=url;a.onended=resolve;a.onpause=resolve;
        a.onerror=()=>reject(new Error('Không phát được giọng đọc.'));
        a.play().catch(()=>reject(new Error('Trình duyệt chặn tự phát âm thanh. Nhấn phát ở thanh giọng đọc.')));
      }),
      onError:error=>setSpeechError(error.message)
    });
    setError('');setStats(null);active.current=true;setRunning(true);
    let lastUpdate=performance.now(), count=0;
    const initialQuality=v.getVideoPlaybackQuality?.();
    let previousFrames=initialQuality?.totalVideoFrames??0;
    try{
      await v.play();
      await analyzeLatest({video:v, active:()=>active.current&&own===generation.current,
        capture:()=>{
          context.drawImage(v,0,0,canvas.width,canvas.height);
          return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(new Error('Không đọc được khung hình video.')),'image/jpeg',.85));
        },
        analyze:(blob,frameTimestamp)=>onRun({file:new File([blob],'frame.jpg',{type:'image/jpeg'}),latitude,longitude,speak:false,stream:true,roadBottom,signal,streamId,frameTimestamp,excludeRects:exclusions,laneVisualRoi,laneDisplayMode,auxiliarySignReview:reviewAuxiliary,
          observedAt:stream.current?new Date().toISOString():recordingStart?new Date(new Date(recordingStart).getTime()+frameTimestamp*1000).toISOString():undefined}),
        publish:(result,timestamp)=>{
          latest.current={result,timestamp};
          const text=result.reasoning.speech_text||result.reasoning.observation_speech_text||'';
          setObservation([result.reasoning.car_summary,result.reasoning.observation_text||result.reasoning.speech_text].filter(Boolean).join(' ')||'Chưa đủ bằng chứng để phát cảnh báo lái xe.');
          if(speak)void speaker.current.update(text,timestamp,()=>v.currentTime);
        },
        observe:(result,latency,fresh)=>{
          count++;
          const now=performance.now();
          if(now-lastUpdate>=1000){
            const quality=v.getVideoPlaybackQuality?.();
            setStats({fps:count*1000/(now-lastUpdate),playbackFps:quality?(quality.totalVideoFrames-previousFrames)*1000/(now-lastUpdate):null,latency,signs:result.perception.signs.length,lanes:result.perception.lane.display_polylines?.length??result.perception.lane.polylines.length,fresh,
              dropped:(quality?.droppedVideoFrames??0)-(initialQuality?.droppedVideoFrames??0)});
            previousFrames=quality?.totalVideoFrames??0;
            count=0;lastUpdate=now;
          }
        }});
    }catch(e){if(own===generation.current&&e.name!=='AbortError')setError(e.message||'Không xử lý được video.');}
    finally{if(own===generation.current)stop();}
  };
  return <section className="panel" style={{marginTop:24}}>
    <h2>Video MP4 — demo trên máy tính, không cần camera thật</h2>
    <p>Video phát ở tốc độ gốc. Nhận diện cập nhật tối đa 10 lần/giây; kết quả quá 300 ms sẽ ẩn. Ngữ cảnh ô tô, GPS mô phỏng. Đường làn và mã biển là dự đoán, KHÔNG dùng để điều khiển xe. Cảnh báo âm thanh chỉ phát nếu mã biển, loại xe và phạm vi làn đã được xác minh.</p>
    <label htmlFor="video-file">Video trên máy</label> <input id="video-file" type="file" accept="video/*" onChange={choose} disabled={running}/>

    <label><input type="checkbox" checked={speak} disabled={running} onChange={e=>setSpeak(e.target.checked)}/> Đọc tên biển AI dự đoán bằng giọng Việt tại máy; cảnh báo chỉ khi đã xác minh</label>
    <label>Ngày giờ bắt đầu ghi hình (theo múi giờ thiết bị), nếu biết <input type="datetime-local" value={recordingStart} disabled={running} onChange={e=>setRecordingStart(e.target.value)}/></label>
    <p>Để trống nếu không biết thời điểm ghi hình. Biển theo giờ/ngày sẽ giữ điều kiện đó; thời gian phát lại không được dùng thay thế.</p>
    <div style={{position:'relative',marginTop:12}}>
      <video ref={video} controls muted playsInline onSeeking={stop} onPause={stop} onEnded={stop} style={{display:'block',width:'100%'}}/>
      <canvas ref={overlay} aria-hidden={!masking} aria-label={masking?'Kéo chuột để khoanh vùng camera cần loại trừ':undefined} onPointerDown={beginMask} onPointerUp={endMask} style={{position:'absolute',inset:0,width:'100%',height:'100%',pointerEvents:masking?'auto':'none',touchAction:'none',cursor:masking?'crosshair':'default'}}/>
    </div>
    <div style={{display:'flex',gap:8,flexWrap:'wrap',marginTop:10}}>
      <button type="button" disabled={running} onClick={()=>{video.current?.pause();setMasking(v=>!v)}}>{masking?'Xong khoanh vùng':'Khoanh vùng loại trừ cabin'}</button>
      <button type="button" disabled={running||exclusions.length===0} onClick={()=>setExclusions([])}>Xóa vùng loại trừ ({exclusions.length})</button>
    </div>
    <p>Để tránh nhận nhầm đồng hồ, bảng điều khiển và đồ vật cố định: tạm dừng video, chọn “Khoanh vùng”, kéo chuột vẽ tối đa 8 hình chữ nhật màu đỏ rồi chọn “Xong”. Biển thật trong vùng khoanh cũng bị bỏ qua. Vùng loại trừ chỉ áp dụng cho video/camera đang chọn.</p>
    <label><input type="checkbox" checked={laneVisualRoi} disabled={running} onChange={e=>setLaneVisualRoi(e.target.checked)}/>
      Nhận diện vạch đường theo vùng mặt đường, lọc tap-lô và xe che khuất
    </label>
    <p>Hệ thống tìm vùng mặt đường theo khung hình, kết hợp mô hình làn và kiểm tra vạch sơn nhìn thấy. Vạch màu vàng là dự đoán; phần bị xe che khuất không dùng để xác định phạm vi biển báo.</p>
    <label>Hiển thị ứng viên UFLD-v2 (chỉ trực quan): <select value={laneDisplayMode} disabled={running} onChange={e=>setLaneDisplayMode(e.target.value)}><option value="precision">Ưu tiên ít nét sai</option><option value="candidate">Giữ vạch chưa chắc chắn (để rà)</option></select></label>
    <p>Chế độ giữ vạch có thể hiện thêm nét sai. Khi mô hình vạch sơn theo mặt đường đủ điều kiện hoạt động, lớp phủ đó vẫn được ưu tiên; không có chế độ nào tự xác nhận làn pháp lý.</p>
    <label><input type="checkbox" checked={reviewAuxiliary} disabled={running} onChange={e=>setReviewAuxiliary(e.target.checked)}/> Bật YOLO11 optimized (59 lớp) làm lớp gợi ý màu tím, chỉ dùng rà soát — không xác minh luật</label>
    <label htmlFor="road-bottom">Ranh dưới vùng phát hiện biển: {Math.round(roadBottom*100)}% chiều cao ảnh</label>
    <input id="road-bottom" type="range" min="30" max="100" value={Math.round(roadBottom*100)} disabled={running} onChange={e=>setRoadBottom(Number(e.target.value)/100)}/>
    <p>Đặt ranh cắt biển phía trên tap-lô để giảm nhận nhầm đồng hồ trong xe. Model làn xử lý toàn bộ ảnh, không dùng ranh này. 65% là thiết lập tham khảo cho video cũ, cần hiệu chỉnh theo camera.</p>
    <div style={{display:'flex',flexWrap:'wrap',gap:12,marginTop:12}}>
      <label>Vĩ độ <input type="number" min="-90" max="90" value={lat} onChange={e=>setLat(e.target.value)} disabled={running}/></label>
      <label>Kinh độ <input type="number" min="-180" max="180" value={lon} onChange={e=>setLon(e.target.value)} disabled={running}/></label>
      <button type="button" onClick={start} disabled={running||masking}>Phân tích video</button>
      <button type="button" onClick={stop} disabled={!running}>Dừng phân tích</button>
      <button type="button" onClick={()=>{release();setRunning(false);setStats(null);video.current.pause();video.current.srcObject=null;video.current.removeAttribute('src');video.current.load();}}>Đóng nguồn video</button>
    </div>
    <p role="status">{running?(stats?`${stats.playbackFps===null?'':`Video ${stats.playbackFps.toFixed(1)} FPS · `}${stats.fps.toFixed(1)} lượt nhận diện/giây · phản hồi ${stats.latency.toFixed(0)} ms · ${stats.signs} biển · ${stats.dropped} khung hình phát bị bỏ`:'Đang khởi động nhận diện; video vẫn tiếp tục phát.'):'Phân tích đã dừng.'}</p>
    {running&&stats&&<p role="status">Vạch đường: {stats.lanes>0?stats.lanes+' đoạn vạch sơn quan sát được':'chưa đủ bằng chứng ở khung hình này'}.</p>}
    {running&&stats&&!stats.fresh&&<p>Kết quả đến chậm nên đã ẩn lớp phủ. Video vẫn phát; thử đóng tiến trình suy luận khác nếu có.</p>}
    {error&&<p role="alert">{error}</p>}
    <p aria-live="polite">{observation}</p>
    <button type="button" disabled={running}
      onClick={async()=>{setSpeechError('');try{
        const response=await fetch(API+'/pipeline/speech',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({text:'Đây là giọng đọc thử của hệ thống ITAS MM.'})});
        if(!response.ok)throw new Error('Bộ đọc tiếng Việt tại máy tạm không khả dụng.');
        audio.current.src=API+(await response.json()).audio_url;
        await audio.current.play();
      }catch(e){setSpeechError(e.message||'Không phát được giọng thử.')}}}>Kiểm tra giọng đọc tiếng Việt</button>
    <audio ref={audio} controls aria-label="Giọng đọc biển báo"/>
    {speechError&&<p role="status">{speechError}</p>}
  </section>;
}
