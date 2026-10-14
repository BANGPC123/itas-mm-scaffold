import {useEffect,useState} from "react";

export default function DemoVideos({apiBase=""}) {
  const [status,setStatus]=useState(null);
  const [completion,setCompletion]=useState(null);
  useEffect(()=>{
    let active=true;
    fetch(apiBase+"/pipeline/capstone-status")
      .then(r=>r.ok?r.json():Promise.reject(Error("API unavailable")))
      .then(data=>{if(active)setStatus(data)})
      .catch(()=>{if(active)setStatus({offline_demo_ready:false})});
    fetch(apiBase+"/pipeline/video-completion-status")
      .then(r=>r.ok?r.json():Promise.reject(Error("API unavailable")))
      .then(data=>{if(active)setCompletion(data)})
      .catch(()=>{if(active)setCompletion({complete:false})});
    return()=>{active=false};
  },[apiBase]);
  const testMetrics=completion?.sign_test?.results?.[completion?.selected_sign_method||"960"];
  return <section className="panel" style={{marginTop:24}}>
    <h2>Video kết quả và kiểm thử</h2>
    {completion?.complete && <div style={{marginBottom:24}}>
      <h3>test_video.mp4 — vạch đường, biển báo và ngữ cảnh ô tô</h3>
      <video controls playsInline preload="metadata" style={{width:"100%",maxWidth:1100,borderRadius:8}}
        src={apiBase+completion.video_url}/>
      <p>{completion.summary.output_frames.toLocaleString("vi-VN")} frame, {completion.summary.output_fps} FPS;
        AI phân tích {completion.summary.analysis_hz} lần/giây. Có tên biển, điều kiện cho ô tô và giọng đọc theo cảnh.</p>
      <p>Đã giải mã kiểm tra video xuất, đối chiếu dự đoán trước/sau và kiểm tra âm thanh.
        Vạch hiển thị là dự đoán; biển phụ chưa đủ rõ được giữ trạng thái chưa xác định.</p>
      {testMetrics && <p>
        TEST độc lập gồm {completion.sign_test.images.toLocaleString("vi-VN")} ảnh:
        precision mã biển {(100*testMetrics.known_code_precision).toFixed(2)}%;
        recall {(100*testMetrics.code_recall).toFixed(2)}%.
        Đây là phép đo trên ảnh có nhãn, chưa phải độ chính xác thực địa của video này.
      </p>}
      <a href={apiBase+completion.report_url}>Đọc báo cáo đánh giá và số liệu TEST</a>
    </div>}
    {completion?.independent_video?.complete && <div style={{marginBottom:24}}>
      <h3>Hành trình ô tô khác — Hòa Bình</h3>
      <video controls playsInline preload="metadata" style={{width:"100%",maxWidth:960,borderRadius:8}}
        src={apiBase+completion.independent_video.video_url}/>
      <p>Đã chạy toàn bộ {completion.independent_video.summary.source_frames.toLocaleString("vi-VN")} frame.
        Nguồn hành trình khác dùng để kiểm tra vận hành và sửa lỗi hiển thị làn, gồm cảnh quay ngang và phản chiếu kính.</p>
      <p>11 ảnh kiểm tra âm tính do AI rà soát; chưa có nhãn chuẩn của người để đo F1 làn.</p>
      <p style={{fontSize:13}}>Nguồn: Obakeneko, <a href="https://commons.wikimedia.org/wiki/File:Driving_in_Kim_Boi_and_Luong_Son_Districts.ogv">Wikimedia Commons</a>,
        <a href="https://creativecommons.org/licenses/by/3.0"> CC BY 3.0</a>. Đã thêm lớp phủ ITAS-MM; video phụ không có giọng đọc.</p>
    </div>}
    <p>Các đoạn trình diễn bổ sung có sẵn; bạn cũng có thể tải MP4 để chạy trực tiếp.</p>
    {status?.offline_demo_ready?
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(300px,1fr))",gap:14}}>
        <div>
          <h3>1. UFLD-v2 — Highway (video liên tục)</h3>
          <video controls playsInline preload="metadata" style={{width:"100%",borderRadius:8}}
            src={apiBase+"/pipeline/capstone-video/lane"}/>
          <p>12 giây, 1280×720, làn đường + GPS mô phỏng. Cảnh này không có biển rõ.</p>
        </div>
        <div>
          <h3>2. YOLO11n — Tên biển tiếng Việt và TTS theo cảnh</h3>
          <video controls playsInline preload="metadata" style={{width:"100%",borderRadius:8}}
            src={apiBase+"/pipeline/capstone-video/signs"}/>
          <p>24 ảnh phát triển ghép thành video; tên biển xuất hiện trực tiếp trên hình và giọng đọc mô tả theo cảnh có biển. Không phải ground truth.</p>
        </div>
      </div>:
      <p>Video dựng sẵn chưa khả dụng; vẫn có thể tải MP4 của bạn để chạy trực tiếp.</p>}
    <p style={{fontSize:13}}>Chỉ dành cho trình diễn nghiên cứu. Không sử dụng các dự đoán để đưa ra quyết định lái xe thực tế.</p>
  </section>;
}
