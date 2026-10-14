import {useEffect,useState} from 'react';
const API=import.meta.env.VITE_API_BASE||(import.meta.env.DEV?'http://localhost:8000':'');
export default function EvaluationPanel(){
  const [data,setData]=useState(null);
  const [current,setCurrent]=useState(null);
  useEffect(()=>{let cancelled=false;fetch(`${API}/pipeline/retraining`).then(r=>r.ok?r.json():null).then(value=>{if(!cancelled)setCurrent(value);}).catch(()=>{});return()=>{cancelled=true;};},[]);
  useEffect(()=>{let cancelled=false;fetch(`${API}/pipeline/evaluation`).then(r=>r.ok?r.json():null).then(value=>{if(!cancelled)setData(value);}).catch(()=>{});return()=>{cancelled=true;};},[]);
  if(!data?.available&&!current?.available)return null;
  const processing=current?.product_video?.processing_ms||current?.latency?.processing_ms;
  return <section className="panel" style={{marginTop:24}}>
    {current?.available&&<>
      <h2>Phiên bản huấn luyện lại 09/10/2026</h2>
      <p>Model được chọn trên VAL. Video của bạn dùng để kiểm thử phát triển, không đưa vào train. Nhãn video do AI rà soát, chưa có holdout độc lập.</p>
      <dl>
        <dt>YOLO11n — mAP50–95 trên TEST</dt><dd>{current.detector?`${(current.detector.metrics['metrics/mAP50-95(B)']*100).toFixed(2)}%`:'Đang đánh giá'}</dd>
        <dt>MobileNetV3-Small — accuracy trên crop TEST</dt><dd>{current.recognizer?`${(current.recognizer.metrics.all.accuracy*100).toFixed(2)}%`:'Đang đánh giá'}</dd>
        <dt>UFLD-v2</dt><dd>{current.deployment.lane_candidate_selected?'Dùng bản fine-tune đã vượt baseline trên VAL':'Đã fine-tune; giữ checkpoint gốc vì bản mới chưa vượt tiêu chí VAL'}</dd>
        <dt>Xử lý mỗi lần suy luận — trung bình / P95 (không gồm TTS và xuất video)</dt><dd>{processing?`${processing.mean.toFixed(1)} / ${processing.p95.toFixed(1)} ms`:'Đang đo'}</dd>
      </dl>
      {current.video_url&&<a href={`${API}${current.video_url}`} target="_blank" rel="noreferrer">{current.product_video?`Xem video ${current.product_video.output_fps} FPS, tên biển tiếng Việt và Edge TTS`:'Xem bản kiểm thử cũ (3 FPS, không âm thanh)'}</a>}
    </>}
    {data?.available&&<details><summary>Kết quả baseline trước khi huấn luyện lại</summary>
    <h2>Kết quả TEST của phiên bản baseline</h2>
    <p>Đây là số đo của checkpoint gốc trước cập nhật 09/10/2026. Classifier và xử lý video hiện tại đã thay đổi, đang chờ nhãn người duyệt và holdout mới. Các chỉ số dưới đây không phải accuracy của phiên bản hiện tại hoặc của hướng dẫn lái xe.</p>
    <dl>
      <dt>Detector mAP50–95</dt><dd>{data.detector_map50_95.toFixed(4)}</dd>
      <dt>Nhận dạng trên crop chuẩn: accuracy / macro-F1</dt><dd>{(data.recognizer_accuracy*100).toFixed(2)}% / {data.recognizer_macro_f1.toFixed(4)}</dd>
      <dt>Phục hồi đúng mã biển từ ảnh đầy đủ: recall</dt><dd>{(data.correct_code_recall*100).toFixed(2)}%</dd>
    </dl>
    <p>Nhãn ngữ nghĩa chưa được người kiểm chứng độc lập. Chưa đo độ chính xác làn đường và điều kiện áp dụng trên video thực địa.</p>
    {data.report_url&&<a href={`${API}${data.report_url}`} target="_blank" rel="noreferrer">Mở báo cáo Model &amp; Results (PDF)</a>}
    </details>}
  </section>;
}
