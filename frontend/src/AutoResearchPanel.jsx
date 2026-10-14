import {useEffect,useState} from "react";
export default function AutoResearchPanel({apiBase=""}) {
  const [data,setData]=useState(null);
  useEffect(()=>{let live=true;fetch(apiBase+"/pipeline/automatic-research-status").then(r=>r.ok?r.json():null).then(d=>{if(live)setData(d)}).catch(()=>{});return()=>{live=false}},[apiBase]);
  if(!data?.labels)return null;
  return <section className="panel" style={{marginTop:20}}>
    <h2>Kết quả nghiên cứu tự động</h2>
    <p>Đã xử lý {data.labels.frames} ảnh; giữ {data.labels.frames_with_consensus} ảnh có đoạn làn đồng thuận và dùng {data.labels.training_eligible} ảnh để fine-tune {data.trained_models.length} mô hình thử nghiệm.</p>
    <p>Nhãn do máy tạo. Chưa có F1 thực địa Việt Nam hoặc tỷ lệ cảnh báo sai được xác minh.</p>
    <details><summary>So sánh trên CULane</summary>
      <p>{data.comparison_dataset}. Điểm dưới đây là F1 pixel vạch sơn.</p>
      <table><thead><tr><th>Mô hình</th><th>F1 pixel</th></tr></thead><tbody>{data.comparison.map(x=><tr key={x.model}><td>{x.model}</td><td>{(x.pixel_f1*100).toFixed(2)}%</td></tr>)}</tbody></table>
      <p>Bản fine-tune chưa được chọn làm mô hình đang chạy.</p>
    </details>
    {data.system&&<p>Kiểm tra vận hành: {data.system.frames} frame, {data.system.failures} lỗi pipeline; p95 {data.system.latency.p95_ms.toFixed(0)} ms. Truy hồi: Recall@4 {(data.recall_at_4*100).toFixed(1)}% trên 24 câu hồi quy.</p>}
  </section>;
}
