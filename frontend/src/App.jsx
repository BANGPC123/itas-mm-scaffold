import { useState, useRef, useEffect, useCallback } from "react";
import VideoPanel from "./VideoPanel";
import ContextReview from "./ContextReview";
import EvaluationPanel from "./EvaluationPanel";
import LegalResearchPanel from "./LegalResearchPanel";
import DemoVideos from "./DemoVideos";
import AutoResearchPanel from "./AutoResearchPanel";

const API_BASE = import.meta.env.VITE_API_BASE || (import.meta.env.DEV ? "http://localhost:8000" : "");

function useBackendStatus() {
  const [status, setStatus] = useState("checking");

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/health`)
      .then((res) => {
        if (!cancelled) setStatus(res.ok ? "connected" : "error");
      })
      .catch(() => {
        if (!cancelled) setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return status;
}

function StatusPill({ status }) {
  const text = {
    checking: "checking backend…",
    connected: "backend connected",
    error: "backend unreachable",
  }[status];

  return <span className={`status-pill ${status}`}>{text}</span>;
}

function UploadPanel({ onRun, isRunning }) {
  const [file, setFile] = useState(null);
  const [latitude, setLatitude] = useState("10.7769");
  const [longitude, setLongitude] = useState("106.7009");
  const [speak, setSpeak] = useState(false);
  const inputRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    if (selected) setFile(selected);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file) return;
    onRun({ file, latitude: parseFloat(latitude), longitude: parseFloat(longitude), speak });
  };

  return (
    <form className="panel" onSubmit={handleSubmit}>
      <h2>Input khung hình</h2>

      <div className="field">
        <label htmlFor="frame-upload">Ảnh từ camera phía trước</label>
        <div
          className={`dropzone ${file ? "has-file" : ""}`}
          role="button"
          tabIndex={0}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        >
          {file ? file.name : "Chọn ảnh để chạy pipeline"}
        </div>
        <input
          id="frame-upload"
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: "none" }}
        />
      </div>

      <div className="field">
        <label htmlFor="lat">Vĩ độ (latitude) — GPS mô phỏng</label>
        <input
          id="lat"
          type="number"
          step="0.0001"
          min="-90"
          max="90"
          value={latitude}
          onChange={(e) => setLatitude(e.target.value)}
          required
        />
      </div>

      <div className="field">
        <label htmlFor="lon">Kinh độ (longitude) — GPS mô phỏng</label>
        <input
          id="lon"
          type="number"
          step="0.0001"
          min="-180"
          max="180"
          value={longitude}
          onChange={(e) => setLongitude(e.target.value)}
          required
        />
      </div>

      <label><input type="checkbox" checked={speak} onChange={e => setSpeak(e.target.checked)} /> Đọc kết quả bằng tiếng Việt</label>
      <p>Ngữ cảnh xe hiện tại: ô tô.</p>
      <button className="run-button" type="submit" disabled={!file || isRunning}>
        {isRunning ? "Đang chạy pipeline…" : "Chạy pipeline"}
      </button>
    </form>
  );
}

function ResultsPanel({ result, error, isRunning, onReview }) {
  if (error) {
    return (
      <div className="panel">
        <h2>Kết quả</h2>
        <div className="error-box">{error}</div>
      </div>
    );
  }

  if (isRunning && !result) {
    return (
      <div className="panel">
        <h2>Kết quả</h2>
        <div className="empty-state">Đang xử lý qua 4 giai đoạn…</div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="panel">
        <h2>Kết quả</h2>
        <div className="empty-state">Chọn ảnh và chạy pipeline để xem kết quả tại đây.</div>
      </div>
    );
  }

  const { perception, context, reasoning, audio_url, annotated_image, warnings = [], timings_ms = {} } = result;

  return (
    <div className="results-stack">
      {annotated_image && <img src={annotated_image} alt="Khung hình với vị trí biển báo và các đường làn được phát hiện" style={{width: "100%", height: "auto"}} />}
      <ContextReview result={result} onReview={onReview}/>
      {warnings.length > 0 && <div className="panel" role="status"><h2>Trạng thái kết quả</h2><ul>{warnings.map(w => <li key={w}>{w}</li>)}</ul></div>}
      <div className="panel">
        <h2>Perception</h2>
        {perception.signs.length === 0 ? (
          <p className="empty-state" style={{ padding: "8px 0" }}>
            Chưa phát hiện biển báo đủ tin cậy trong khung hình này.
          </p>
        ) : (
          <ul className="sign-list">
            {perception.signs.map((sign, i) => (
              <li key={i}>
                <span>
                  {sign.is_compound
                    ? `Biển hợp thành: ${sign.panel_labels.join(", ")}`
                    : sign.name_vi?`${sign.label} — ${sign.name_vi}`:sign.label}
                  <small style={{display: "block"}}>{({applies:"Áp dụng",ignored:"Bỏ qua",uncertain:"Chưa xác định điều kiện áp dụng"})[sign.applicability]}</small>
                  {sign.car_context?.text && <small style={{display:"block",marginTop:4}}>{sign.car_context.text}</small>}
                  {!!sign.car_context?.conditions?.length && <small style={{display:"block"}}>
                    {sign.car_context.conditions.filter(c=>c.type==='time').map(c=>c.windows.map(w=>w.join('–')).join('; ')).join(' · ')}
                  </small>}
                </span>
                <span className="conf">Phát hiện {(sign.confidence * 100).toFixed(0)}%{sign.recognition_confidence != null && <small style={{display:"block"}}>Nhận dạng {(sign.recognition_confidence * 100).toFixed(0)}%</small>}</span>
              </li>
            ))}
          </ul>
        )}
        {!!perception.supplementary_candidates?.length && <div>
          <h3>Biển phụ đọc tự động</h3>
          <p>Kết quả OCR do máy đọc, chưa xác định hiệu lực pháp lý.</p>
          {perception.supplementary_candidates.map((p,i)=><p key={i}>
            {p.primary_label}: {p.time_windows?.length ? p.time_windows.map(w=>w.join("–")).join("; ") : p.raw_texts.filter(Boolean).join(" / ") || "Chưa đọc rõ"}
            {p.vehicle_candidates?.length ? " · " + (p.condition_polarity==='exclusion'?'ngoại lệ: ':'') + p.vehicle_candidates.join(", ") : ""}
          </p>)}
        </div>}
        <div className="result-row">
          <span className="label">Số đoạn vạch đường hiển thị</span>
          <span className="value">{perception.lane_line_count}</span>
        </div>
        <div className="result-row">
          <span className="label">Độ lệch tâm làn (px)</span>
          <span className="value">
            {perception.lane_center_offset_px !== null
              ? perception.lane_center_offset_px.toFixed(1)
              : "—"}
          </span>
        </div>
      </div>

      <div className="panel">
        <h2>Context</h2>
        <div className="result-row">
          <span className="label">Khu vực</span>
          <span className="zone-badge">{context.zone_label}</span>
        </div>
        <div className="result-row">
          <span className="label">Tọa độ</span>
          <span className="value">
            {context.latitude.toFixed(4)}, {context.longitude.toFixed(4)}
          </span>
        </div>
      </div>

      <div className="panel">
        <h2>Reasoning + Interaction</h2>
        <p className="guidance-text">{reasoning.guidance_text}</p>
        {reasoning.references?.length > 0 && <details><summary>Tra cứu mã biển trong QCVN 41:2024</summary><p>Văn bản tham khảo theo phiên bản đã lưu; điều kiện áp dụng và hiệu lực hiện tại cần kiểm chứng.</p>{reasoning.references.map(ref => <blockquote key={ref.code}><p>{ref.excerpt}</p><a href={`${ref.source_url}#page=${ref.page}`} target="_blank" rel="noreferrer">{ref.code}: {ref.edition}, trang PDF {ref.page}</a></blockquote>)}</details>}
        <div className="result-row">
          <span className="label">Số đoạn quy định được truy hồi</span>
          <span className="value">{reasoning.retrieved_chunk_count}</span>
        </div>
        {audio_url && (
          <div style={{ marginTop: "12px" }}>
            <audio controls src={`${API_BASE}${audio_url}`} aria-label="Giọng đọc kết quả nhận dạng" />
          </div>
        )}
        <p>Thời gian xử lý: {timings_ms.total?.toFixed(0)} ms</p>
      </div>
    </div>
  );
}

export default function App() {
  const backendStatus = useBackendStatus();
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const requestSeq = useRef(0);

  const handleRun = useCallback(async ({ file, latitude, longitude, speak = false, stream = false, roadBottom = 1, signal, streamId, frameTimestamp, excludeRects = [], laneVisualRoi = false, observedAt }) => {
    if (stream) {
      const form = new FormData();
      form.append('image', file);
      form.append('latitude', latitude);
      form.append('longitude', longitude);
      form.append('synthesize_audio', 'false');
      form.append('annotate', 'false');
      form.append('road_bottom', roadBottom);
      form.append('lane_visual_roi', String(laneVisualRoi));
      form.append('stream_id', streamId);
      if(observedAt)form.append('observed_at',observedAt);
      form.append('frame_timestamp', frameTimestamp);
      form.append('exclusion_zones', JSON.stringify(excludeRects));
      const timeout = AbortSignal.timeout(20000);
      const response = await fetch(`${API_BASE}/pipeline/run`, {method:'POST', body:form, signal:signal?AbortSignal.any([signal,timeout]):timeout});
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        const error = new Error(typeof body.detail === 'string' ? body.detail : `Lỗi xử lý (${response.status}).`);
        error.status = response.status;
        throw error;
      }
      return response.json();
    }
    const sequence = ++requestSeq.current;
    setIsRunning(true);
    setError(null);
    if (!stream) setResult(null);

    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("latitude", latitude);
      formData.append("longitude", longitude);
      formData.append("synthesize_audio", String(speak));

      const response = await fetch(`${API_BASE}/pipeline/run`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(typeof body.detail === "string" ? body.detail : `Yêu cầu chưa hợp lệ (${response.status}).`);
      }

      const data = await response.json();
      if (sequence === requestSeq.current) setResult(data);
    } catch (err) {
      if (sequence === requestSeq.current) setError(err.message || "Không thể kết nối tới backend.");
      if (stream) throw err;
    } finally {
      if (sequence === requestSeq.current) setIsRunning(false);
    }
  }, []);

  return (
    <div className="app">
      <div className="header">
        <div>
          <h1>ITAS-MM — Video, GPS mô phỏng &amp; RAG pháp luật</h1>
          <p>
            Tải một khung hình từ camera phía trước và tọa độ GPS mô phỏng để
            chạy thử toàn bộ pipeline: Perception → Context → Reasoning →
            Interaction.
          </p>
        </div>
        <StatusPill status={backendStatus} />
      </div>

      <div className="grid">
        <UploadPanel onRun={handleRun} isRunning={isRunning} />
        <ResultsPanel result={result} error={error} isRunning={isRunning} onReview={reviewed => setResult(old => ({...old,perception:{...old.perception,signs:reviewed.signs},audio_url:null,reasoning:{...old.reasoning,guidance_text:reviewed.guidance_text,references:reviewed.references,retrieved_chunks:reviewed.retrieved_chunks,speech_text:reviewed.speech_text,retrieved_chunk_count:reviewed.retrieved_chunks.length}}))} />
      </div>
      <DemoVideos apiBase={API_BASE}/>
      <VideoPanel onRun={handleRun} />
      <AutoResearchPanel apiBase={API_BASE}/>
      <LegalResearchPanel apiBase={API_BASE}/>
      <EvaluationPanel />
    </div>
  );
}
