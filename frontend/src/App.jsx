import { useState, useRef, useEffect, useCallback } from "react";

const API_BASE = "http://localhost:8000";

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
  const inputRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    if (selected) setFile(selected);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!file) return;
    onRun({ file, latitude: parseFloat(latitude), longitude: parseFloat(longitude) });
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
          value={longitude}
          onChange={(e) => setLongitude(e.target.value)}
          required
        />
      </div>

      <button className="run-button" type="submit" disabled={!file || isRunning}>
        {isRunning ? "Đang chạy pipeline…" : "Chạy pipeline"}
      </button>
    </form>
  );
}

function ResultsPanel({ result, error, isRunning }) {
  if (error) {
    return (
      <div className="panel">
        <h2>Kết quả</h2>
        <div className="error-box">{error}</div>
      </div>
    );
  }

  if (isRunning) {
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

  const { perception, context, reasoning, audio_path } = result;

  return (
    <div className="results-stack">
      <div className="panel">
        <h2>Perception</h2>
        {perception.signs.length === 0 ? (
          <p className="empty-state" style={{ padding: "8px 0" }}>
            Không phát hiện biển báo nào (kiểm tra weights_path trong
            configs/perception.yaml nếu model chưa được train).
          </p>
        ) : (
          <ul className="sign-list">
            {perception.signs.map((sign, i) => (
              <li key={i}>
                <span>
                  {sign.is_compound
                    ? `Biển hợp thành: ${sign.panel_labels.join(", ")}`
                    : sign.label}
                </span>
                <span className="conf">{(sign.confidence * 100).toFixed(0)}%</span>
              </li>
            ))}
          </ul>
        )}
        <div className="result-row">
          <span className="label">Số đoạn làn phát hiện</span>
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
        <div className="result-row">
          <span className="label">Số đoạn quy định được truy hồi</span>
          <span className="value">{reasoning.retrieved_chunk_count}</span>
        </div>
        {audio_path && (
          <div style={{ marginTop: "12px" }}>
            <p style={{ fontSize: "12px", color: "var(--text-secondary)", margin: "0 0 6px" }}>
              File giọng nói được lưu tại: <span style={{ fontFamily: "var(--font-mono)" }}>{audio_path}</span>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const backendStatus = useBackendStatus();
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  const handleRun = useCallback(async ({ file, latitude, longitude }) => {
    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("latitude", latitude);
      formData.append("longitude", longitude);

      const response = await fetch(`${API_BASE}/pipeline/run`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Backend trả về lỗi ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Không thể kết nối tới backend.");
    } finally {
      setIsRunning(false);
    }
  }, []);

  return (
    <div className="app">
      <div className="header">
        <div>
          <h1>ITAS-MM demo console</h1>
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
        <ResultsPanel result={result} error={error} isRunning={isRunning} />
      </div>
    </div>
  );
}
