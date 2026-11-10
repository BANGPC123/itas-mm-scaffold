import { useRef, useState } from "react";

// Drop zone plus file summary and one primary action. The parent decides what the file is used for.
export default function UploadCard({ accept, extensions, hint, file, onPick, onAnalyze, busy, actionLabel, error }) {
  const input = useRef(null);
  const [dragging, setDragging] = useState(false);

  const choose = (candidate) => {
    if (!candidate) return;
    const lower = candidate.name.toLowerCase();
    if (!extensions.some((ext) => lower.endsWith(ext))) {
      onPick(null, `Tệp không đúng định dạng. Chọn tệp ${extensions.join(", ")}.`);
      return;
    }
    onPick(candidate, "");
  };

  const onDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    choose(event.dataTransfer.files?.[0]);
  };

  return (
    <section className="card upload-card">
      <span className="upload-icon" aria-hidden="true" />
      <div
        className={`dropzone ${dragging ? "dragging" : ""}`}
        onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <p className="dropzone-title">{file ? file.name : "Kéo thả tệp vào đây"}</p>
        <p className="muted">{file ? `${(file.size / (1024 * 1024)).toFixed(1)} MB` : hint}</p>
        <button type="button" className="btn btn-secondary" onClick={() => input.current?.click()} disabled={busy}>
          Chọn tệp
        </button>
        <input
          ref={input}
          type="file"
          accept={accept}
          hidden
          onChange={(event) => { choose(event.target.files?.[0]); event.target.value = ""; }}
        />
      </div>
      {error && <p className="error-box" role="alert">{error}</p>}
      <button type="button" className="btn btn-primary btn-wide" onClick={onAnalyze} disabled={!file || busy}>
        {busy ? "Đang gửi…" : actionLabel}
      </button>
      <ul className="features">
        <li><strong>87 mã biển</strong><span>theo QCVN 41:2024</span></li>
        <li><strong>Giọng đọc</strong><span>tên biển bằng tiếng Việt</span></li>
        <li><strong>Trích quy định</strong><span>kèm văn bản nguồn</span></li>
      </ul>
    </section>
  );
}
