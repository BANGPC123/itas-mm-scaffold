import { useRef, useState } from "react";
import { API, requestOrThrow, speechText } from "./api.js";
import UploadCard from "./UploadCard.jsx";
import LegalLookup from "./LegalLookup.jsx";
import { badgeStyle, imageRows } from "./logic.js";

// Fixed position for the single-image request; the user does not enter coordinates.
const DEFAULT_POSITION = { latitude: "10.7769", longitude: "106.7009" };

export default function ImageTab({ names }) {
  const [file, setFile] = useState(null);
  const [pickError, setPickError] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [speechError, setSpeechError] = useState("");
  const audio = useRef(null);

  const pick = (chosen, problem) => {
    setPickError(problem);
    if (chosen) {
      setFile(chosen);
      setResult(null);
      setError("");
    }
  };

  const analyze = async () => {
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    setSpeechError("");
    try {
      const body = new FormData();
      body.append("image", file);
      body.append("latitude", DEFAULT_POSITION.latitude);
      body.append("longitude", DEFAULT_POSITION.longitude);
      body.append("synthesize_audio", "false");
      body.append("annotate", "true");
      body.append("auxiliary_sign_review", "false");
      setResult(await requestOrThrow(`${API}/pipeline/run`, { method: "POST", body }, "Không phân tích được ảnh."));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    setFile(null);
    setResult(null);
    setError("");
    setSpeechError("");
  };

  const rows = result ? imageRows(result.perception.signs, names) : [];
  const spoken = result
    ? result.reasoning.observation_speech_text || result.reasoning.speech_text || rows.filter((r) => !r.unknown).map((r) => `${r.code}: ${r.name}`).join(". ")
    : "";

  const read = async () => {
    setSpeechError("");
    try {
      const url = await speechText(spoken);
      audio.current ??= new Audio();
      audio.current.src = url;
      await audio.current.play();
    } catch (err) {
      setSpeechError(err.message || "Không phát được giọng đọc.");
    }
  };

  if (!result) {
    return (
      <UploadCard
        accept="image/jpeg,image/png,.jpg,.jpeg,.png"
        extensions={[".jpg", ".jpeg", ".png"]}
        hint="Ảnh JPG hoặc PNG chụp từ phía trước xe."
        file={file}
        onPick={pick}
        onAnalyze={analyze}
        busy={busy}
        actionLabel="Phân tích ảnh"
        error={pickError || error}
      />
    );
  }

  return (
    <div className="result">
      <section className="card">
        <img className="result-image" src={result.annotated_image} alt="Ảnh có khung các biển báo đã nhận diện" />
        <div className="result-actions">
          <button type="button" className="btn btn-secondary" onClick={read} disabled={!spoken}>Đọc kết quả</button>
          <button type="button" className="btn btn-secondary" onClick={reset}>Phân tích ảnh khác</button>
        </div>
        {speechError && <p className="fine" role="status">{speechError}</p>}
        {result.warnings?.length > 0 && (
          <details className="tech">
            <summary>Chi tiết kỹ thuật</summary>
            <ul>{result.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
            <p className="muted">Thời gian xử lý: {result.timings_ms?.total?.toFixed(0) ?? "—"} ms</p>
          </details>
        )}
      </section>

      <section className="card">
        <h2>Biển báo phát hiện</h2>
        {rows.length === 0 ? (
          <p className="muted">Chưa phát hiện biển báo nào trong ảnh này.</p>
        ) : (
          <ul className="sign-list">
            {rows.map((row) => (
              <li className="sign-row" key={row.key}>
                {row.unknown ? (
                  <span className="badge badge-unknown">?</span>
                ) : (
                  <span className="badge" style={badgeStyle(row.code)}>{row.code}</span>
                )}
                <span className="sign-name">
                  {row.name}
                  <small className="muted"> · {row.confidence}%</small>
                </span>
                {row.canLookUp && <LegalLookup code={row.code} name={row.name} />}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
