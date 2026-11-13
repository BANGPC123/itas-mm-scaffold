import { useRef, useState } from "react";
import { API, requestOrThrow, speechText } from "./api.js";
import UploadCard from "./UploadCard.jsx";
import LegalLookup from "./LegalLookup.jsx";
import SignThumb from "./SignThumb.jsx";
import { badgeStyle, imageRows, obstacleBoxes } from "./logic.js";

// Fixed position for the single-image request; the user does not enter coordinates.
const DEFAULT_POSITION = { latitude: "10.7769", longitude: "106.7009" };

export default function ImageTab({ names }) {
  const [file, setFile] = useState(null);
  const [pickError, setPickError] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [speechError, setSpeechError] = useState("");
  const [obstacles, setObstacles] = useState(null);
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
    setObstacles(null);
    try {
      const body = new FormData();
      body.append("image", file);
      body.append("latitude", DEFAULT_POSITION.latitude);
      body.append("longitude", DEFAULT_POSITION.longitude);
      body.append("synthesize_audio", "false");
      body.append("annotate", "true");
      body.append("auxiliary_sign_review", "false");
      const reply = await requestOrThrow(`${API}/pipeline/run`, { method: "POST", body }, "Không phân tích được ảnh.");
      setResult(reply);
      // M9: obstacle boxes come from a second request; a failure only hides them.
      const obstacleBody = new FormData();
      obstacleBody.append("image", file);
      requestOrThrow(`${API}/pipeline/forward-objects`, { method: "POST", body: obstacleBody }, "Không phát hiện được vật cản.")
        .then((data) => setObstacles(data))
        .catch(() => setObstacles({ enabled: true, objects: [], failed: true }));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    setFile(null);
    setResult(null);
    setObstacles(null);
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
        <div className="obstacle-wrap">
          <img className="result-image" src={result.annotated_image} alt="Ảnh có khung các biển báo đã nhận diện" />
          {obstacleBoxes(obstacles?.objects, result.frame_shape).map((box) => (
            <div
              key={box.key}
              className={`obstacle-box ${box.inZone ? "obstacle-box-zone" : "obstacle-box-outside"}`}
              style={{ left: `${box.left}%`, top: `${box.top}%`, width: `${box.width}%`, height: `${box.height}%` }}
            >
              <span>{box.label}</span>
            </div>
          ))}
        </div>
        {obstacles?.enabled && (
          <p className="fine">
            Khung cam: vật cản nằm trong vùng đường đi phía trước; khung xám: ngoài vùng. Đây là phát hiện tự động, chưa xác minh.
          </p>
        )}
        <div className="result-actions">
          {result.sign_speech_enabled && (
            <button type="button" className="btn btn-secondary" onClick={read} disabled={!spoken}>Đọc kết quả</button>
          )}
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
        <div className="list-head">
          <h2>Biển báo phát hiện</h2>
          <span className="count">{rows.filter((r) => !r.unknown).length} nhận dạng · {rows.length} phát hiện</span>
        </div>
        {rows.length === 0 ? (
          <p className="muted">Chưa phát hiện biển báo nào trong ảnh này.</p>
        ) : (
          <ul className="sign-list">
            {rows.map((row) => (
              <li className="sign-row" key={row.key}>
                <SignThumb src={result.annotated_image} box={row.box} frameShape={result.frame_shape} />
                {row.unknown ? (
                  <span className="badge badge-unknown">?</span>
                ) : (
                  <span className="badge" style={badgeStyle(row.code)}>{row.code}</span>
                )}
                <span className="sign-name">
                  {row.name}
                  <small className="muted">Độ tin cậy {row.confidence}%</small>
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
