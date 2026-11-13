import { useEffect, useRef, useState } from "react";
import { API, requestOrThrow, speechText } from "./api.js";
import UploadCard from "./UploadCard.jsx";
import LegalLookup from "./LegalLookup.jsx";
import { createVideoSpeech } from "./videoSpeech.js";
import { takeDueEvents, handledBefore } from "./speechEvents.js";
import { badgeStyle, estimateRemaining, forwardRows, formatRemaining, videoRows } from "./logic.js";

const ACTIVE = ["queued", "waiting_for_inference", "running"];

function useNow(enabled) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!enabled) return undefined;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [enabled]);
  return now;
}

function Progress({ job, runSince, now }) {
  const done = job.progress?.encoded_source_frames ?? 0;
  const total = job.progress?.expected ?? 0;
  const percent = total > 0 ? Math.min(100, Math.round((done * 100) / total)) : 0;
  const elapsed = runSince ? (now - runSince) / 1000 : 0;
  const preparing = total === 0;
  const remaining = job.status === "running" && !preparing ? estimateRemaining(elapsed, done, total) : null;
  const label = {
    queued: "Đang chờ trong hàng xử lý",
    waiting_for_inference: "Đang chờ máy xử lý",
    running: "Đang phân tích video",
  }[job.status] || "Đang xử lý";
  return (
    <section className="card narrow">
      <h2>{label}</h2>
      <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}>
        <span style={{ width: `${percent}%` }} />
      </div>
      {preparing ? (
        <p className="muted">Đang chuẩn bị video…</p>
      ) : (
        <>
          <p className="muted">{done} / {total} khung hình · {percent}%</p>
          {job.status === "running" && <p className="muted">{formatRemaining(remaining)}</p>}
        </>
      )}
      <p className="fine">Giữ trang này mở. Video dài sẽ mất nhiều thời gian hơn độ dài của video.</p>
    </section>
  );
}

export default function VideoTab({ names }) {
  const [file, setFile] = useState(null);
  const [pickError, setPickError] = useState("");
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState("");
  const [job, setJob] = useState(null);
  const [runSince, setRunSince] = useState(null);
  const [speechOn, setSpeechOn] = useState(true);
  const [speechError, setSpeechError] = useState("");
  const video = useRef(null);
  const audio = useRef(null);
  const speaker = useRef(null);
  const handled = useRef(new Set());
  const running = ACTIVE.includes(job?.status);
  const now = useNow(running);

  // Progress polling for the active job; stops at complete or failed.
  useEffect(() => {
    if (!job?.job_id || !running) return undefined;
    let alive = true;
    const id = setInterval(async () => {
      try {
        const reply = await fetch(`${API}/pipeline/offline-video/jobs/${job.job_id}`);
        if (!alive) return;
        if (reply.ok) setJob(await reply.json());
        else if (reply.status === 404) {
          // The server keeps jobs in memory; after a restart the old job is gone. Stop polling.
          setJob((prev) => ({
            ...prev,
            status: "failed",
            message: "Máy chủ đã khởi động lại nên không còn công việc này. Hãy tải video lên lại.",
          }));
        }
      } catch {
        // Keep the last progress; the connection may come back.
      }
    }, 3500);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [job?.job_id, running]);

  useEffect(() => {
    if (job?.status === "running" && runSince === null) setRunSince(Date.now());
  }, [job?.status, runSince]);

  const player = () => (audio.current ??= new Audio());

  const cueSpeaker = () => {
    speaker.current ??= createVideoSpeech({
      maxAge: 8,
      queueWait: 2,
      synthesize: (text) => speechText(text),
      play: (url) => new Promise((resolve, reject) => {
        const element = player();
        element.src = url;
        element.onended = resolve;
        element.onpause = resolve;
        element.onerror = () => reject(new Error("Không phát được giọng đọc."));
        element.play().catch(() => reject(new Error("Trình duyệt chặn tự phát âm thanh. Nhấn phát ở thanh giọng đọc.")));
      }),
      onError: (error) => setSpeechError(error.message),
    });
    return speaker.current;
  };

  const silence = () => {
    speaker.current?.reset();
    audio.current?.pause();
  };

  const pick = (chosen, error) => {
    setPickError(error);
    if (chosen) setFile(chosen);
  };

  const analyze = async () => {
    if (!file) return;
    setStartError("");
    setStarting(true);
    setJob(null);
    setRunSince(null);
    handled.current = new Set();
    speaker.current?.reset();
    try {
      const body = new FormData();
      body.append("file", file);
      body.append("analysis_hz", "6");
      body.append("auxiliary_review", "false");
      setJob(await requestOrThrow(`${API}/pipeline/offline-video/start`, { method: "POST", body }, "Không khởi động được phân tích video."));
    } catch (error) {
      setStartError(error.message);
    } finally {
      setStarting(false);
    }
  };

  const reset = () => {
    silence();
    setFile(null);
    setJob(null);
    setRunSince(null);
    setStartError("");
    setSpeechError("");
    handled.current = new Set();
  };

  const seekTo = (seconds) => {
    const element = video.current;
    if (!element) return;
    element.currentTime = seconds;
    element.play().catch(() => {});
  };

  if (!job || starting) {
    return (
      <UploadCard
        accept="video/mp4,.mp4"
        extensions={[".mp4"]}
        hint="Tệp MP4. Video dài nên được cắt xuống vài phút."
        file={file}
        onPick={pick}
        onAnalyze={analyze}
        busy={starting}
        actionLabel="Phân tích video"
        error={pickError || startError}
      />
    );
  }

  if (running) return <Progress job={job} runSince={runSince} now={now} />;

  if (job.status === "failed") {
    return (
      <section className="card narrow">
        <h2>Không phân tích được video</h2>
        <p className="error-box" role="alert">{job.message || "Phân tích gặp lỗi; kết quả chưa được công nhận."}</p>
        <button type="button" className="btn btn-primary" onClick={reset}>Thử video khác</button>
      </section>
    );
  }

  const cues = job.speech_events || [];
  // M9: the sign list comes from sign_cues, which exist even when sign speech is off.
  const rows = videoRows(job.sign_cues || [], names);
  const forward = job.forward_warning;
  const alerts = forwardRows(forward?.alerts);
  const spokenAlerts = alerts.filter((row) => row.spoken).length;
  return (
    <div className="result">
      <section className="card">
        <video
          ref={video}
          className="result-video"
          controls
          playsInline
          preload="metadata"
          src={API + job.video_url}
          onPlay={() => { if (speechOn) cueSpeaker(); }}
          onTimeUpdate={(event) => {
            if (!speechOn) return;
            const element = event.currentTarget;
            const speak = cueSpeaker();
            for (const cue of takeDueEvents(cues, element.currentTime, handled.current)) {
              speak.update(cue.text, cue.t_seconds, () => element.currentTime);
            }
          }}
          onSeeked={(event) => {
            handled.current = handledBefore(cues, event.currentTarget.currentTime);
            silence();
          }}
        />
        <div className="result-actions">
          <label className="toggle">
            <input type="checkbox" checked={speechOn} onChange={(event) => { setSpeechOn(event.target.checked); if (!event.target.checked) silence(); }} />
            Giọng đọc
          </label>
          <a className="btn btn-secondary" href={API + job.video_url} download="itas_mm_ket_qua.mp4">Tải video kết quả</a>
          <button type="button" className="btn btn-secondary" onClick={reset}>Phân tích video khác</button>
        </div>
        {speechError && <p className="fine" role="status">{speechError}</p>}
        <details className="tech">
          <summary>Chi tiết kỹ thuật</summary>
          <p className="muted">Mã công việc: {job.job_id}</p>
          <p className="muted">Khung hình xử lý: {job.progress?.encoded_source_frames ?? "—"}</p>
          {job.report_url && <p><a href={API + job.report_url}>Tải báo cáo kiểm chứng (JSON)</a></p>}
        </details>
      </section>

      {forward?.enabled && (
        <section className="card">
          <div className="list-head">
            <h2>Cảnh báo phía trước</h2>
            <span className="count">{alerts.length} cảnh báo</span>
          </div>
          <p className="fine">
            Khung cam: chú ý · khung đỏ: nguy cơ. Chỉ một số cảnh báo được đọc to ({spokenAlerts} câu
            {forward.spoken_per_minute != null ? `, ${forward.spoken_per_minute} câu/phút` : ""}).
            Bấm mốc thời gian để tua tới cảnh báo.
          </p>
          {alerts.length === 0 ? (
            <p className="muted">Không có vật cản nào vào vùng đường đi trong video này.</p>
          ) : (
            <ul className="sign-list">
              {alerts.map((row) => (
                <li className="sign-row" key={row.key}>
                  <button type="button" className="time-chip" onClick={() => seekTo(row.seconds)} aria-label={`Tua tới ${row.time}`}>
                    {row.time}
                  </button>
                  <span className={row.levelLabel === "Nguy cơ" ? "badge badge-danger" : "badge badge-caution"}>{row.levelLabel}</span>
                  <span className="sign-name">
                    {row.label}
                    <small className="muted">{row.spokenLabel}</small>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      <section className="card">
        <div className="list-head">
          <h2>Biển báo phát hiện</h2>
          <span className="count">{rows.length} biển</span>
        </div>
        <p className="fine">Bấm mốc thời gian để tua tới biển đó.</p>
        {rows.length === 0 ? (
          <p className="muted">Không phát hiện biển báo nào được xác nhận.</p>
        ) : (
          <ul className="sign-list">
            {rows.map((row) => (
              <li className="sign-row" key={row.key}>
                <button type="button" className="time-chip" onClick={() => seekTo(row.seconds)} aria-label={`Tua tới ${row.time}`}>
                  {row.time}
                </button>
                <span className="badge" style={badgeStyle(row.code)}>{row.code}</span>
                <span className="sign-name">{row.name}</span>
                <LegalLookup code={row.code} name={row.name} />
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
