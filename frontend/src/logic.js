// Pure helpers for the upload-only web UI. No React or DOM access here, so they run under node --test.

export function formatTime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

export function signGroup(code) {
  const prefix = String(code || "").split(".")[0];
  if (prefix === "P" || prefix === "DP") return "prohibition";
  if (prefix === "W") return "warning";
  if (prefix === "R") return "mandatory";
  if (prefix === "I") return "information";
  return "other";
}

const BADGE = {
  prohibition: { background: "#c0392b", color: "#ffffff" },
  warning: { background: "#f5c518", color: "#1f2937" },
  mandatory: { background: "#1d4ed8", color: "#ffffff" },
  information: { background: "#15803d", color: "#ffffff" },
  other: { background: "#6b7280", color: "#ffffff" },
};

export function badgeStyle(code) {
  return BADGE[signGroup(code)];
}

// Seconds left, from the time spent so far and the share of frames done. Null until there is progress.
export function estimateRemaining(elapsedSeconds, done, total) {
  if (!(total > 0) || !(done > 0)) return null;
  if (done >= total) return 0;
  return (elapsedSeconds * (total - done)) / done;
}

export function formatRemaining(seconds) {
  if (seconds === null || seconds === undefined) return "Đang tính thời gian còn lại…";
  if (seconds < 60) return `Còn khoảng ${Math.max(1, Math.round(seconds))} giây`;
  return `Còn khoảng ${Math.round(seconds / 60)} phút`;
}

// Rows for the video result list, taken from the confirmed cues of the result video.
export function videoRows(cues, names) {
  const rows = [];
  for (const cue of cues || []) {
    for (const code of cue.codes || []) {
      rows.push({
        key: `${cue.t_seconds}-${code}`,
        seconds: cue.t_seconds,
        time: formatTime(cue.t_seconds),
        code,
        name: (names && names[code]) || "",
      });
    }
  }
  return rows;
}

// Rows for the image result list. Unrecognised panels are kept, labelled as such.
export function imageRows(signs, names) {
  return (signs || []).map((sign, index) => {
    const unknown = sign.label === "unknown";
    const code = sign.is_compound ? (sign.panel_labels || []).join(" + ") : sign.label;
    return {
      key: index,
      code: unknown ? "" : code,
      name: unknown ? "Chưa nhận dạng" : sign.name_vi || (names && names[sign.label]) || "",
      confidence: Math.round((sign.recognition_confidence ?? sign.confidence ?? 0) * 100),
      box: Array.isArray(sign.bbox_xyxy) ? sign.bbox_xyxy : null,
      unknown,
      canLookUp: !unknown && !sign.is_compound,
    };
  });
}

export function hitLocator(hit) {
  return hit.locator || (hit.page ? `trang PDF ${hit.page}` : "đoạn nguồn");
}

export function shortExcerpt(text, max = 360) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  return clean.length <= max ? clean : `${clean.slice(0, max - 1).trimEnd()}…`;
}

const NETWORK = "Mất kết nối tới máy chủ. Kiểm tra máy chủ đang chạy rồi thử lại.";

// Maps an HTTP failure to a Vietnamese message. status is 0 or undefined when the request never arrived.
export function errorMessage(status, detail, fallback) {
  if (!status) return NETWORK;
  if (status === 409) return "Hệ thống đang xử lý một video khác, vui lòng thử lại sau.";
  if (status === 429) return "Hệ thống đang bận xử lý khung hình khác, vui lòng thử lại sau.";
  if (typeof detail === "string" && detail) return detail;
  return fallback;
}

// Square crop around a detected box, in displayed-image pixels, with padding. The box is in frame
// coordinates; the displayed image may be scaled. Returns null for a missing or empty box.
export function cropRect(box, imageWidth, imageHeight, frameWidth, frameHeight, pad = 0.3) {
  if (!box || !(imageWidth > 0) || !(imageHeight > 0)) return null;
  const sx = imageWidth / (frameWidth || imageWidth);
  const sy = imageHeight / (frameHeight || imageHeight);
  const [x1, y1, x2, y2] = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy];
  const w = x2 - x1;
  const h = y2 - y1;
  if (!(w > 0) || !(h > 0)) return null;
  const side = Math.min(Math.max(w, h) * (1 + 2 * pad), imageWidth, imageHeight);
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  const x = Math.min(Math.max(0, cx - side / 2), imageWidth - side);
  const y = Math.min(Math.max(0, cy - side / 2), imageHeight - side);
  return [x, y, side, side];
}
