import { useEffect, useState } from "react";
import { API } from "./api.js";
import VideoTab from "./VideoTab.jsx";
import ImageTab from "./ImageTab.jsx";

function useBackendStatus() {
  const [status, setStatus] = useState("checking");
  useEffect(() => {
    let alive = true;
    fetch(`${API}/health`)
      .then((res) => alive && setStatus(res.ok ? "connected" : "error"))
      .catch(() => alive && setStatus("error"));
    return () => {
      alive = false;
    };
  }, []);
  return status;
}

// Vietnamese display names for every sign code, shared by both tabs.
function useSignNames() {
  const [names, setNames] = useState({});
  useEffect(() => {
    let alive = true;
    fetch(`${API}/pipeline/sign-catalog`)
      .then((res) => (res.ok ? res.json() : {}))
      .then((data) => alive && setNames(data.names || {}))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);
  return names;
}

const STATUS = {
  checking: { label: "Đang kiểm tra", className: "pill-checking" },
  connected: { label: "Sẵn sàng", className: "pill-ok" },
  error: { label: "Mất kết nối", className: "pill-error" },
};

export default function App() {
  const status = useBackendStatus();
  const names = useSignNames();
  // The URL hash (#image or #video) picks the starting tab, so a tab can be linked directly.
  const [tab, setTab] = useState(() => (window.location.hash === "#image" ? "image" : "video"));
  const pill = STATUS[status];

  return (
    <div className="page">
      <header className="navbar">
        <div className="navbar-inner">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true" />
            <div>
              <span className="brand-name">ITAS-MM</span>
              <span className="brand-sub">Nhận diện biển báo giao thông Việt Nam</span>
            </div>
          </div>
          <span className={`pill ${pill.className}`} role="status"><span className="dot" aria-hidden="true" />{pill.label}</span>
        </div>
      </header>

      <div className="app">
        <div className="intro">
          <h1>Phân tích biển báo từ video hoặc ảnh</h1>
          <p className="muted">Tải lên video MP4 hoặc ảnh chụp từ phía trước xe. Hệ thống đánh dấu biển báo, đọc tên bằng giọng Việt và trích quy định liên quan.</p>
          <div className="segmented" role="tablist" aria-label="Loại tệp">
            <button role="tab" type="button" aria-selected={tab === "video"} className={tab === "video" ? "active" : ""} onClick={() => setTab("video")}>
              Video
            </button>
            <button role="tab" type="button" aria-selected={tab === "image"} className={tab === "image" ? "active" : ""} onClick={() => setTab("image")}>
              Ảnh
            </button>
          </div>
        </div>

        <main>
          <div hidden={tab !== "video"}><VideoTab names={names} /></div>
          <div hidden={tab !== "image"}><ImageTab names={names} /></div>
        </main>

        <footer className="fine">
          Kết quả do AI dự đoán, chỉ mang tính tham khảo; không thay thế quan sát của người lái.
        </footer>
      </div>
    </div>
  );
}
