import { useState } from "react";
import { legalLookup } from "./api.js";
import { hitLocator, shortExcerpt } from "./logic.js";

// Opens the cited legal passages for one sign code. Fetched once, then kept while the page is open.
export default function LegalLookup({ code, name }) {
  const [open, setOpen] = useState(false);
  const [state, setState] = useState({ loading: false, hits: null, error: "" });
  const [all, setAll] = useState(false);

  const toggle = async () => {
    const next = !open;
    setOpen(next);
    if (!next || state.hits || state.loading) return;
    setState({ loading: true, hits: null, error: "" });
    try {
      setState({ loading: false, hits: await legalLookup(code, name), error: "" });
    } catch (error) {
      setState({ loading: false, hits: null, error: error.message });
    }
  };

  return (
    <>
      <button type="button" className={`chip-button ${open ? "on" : ""}`} aria-expanded={open} onClick={toggle}>
        {open ? "Ẩn quy định" : "Quy định"}
      </button>
      {open && (
        <div className="legal-body">
          {state.loading && <p className="muted">Đang tra cứu…</p>}
          {state.error && <p className="error-text" role="alert">{state.error}</p>}
          {state.hits && state.hits.length === 0 && <p className="muted">Không có đoạn nguồn đủ phù hợp.</p>}
          {state.hits?.slice(0, all ? undefined : 1).map((hit, index) => (
            <article className="legal-hit" key={`${hit.source_file}-${hit.page}-${index}`}>
              <h4>{hit.document}</h4>
              <p className="muted">{hitLocator(hit)}</p>
              <p className="excerpt">{shortExcerpt(hit.excerpt)}</p>
              <a href={hit.source_url + (hit.page ? `#page=${hit.page}` : "")} target="_blank" rel="noreferrer">
                Mở văn bản nguồn
              </a>
            </article>
          ))}
          {state.hits?.length > 1 && (
            <button type="button" className="link-button" onClick={() => setAll(!all)}>
              {all ? "Thu gọn" : `Xem thêm ${state.hits.length - 1} đoạn`}
            </button>
          )}
          {state.hits?.length > 0 && <p className="fine">Chưa kiểm tra hiệu lực pháp lý.</p>}
        </div>
      )}
    </>
  );
}
