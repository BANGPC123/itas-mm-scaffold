import { errorMessage } from "./logic.js";

export const API = import.meta.env.VITE_API_BASE || (import.meta.env.DEV ? "http://localhost:8000" : "");

// Builds an Error with a Vietnamese message from a failed response; a null response means no connection.
export async function failureFrom(response, fallback) {
  if (!response) return new Error(errorMessage(0, "", fallback));
  const body = await response.json().catch(() => ({}));
  const detail = typeof body.detail === "string" ? body.detail : "";
  return new Error(errorMessage(response.status, detail, fallback));
}

export async function requestOrThrow(url, options, fallback) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw await failureFrom(null, fallback);
  }
  if (!response.ok) throw await failureFrom(response, fallback);
  return response.json();
}

export function speechText(text, signal) {
  return requestOrThrow(
    `${API}/pipeline/speech`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }), signal },
    "Không tạo được giọng đọc."
  ).then((data) => `${API}${data.audio_url}`);
}

export function legalLookup(code, name) {
  return requestOrThrow(
    `${API}/pipeline/legal-search`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: `${code} ${name || ""}`.trim(), top_k: 2 }),
    },
    "Không tra cứu được quy định."
  ).then((data) => data.hits || []);
}
