"""Extract and normalize legal-document text layers."""
from __future__ import annotations

import html
from html.parser import HTMLParser
from pathlib import Path
import re
import unicodedata

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


def normalize_vietnamese(value: str) -> str:
    """Normalize whitespace and Unicode as in the pinned LuatRAG snapshot."""
    value = unicodedata.normalize("NFC", value).replace("\0", "")
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[\t\f\v]+", " ", value)
    value = re.sub(r" {2,}", " ", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def html_to_text(value: str) -> str:
    """Remove markup while retaining the pinned block boundaries."""
    value = re.sub(r"<!--[\s\S]*?-->", " ", value)
    value = re.sub(r"<(script|style|head)\b[^>]*>[\s\S]*?</\1\s*>", " ", value, flags=re.I)
    value = re.sub(r"<br\s*/?\s*>", "\n", value, flags=re.I)
    value = re.sub(
        r"</\s*(?:p|div|li|tr|td|th|h[1-6]|table|section|article)\s*>", "\n", value, flags=re.I
    )
    value = re.sub(r"<li\b[^>]*>", "• ", value, flags=re.I)
    value = html.unescape(re.sub(r"<[^>]+>", " ", value))
    value = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]", "", value)
    return normalize_vietnamese(re.sub(r" *\n *", "\n", value))


class _ArticleBodyExtractor(HTMLParser):
    _BLOCK_TAGS = {"br", "div", "li", "p", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6"}
    _HIDDEN_TAGS = {"script", "style", "head"}
    _VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.depth = 0
        self.hidden = 0
        self.found = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self.depth:
            if self.found or dict(attrs).get("itemprop") != "articleBody":
                return
            self.depth = 1
            self.found = True
        elif tag not in self._VOID_TAGS:
            self.depth += 1
        if tag in self._HIDDEN_TAGS:
            self.hidden += 1
        if tag == "li":
            self.parts.append("• ")
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if not self.depth:
            return
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")
        if tag in self._HIDDEN_TAGS and self.hidden:
            self.hidden -= 1
        if tag not in self._VOID_TAGS:
            self.depth -= 1

    def handle_data(self, data: str) -> None:
        if self.depth and not self.hidden:
            self.parts.append(data)


def extract_vbpl_text(payload: dict) -> str:
    """Return normalized text from VBPL's configured document-content field."""
    try:
        value = payload["data"]["documentContent"]["content"]
    except (KeyError, TypeError) as exc:
        raise ValueError("VBPL payload does not contain documentContent.content") from exc
    if not isinstance(value, str):
        raise ValueError("VBPL payload does not contain documentContent.content")
    return html_to_text(value)


def extract_official_article_text(page_html: str) -> str:
    """Return only an official page's schema.org article body."""
    parser = _ArticleBodyExtractor()
    parser.feed(page_html)
    parser.close()
    if not parser.found:
        raise ValueError("Official page does not contain an articleBody")
    return normalize_vietnamese(re.sub(r" *\n *", "\n", "".join(parser.parts)))


def _resolved(value):
    return value.get_object() if hasattr(value, "get_object") else value


def extract_pdf_text(path: str | Path) -> str:
    """Return text from PDF pages with font resources, failing closed otherwise."""
    if PdfReader is None:
        raise RuntimeError("pypdf is required for PDF text extraction")
    text = []
    for page in PdfReader(path).pages:
        resources = _resolved(page.get("/Resources", {})) or {}
        if "/Font" not in resources:
            continue
        text.append(page.extract_text() or "")
    normalized = normalize_vietnamese("\n".join(text))
    if len(normalized) < 20 or not re.search(r"\w", normalized, re.UNICODE):
        raise ValueError("PDF has no usable text layer; OCR is not supported")
    return normalized


def normalize_document(*_args, **_kwargs):
    """Reject the removed hierarchy-normalization workflow."""
    raise RuntimeError("normalize_document was replaced by extraction and chunking primitives")
