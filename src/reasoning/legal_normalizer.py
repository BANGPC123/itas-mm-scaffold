"""Extract text layers and preserve only explicit legal hierarchy."""
from __future__ import annotations

from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
import re
import unicodedata

try:
    from pypdf import PdfReader
except ImportError:  # Allows non-PDF commands to explain a missing optional runtime package.
    PdfReader = None

from pydantic import ValidationError

from src.reasoning.legal_models import (
    LegalArticle,
    LegalClause,
    LegalDocument,
    LegalPoint,
    LegalSection,
    LegalSource,
)


class _TextExtractor(HTMLParser):
    _BLOCK_TAGS = {"br", "div", "li", "p", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def _normalized_lines(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", line).strip()
        for line in unicodedata.normalize("NFC", text).replace("\r", "").split("\n")
        if line.strip()
    ]


def extract_vbpl_text(payload: dict) -> str:
    """Return NFC text from the configured VBPL JSON record's HTML body."""
    body = next(
        (
            payload[key]
            for key in ("content", "contentHtml", "htmlContent", "documentContent", "fullText")
            if isinstance(payload.get(key), str)
        ),
        None,
    )
    if body is None and isinstance(payload.get("data"), dict):
        return extract_vbpl_text(payload["data"])
    if body is None and isinstance(payload.get("documentContent"), dict):
        return extract_vbpl_text(payload["documentContent"])
    if not isinstance(body, str):
        raise ValueError("VBPL payload does not contain HTML text")
    parser = _TextExtractor()
    parser.feed(body)
    parser.close()
    return "\n".join(_normalized_lines("".join(parser.parts)))


def extract_pdf_text(path: str | Path) -> str:
    """Return the PDF text layer; scanned or near-empty files fail closed."""
    if PdfReader is None:
        raise RuntimeError("pypdf is required for PDF text extraction")
    text = "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
    normalized = "\n".join(_normalized_lines(text))
    if len(normalized) < 20 or not re.search(r"\w", normalized, re.UNICODE):
        raise ValueError("PDF has no usable text layer; OCR is not supported")
    return normalized


_ARTICLE = re.compile(r"^Điều\s+(\d+[A-Za-z]?)\s*[\.:]\s*(.*)$", re.IGNORECASE)
_CLAUSE = re.compile(r"^(?:Khoản\s+)?(\d+)\.\s*(.+)$", re.IGNORECASE)
_POINT = re.compile(r"^(?:Điểm\s+)?([a-zđ])\)\s*(.+)$", re.IGNORECASE)
_SECTION = re.compile(r"^(\d+)\.\s*(.+)$")
_SUBSECTION = re.compile(r"^(\d+(?:\.\d+)+)\.\s*(.+)$")
_STRUCTURAL_HEADING = re.compile(
    r"^(?:PHẦN|CHƯƠNG|MỤC|TIỂU\s+MỤC)\b", re.IGNORECASE
)


def _append(node: dict | None, text: str) -> bool:
    if node is None:
        return False
    node["text"] = " ".join(part for part in (node.get("text", ""), text) if part).strip()
    return True


def _finish_unresolved(unresolved: list[str], lines: list[str]) -> None:
    if lines:
        unresolved.append("\n".join(lines))
        lines.clear()


def _parse_law(lines: list[str]) -> tuple[list[dict], list[str]]:
    articles: list[dict] = []
    unresolved: list[str] = []
    unknown: list[str] = []
    article = clause = point = None
    for line in lines:
        if match := _ARTICLE.match(line):
            _finish_unresolved(unresolved, unknown)
            article = {"article_id": match.group(1), "clauses": []}
            if match.group(2):
                article["text"] = match.group(2)
            articles.append(article)
            clause = point = None
        elif match := _CLAUSE.match(line):
            if article is None:
                unknown.append(line)
                continue
            clause = {"clause_id": match.group(1), "text": match.group(2), "points": []}
            article["clauses"].append(clause)
            point = None
        elif match := _POINT.match(line):
            if clause is None:
                unknown.append(line)
                continue
            point = {"point_id": match.group(1), "text": match.group(2)}
            clause["points"].append(point)
        elif _STRUCTURAL_HEADING.match(line):
            unknown.append(line)
            article = clause = point = None
        elif not _append(point or clause or article, line):
            unknown.append(line)
    _finish_unresolved(unresolved, unknown)
    return articles, unresolved


def _parse_qcvn(lines: list[str]) -> tuple[list[dict], list[str]]:
    sections: list[dict] = []
    unresolved: list[str] = []
    unknown: list[str] = []
    section = article = None
    for line in lines:
        if match := _SUBSECTION.match(line):
            if section is None or not match.group(1).startswith(f"{section['section_id']}."):
                unknown.append(line)
                continue
            article = {"article_id": match.group(1), "text": match.group(2)}
            section["articles"].append(article)
        elif match := _SECTION.match(line):
            _finish_unresolved(unresolved, unknown)
            section = {"section_id": match.group(1), "text": match.group(2), "articles": []}
            sections.append(section)
            article = None
        elif _STRUCTURAL_HEADING.match(line):
            unknown.append(line)
            section = article = None
        elif not _append(article or section, line):
            unknown.append(line)
    _finish_unresolved(unresolved, unknown)
    return sections, unresolved


def _assisted_nodes(
    unresolved: list[str], assist: Callable[[str], dict] | None, document_type: str
) -> tuple[list[LegalArticle], list[LegalSection], list[str]]:
    if assist is None:
        return [], [], unresolved
    articles: list[LegalArticle] = []
    sections: list[LegalSection] = []
    for block in unresolved:
        try:
            if document_type.casefold() == "qcvn":
                sections.append(LegalSection.model_validate(assist(block)))
            else:
                articles.append(LegalArticle.model_validate(assist(block)))
        except (TypeError, ValidationError) as exc:
            raise ValueError("invalid assist payload") from exc
    return articles, sections, []


def normalize_document(
    raw_text: str,
    *,
    document_id: str,
    title: str,
    document_type: str,
    source: LegalSource,
    assist: Callable[[str], dict] | None = None,
) -> tuple[LegalDocument, list[str]]:
    """Normalize explicit headings and return text blocks that need review."""
    lines = _normalized_lines(raw_text)
    if document_type.casefold() == "qcvn":
        section_data, unresolved = _parse_qcvn(lines)
        articles: list[LegalArticle] = []
        sections = [LegalSection.model_validate(item) for item in section_data]
    else:
        article_data, unresolved = _parse_law(lines)
        articles = [LegalArticle.model_validate(item) for item in article_data]
        sections = []
    assisted_articles, assisted_sections, unresolved = _assisted_nodes(
        unresolved, assist, document_type
    )
    return (
        LegalDocument(
            document_id=document_id,
            title=title,
            document_type=document_type,
            source=source,
            articles=articles + assisted_articles,
            sections=sections + assisted_sections,
        ),
        unresolved,
    )
