"""Acquire curated official regulations and publish one immutable corpus."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import requests

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reasoning.document_loader import chunk_regulation_text
from src.reasoning.legal_normalizer import (
    extract_official_article_text,
    extract_pdf_text,
    extract_vbpl_text,
)


VBPL_API = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{}"
REQUEST_TIMEOUT_SECONDS = 30
MAX_DOCUMENT_CHARS = 600_000
MAX_TOTAL_CHARS = 8_000_000
UPSTREAM_REPOSITORY = "lqb464/LuatRAG"
UPSTREAM_COMMIT = "ae2b1c796503e2a58493771bc341b66fb488e053"


def _required(entry: dict, field: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"source entry requires {field}")
    return value


def fetch_with_retries(url: str, *, attempts: int = 4) -> requests.Response:
    """Fetch an official URL with the pinned transient-status retry policy."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(min(10, 0.75 * 2**attempt))
            continue
        if response.status_code not in {408, 429} and response.status_code < 500:
            response.raise_for_status()
            return response
        if attempt == attempts - 1:
            response.raise_for_status()
        time.sleep(min(10, 0.75 * 2**attempt))
    raise RuntimeError("unreachable retry state")


def _write_immutable(path: Path, raw_bytes: bytes) -> None:
    if path.exists():
        if path.read_bytes() != raw_bytes:
            raise ValueError(f"immutable raw artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as artifact:
            artifact.write(raw_bytes)
    except FileExistsError:
        if path.read_bytes() != raw_bytes:
            raise ValueError(f"immutable raw artifact differs: {path}")


def _replace_corpus(path: Path, corpus: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            json.dump(corpus, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _extract_pdf_bytes(raw_bytes: bytes) -> str:
    descriptor, temporary_path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(raw_bytes)
        return extract_pdf_text(temporary_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _candidate(
    entry: dict, retrieved_at: str
) -> tuple[Path, bytes, dict, list[dict], int]:
    document_id = _required(entry, "document_id")
    source_kind = _required(entry, "source_kind")
    source_url = _required(entry, "source_url")
    expected_number = _required(entry, "expected_document_number")
    title = _required(entry, "title")

    if source_kind == "vbpl_json":
        response = fetch_with_retries(VBPL_API.format(_required(entry, "item_id")))
        raw_bytes, extension = response.content, "json"
        try:
            payload = json.loads(raw_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"VBPL item {entry['item_id']} did not return JSON") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        number = data.get("docNum") if isinstance(data, dict) else None
        if number != expected_number:
            raise ValueError(
                f"VBPL document number mismatch for {entry['item_id']}: "
                f"expected {expected_number!r}, got {number!r}"
            )
        text = extract_vbpl_text(payload)
        if isinstance(data.get("title"), str) and data["title"].strip():
            title = data["title"].strip()
    elif source_kind == "official_html":
        response = fetch_with_retries(source_url)
        raw_bytes, extension = response.content, "html"
        page_html = raw_bytes.decode("utf-8")
        number = expected_number
        if expected_number not in page_html:
            raise ValueError(f"official HTML document number mismatch: {expected_number!r}")
        text = extract_official_article_text(page_html)
    elif source_kind == "official_pdf":
        response = fetch_with_retries(_required(entry, "attachment_url"))
        raw_bytes, extension = response.content, "pdf"
        if not raw_bytes:
            raise ValueError(f"attachment for {document_id} is empty")
        text = _extract_pdf_bytes(raw_bytes)
        number = expected_number
        if expected_number not in text:
            raise ValueError(f"official PDF document number mismatch: {expected_number!r}")
    else:
        raise ValueError(f"unsupported source_kind: {source_kind}")

    if not text:
        raise ValueError(f"source text is empty: {document_id}")
    if len(text) > MAX_DOCUMENT_CHARS:
        raise ValueError(f"source text exceeds {MAX_DOCUMENT_CHARS} characters: {document_id}")
    chunks = chunk_regulation_text(document_id, text)
    raw_file = f"raw/{document_id}/source.{extension}"
    source = {
        "document_id": document_id,
        "document_number": number,
        "title": title,
        "source_kind": source_kind,
        "source_url": source_url,
        "raw_file": raw_file,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "retrieved_at": retrieved_at,
        "chunk_count": len(chunks),
    }
    return Path(document_id) / f"source.{extension}", raw_bytes, source, chunks, len(text)


def fetch_regulations(
    config_path: str | Path, raw_dir: str | Path, corpus_path: str | Path
) -> dict:
    """Acquire configured sources and atomically replace a validated corpus."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    documents = config.get("documents") if isinstance(config, dict) else None
    if not isinstance(documents, list) or not documents:
        raise ValueError("config requires a non-empty documents list")

    seen_ids: set[str] = set()
    for entry in documents:
        if not isinstance(entry, dict):
            raise ValueError("each source entry must be an object")
        document_id = _required(entry, "document_id")
        if document_id in seen_ids:
            raise ValueError(f"duplicate document_id: {document_id}")
        seen_ids.add(document_id)

    retrieved_at = datetime.now(timezone.utc).isoformat()
    staged = [_candidate(entry, retrieved_at) for entry in documents]
    total_chars = sum(text_length for _, _, _, _, text_length in staged)
    if total_chars > MAX_TOTAL_CHARS:
        raise ValueError(f"corpus text exceeds {MAX_TOTAL_CHARS} characters")

    sources = [source for _, _, source, _, _ in staged]
    chunks = [chunk for _, _, _, source_chunks, _ in staged for chunk in source_chunks]
    if len({chunk["id"] for chunk in chunks}) != len(chunks):
        raise ValueError("duplicate chunk id")
    if len({source["document_id"] for source in sources}) != len(sources):
        raise ValueError("duplicate source id")

    raw_root = Path(raw_dir)
    for relative_path, raw_bytes, _, _, _ in staged:
        raw_path = raw_root / relative_path
        if raw_path.exists() and raw_path.read_bytes() != raw_bytes:
            raise ValueError(f"immutable raw artifact differs: {raw_path}")
    for relative_path, raw_bytes, _, _, _ in staged:
        _write_immutable(raw_root / relative_path, raw_bytes)

    corpus = {
        "manifest": {
            "schema_version": "1",
            "upstream": {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT},
            "generated_at": retrieved_at,
            "source_count": len(sources),
            "chunk_count": len(chunks),
        },
        "sources": sources,
        "chunks": chunks,
    }
    _replace_corpus(Path(corpus_path), corpus)
    return corpus


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config" / "regulations.json")
    parser.add_argument("--raw-dir", type=Path, default=root / "data" / "regulations" / "raw")
    parser.add_argument("--corpus", type=Path, default=root / "data" / "regulations" / "corpus.json")
    args = parser.parse_args(argv)
    print(json.dumps(fetch_regulations(args.config, args.raw_dir, args.corpus), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
