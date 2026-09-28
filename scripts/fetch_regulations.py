"""Fetch the curated official legal-source artifacts without overwriting them."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import requests


VBPL_API = "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{}"
REQUEST_TIMEOUT_SECONDS = 30


def _required(entry: dict, field: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"source entry requires {field}")
    return value


def _fetch_vbpl(entry: dict) -> tuple[bytes, str]:
    item_id = _required(entry, "item_id")
    response = requests.get(VBPL_API.format(item_id), timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    raw_bytes = response.content
    try:
        payload = json.loads(raw_bytes)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"VBPL item {item_id} did not return JSON") from exc
    expected = _required(entry, "expected_document_number")
    if payload.get("docNum") != expected:
        raise ValueError(
            f"VBPL document number mismatch for {item_id}: "
            f"expected {expected!r}, got {payload.get('docNum')!r}"
        )
    return raw_bytes, "json"


def _fetch_attachment(entry: dict) -> tuple[bytes, str]:
    response = requests.get(_required(entry, "attachment_url"), timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    if not response.content:
        raise ValueError(f"attachment for {_required(entry, 'document_id')} is empty")
    return response.content, "pdf"


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


def _replace_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            json.dump(manifest, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def fetch_regulations(config_path: str, raw_dir: str, manifest_path: str) -> dict:
    """Acquire all configured sources and atomically publish their manifest."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    documents = config.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("config requires a non-empty documents list")

    staged: list[tuple[Path, bytes, dict]] = []
    seen_ids: set[str] = set()
    for entry in documents:
        if not isinstance(entry, dict):
            raise ValueError("each source entry must be an object")
        document_id = _required(entry, "document_id")
        if document_id in seen_ids:
            raise ValueError(f"duplicate document_id: {document_id}")
        seen_ids.add(document_id)
        source_kind = _required(entry, "source_kind")
        _required(entry, "source_url")
        if source_kind == "vbpl":
            raw_bytes, extension = _fetch_vbpl(entry)
        elif source_kind == "attachment":
            raw_bytes, extension = _fetch_attachment(entry)
        else:
            raise ValueError(f"unsupported source_kind: {source_kind}")

        raw_file = f"raw/{document_id}/source.{extension}"
        staged.append(
            (
                Path(raw_dir) / document_id / f"source.{extension}",
                raw_bytes,
                {
                    "document_id": document_id,
                    "source_url": entry["source_url"],
                    "raw_file": raw_file,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "normalized_file": f"normalized/{document_id}.json",
                    "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                },
            )
        )

    for raw_path, raw_bytes, _ in staged:
        if raw_path.exists() and raw_path.read_bytes() != raw_bytes:
            raise ValueError(f"immutable raw artifact differs: {raw_path}")
    for raw_path, raw_bytes, _ in staged:
        _write_immutable(raw_path, raw_bytes)

    manifest = {"documents": [entry for _, _, entry in staged]}
    _replace_manifest(Path(manifest_path), manifest)
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = fetch_regulations(
        str(root / "config" / "legal-corpus.json"),
        str(root / "data" / "regulations" / "raw"),
        str(root / "data" / "regulations" / "manifest.json"),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
