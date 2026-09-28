"""Normalize acquired legal artifacts into reviewable canonical JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from src.reasoning.legal_models import LegalSource
from src.reasoning.legal_normalizer import (
    extract_pdf_text,
    extract_vbpl_text,
    normalize_document,
)


def _replace_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            json.dump(value, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _document_type(entry: dict) -> str:
    if entry.get("source_kind") == "attachment":
        return "qcvn"
    return "decree" if str(entry.get("document_id", "")).startswith("decree-") else "law"


def normalize_regulations(
    config_path: str | Path,
    manifest_path: str | Path,
    *,
    assist=None,
) -> list[Path]:
    """Verify every raw artifact, normalize it, then atomically publish JSON."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    configured = {entry["document_id"]: entry for entry in config.get("documents", [])}
    staged: list[tuple[Path, dict]] = []

    for record in manifest.get("documents", []):
        document_id = record["document_id"]
        entry = configured.get(document_id)
        if entry is None:
            raise ValueError(f"manifest document is not configured: {document_id}")
        raw_file = record["raw_file"]
        raw_path = manifest_file.parent / raw_file
        raw_bytes = raw_path.read_bytes()
        actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if actual_sha256 != record["sha256"]:
            raise ValueError(f"raw SHA-256 mismatch for {document_id}")
        if entry["source_kind"] == "vbpl":
            raw_text = extract_vbpl_text(json.loads(raw_bytes))
        elif entry["source_kind"] == "attachment":
            raw_text = extract_pdf_text(raw_path)
        else:
            raise ValueError(f"unsupported source_kind: {entry['source_kind']}")

        document, unresolved = normalize_document(
            raw_text,
            document_id=document_id,
            title=entry["expected_document_number"],
            document_type=_document_type(entry),
            source=LegalSource(
                source_url=record["source_url"],
                raw_file=raw_file,
                sha256=record["sha256"],
                retrieved_at=record["retrieved_at"],
            ),
            assist=assist,
        )
        if unresolved:
            raise ValueError(f"unresolved blocks for {document_id}: {len(unresolved)}")
        staged.append((manifest_file.parent / record["normalized_file"], document.model_dump(mode="json")))

    for path, document in staged:
        _replace_json(path, document)
    return [path for path, _ in staged]


def _ollama_assist(block: str) -> dict:
    from src.reasoning.ollama_client import OllamaClient
    from src.utils.config import load_config

    client = OllamaClient(load_config("reasoning")["ollama"])
    response = client.generate(
        "Return one canonical legal node as JSON only. Do not infer text absent from the block.",
        f"Unresolved legal block:\n{block}",
    )
    return json.loads(response)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "config" / "legal-corpus.json")
    parser.add_argument("--manifest", type=Path, default=root / "data" / "regulations" / "manifest.json")
    parser.add_argument("--assist-unresolved", action="store_true")
    args = parser.parse_args()
    try:
        outputs = normalize_regulations(
            args.config,
            args.manifest,
            assist=_ollama_assist if args.assist_unresolved else None,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
