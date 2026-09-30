from __future__ import annotations

import copy

from src.reasoning.document_loader import compute_corpus_fingerprint


from tests.unit.test_document_loader import _corpus


def test_fingerprint_stable_for_same_semantic_corpus():
    corpus = _corpus()

    first = compute_corpus_fingerprint(corpus, "1", "nomic-embed-text")
    second = compute_corpus_fingerprint(corpus, "1", "nomic-embed-text")

    assert first == second
    assert first == compute_corpus_fingerprint(copy.deepcopy(corpus), "1", "nomic-embed-text")


def test_fingerprint_ignores_volatile_corpus_metadata():
    corpus = _corpus()
    relocated = copy.deepcopy(corpus)
    relocated["manifest"]["generated_at"] = "2026-10-01T00:00:00+00:00"
    relocated["sources"][0]["retrieved_at"] = "2026-10-01T00:00:00+00:00"
    relocated["sources"][0]["raw_file"] = "C:/different-machine/source.json"

    assert compute_corpus_fingerprint(corpus, "1", "nomic-embed-text") == compute_corpus_fingerprint(
        relocated, "1", "nomic-embed-text"
    )


def test_fingerprint_changes_for_semantic_corpus_or_index_inputs():
    corpus = _corpus()
    fingerprint = compute_corpus_fingerprint(corpus, "1", "nomic-embed-text")

    changed_source_sha = copy.deepcopy(corpus)
    changed_source_sha["sources"][0]["sha256"] = "c" * 64
    changed_text_sha = copy.deepcopy(corpus)
    changed_text_sha["sources"][0]["text_sha256"] = "d" * 64
    changed_text = copy.deepcopy(corpus)
    changed_text["chunks"][0]["text"] = "Changed legal chunk."
    changed_locator = copy.deepcopy(corpus)
    changed_locator["chunks"][0]["locator"] = "Điều 8"

    for changed in (
        changed_source_sha,
        changed_text_sha,
        changed_text,
        changed_locator,
    ):
        assert fingerprint != compute_corpus_fingerprint(changed, "1", "nomic-embed-text")
    assert fingerprint != compute_corpus_fingerprint(corpus, "2", "nomic-embed-text")
    assert fingerprint != compute_corpus_fingerprint(corpus, "1", "other-embedding")
