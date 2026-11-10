import pytest

from src.reasoning.document_loader import chunk_text


def test_chunk_text_splits_long_text_into_multiple_chunks():
    text = "A" * 2000
    chunks = chunk_text(text, "doc.md", chunk_size_chars=800, chunk_overlap_chars=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 800 for c in chunks)


def test_chunk_text_short_text_produces_one_chunk():
    text = "This is a short regulation snippet."
    chunks = chunk_text(text, "doc.md", chunk_size_chars=800, chunk_overlap_chars=100)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunk_text_empty_string_produces_no_chunks():
    chunks = chunk_text("", "doc.md", chunk_size_chars=800, chunk_overlap_chars=100)
    assert chunks == []


def test_chunk_text_whitespace_only_produces_no_chunks():
    chunks = chunk_text("   \n\n   ", "doc.md", chunk_size_chars=800, chunk_overlap_chars=100)
    assert chunks == []


def test_chunk_text_rejects_overlap_larger_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", "doc.md", chunk_size_chars=100, chunk_overlap_chars=200)


def test_chunk_indices_are_sequential_per_document():
    text = "B" * 2000
    chunks = chunk_text(text, "doc.md", chunk_size_chars=800, chunk_overlap_chars=100)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
