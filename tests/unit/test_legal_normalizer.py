from __future__ import annotations

import pytest

from src.reasoning.legal_normalizer import (
    extract_official_article_text,
    extract_pdf_text,
    extract_vbpl_text,
    html_to_text,
    normalize_vietnamese,
)


def test_normalize_vietnamese_matches_pinned_luatrag_behavior():
    assert normalize_vietnamese("\0Đie\u0302\u0300u\r\n1\t  \n\n\nNội dung") == "Điều\n1 \n\nNội dung"


def test_html_to_text_preserves_block_boundaries_and_removes_hidden_markup():
    value = "<head>ignore</head><p>First<br>Next</p><li>One</li><script>bad()</script><style>x{}</style>"

    assert html_to_text(value) == "First\nNext\n• One"


def test_vbpl_extracts_data_document_content():
    payload = {"data": {"documentContent": {"content": "<p>Điều 1. <b>Quy định</b></p>"}}}

    assert extract_vbpl_text(payload) == "Điều 1. Quy định"


def test_official_html_extracts_only_article_body():
    page_html = (
        "<header>Navigation</header><main><div itemprop=\"articleBody\">"
        "<p>Điều 1. Nội dung chính</p><script>ignore()</script></div></main><footer>Footer</footer>"
    )

    assert extract_official_article_text(page_html) == "Điều 1. Nội dung chính"


def test_pdf_skips_fontless_image_pages(monkeypatch, tmp_path):
    class ImagePage:
        def get(self, key, default=None):
            return {"/Resources": {"/XObject": {}}}.get(key, default)

        def extract_text(self):
            raise AssertionError("fontless pages must be skipped")

    class TextPage:
        def get(self, key, default=None):
            return {"/Resources": {"/Font": {}}}.get(key, default)

        def extract_text(self):
            return "Điều 2. Text layer from the PDF"

    class Reader:
        pages = [ImagePage(), TextPage()]

    monkeypatch.setattr("src.reasoning.legal_normalizer.PdfReader", lambda _: Reader())

    assert extract_pdf_text(tmp_path / "mixed.pdf") == "Điều 2. Text layer from the PDF"


def test_pdf_without_usable_text_fails_closed(monkeypatch, tmp_path):
    class ImagePage:
        def get(self, key, default=None):
            return {"/Resources": {"/XObject": {}}}.get(key, default)

        def extract_text(self):
            raise AssertionError("fontless pages must be skipped")

    class Reader:
        pages = [ImagePage()]

    monkeypatch.setattr("src.reasoning.legal_normalizer.PdfReader", lambda _: Reader())

    with pytest.raises(ValueError, match="usable text"):
        extract_pdf_text(tmp_path / "scanned.pdf")
