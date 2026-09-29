from src.reasoning.document_loader import (
    chunk_regulation_text,
    split_long_paragraph,
    stable_checksum,
)


def test_stable_checksum_matches_pinned_luatrag_behavior():
    assert stable_checksum("abc") == "7aigaz"


def test_chunking_is_stable_for_same_text():
    text = (
        "Điều 1. Phạm vi điều chỉnh\n"
        "Người điều khiển phương tiện phải tuân thủ quy định này. "
        "Việc tuân thủ bảo đảm an toàn giao thông.\n\n"
        "Nội dung không có tiêu đề vẫn phải được định vị ổn định."
    )

    first = chunk_regulation_text("law-1", text, target_size=70, max_size=100)

    assert chunk_regulation_text("law-1", text, target_size=70, max_size=100) == first
    assert first[0]["id"] == f"law-1:0:{first[0]['checksum']}"
    assert first[0]["locator_type"] == "heading"
    assert first[0]["locator"] == "Điều 1. Phạm vi điều chỉnh"
    assert first[0]["heading"] == "Điều 1. Phạm vi điều chỉnh"
    assert all(chunk["checksum"] for chunk in first)

    unheaded = chunk_regulation_text(
        "law-2", "Nội dung không có tiêu đề vẫn phải được định vị ổn định."
    )

    assert unheaded[0]["locator_type"] == "chunk"
    assert unheaded[0]["locator"] == "Đoạn 1"


def test_split_long_paragraph_prefers_sentence_boundaries():
    assert split_long_paragraph("One. Two sentence.", 10) == ["One.", "Two senten", "ce."]
