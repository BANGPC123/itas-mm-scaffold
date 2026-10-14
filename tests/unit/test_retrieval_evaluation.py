from scripts.evaluate_legal_retrieval import score_case, summarize_scores
from src.reasoning.document_loader import DocumentChunk


def _chunk(document_id: str, locator: str) -> DocumentChunk:
    return DocumentChunk(
        text="evidence",
        source_file="corpus.json",
        chunk_index=0,
        document_id=document_id,
        locator_type="heading",
        locator=locator,
    )


def test_score_case_counts_a_rank_one_exact_metadata_hit():
    score = score_case({("law", "Điều 1")}, [_chunk("law", "Điều 1")], (1, 3, 4))

    assert score == {
        "recall_at_1": 1.0,
        "recall_at_3": 1.0,
        "recall_at_4": 1.0,
        "reciprocal_rank": 1.0,
    }


def test_score_case_counts_a_hit_at_rank_three():
    score = score_case(
        {("law", "Điều 3")},
        [_chunk("law", "Điều 1"), _chunk("law", "Điều 2"), _chunk("law", "Điều 3")],
        (1, 3, 4),
    )

    assert score == {
        "recall_at_1": 0.0,
        "recall_at_3": 1.0,
        "recall_at_4": 1.0,
        "reciprocal_rank": 1 / 3,
    }


def test_score_case_returns_zeroes_for_a_complete_miss():
    score = score_case({("law", "Điều 9")}, [_chunk("law", "Điều 1")], (1, 3, 4))

    assert score == {
        "recall_at_1": 0.0,
        "recall_at_3": 0.0,
        "recall_at_4": 0.0,
        "reciprocal_rank": 0.0,
    }


def test_score_case_accepts_any_of_multiple_expected_locators():
    score = score_case(
        {("law", "Điều 1"), ("decree", "Điều 2")},
        [_chunk("decree", "Điều 2")],
        (1, 3, 4),
    )

    assert score["recall_at_1"] == 1.0
    assert score["reciprocal_rank"] == 1.0


def test_summarize_scores_averages_mrr_and_recall():
    summary = summarize_scores(
        [
            {"recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_4": 1.0, "reciprocal_rank": 1.0},
            {"recall_at_1": 0.0, "recall_at_3": 1.0, "recall_at_4": 1.0, "reciprocal_rank": 1 / 3},
            {"recall_at_1": 0.0, "recall_at_3": 0.0, "recall_at_4": 0.0, "reciprocal_rank": 0.0},
        ],
        (1, 3, 4),
    )

    assert summary == {
        "case_count": 3,
        "recall_at_1": 1 / 3,
        "recall_at_3": 2 / 3,
        "recall_at_4": 2 / 3,
        "mrr": 4 / 9,
    }
