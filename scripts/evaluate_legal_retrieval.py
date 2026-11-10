"""Run the fixed legal-retrieval benchmark against the existing local index."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reasoning.document_loader import DocumentChunk, load_regulation_corpus
from src.reasoning.ollama_client import OllamaClient
from src.reasoning.vector_store import VectorStore
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "data/evaluation/legal_retrieval.json"
KS = (1, 3, 4)


def score_case(
    expected: set[tuple[str, str]], retrieved: list[DocumentChunk], ks: tuple[int, ...]
) -> dict:
    """Score one query using only exact document and locator metadata."""
    rank = next(
        (
            index
            for index, chunk in enumerate(retrieved, start=1)
            if (chunk.document_id, chunk.locator) in expected
        ),
        None,
    )
    return {
        **{f"recall_at_{k}": float(rank is not None and rank <= k) for k in ks},
        "reciprocal_rank": 0.0 if rank is None else 1 / rank,
    }


def summarize_scores(case_scores: list[dict], ks: tuple[int, ...]) -> dict:
    """Average benchmark metrics over independently scored queries."""
    count = len(case_scores)
    if not count:
        return {"case_count": 0, **{f"recall_at_{k}": 0.0 for k in ks}, "mrr": 0.0}
    return {
        "case_count": count,
        **{
            f"recall_at_{k}": sum(score[f"recall_at_{k}"] for score in case_scores)
            / count
            for k in ks
        },
        "mrr": sum(score["reciprocal_rank"] for score in case_scores) / count,
    }


def _load_cases() -> list[dict]:
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    cases = benchmark.get("cases") if isinstance(benchmark, dict) else None
    if not isinstance(cases, list) or len(cases) != 24:
        raise ValueError("legal retrieval benchmark requires exactly 24 cases")
    categories = {
        "speed",
        "signals",
        "prohibitory_signs",
        "lanes",
        "stopping_parking",
        "overtaking",
        "penalties",
        "qcvn_sign_meaning",
    }
    if {case.get("category") for case in cases} != categories or any(
        sum(case["category"] == category for case in cases) != 3
        for category in categories
    ):
        raise ValueError("legal retrieval benchmark requires three cases per category")
    if len({case.get("id") for case in cases}) != len(cases):
        raise ValueError("legal retrieval benchmark case ids must be unique")
    return cases


def _expected_targets(case: dict) -> set[tuple[str, str]]:
    targets = case.get("expected")
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"case {case.get('id')!r} requires expected targets")
    try:
        return {(target["document_id"], target["locator"]) for target in targets}
    except (KeyError, TypeError) as exc:
        raise ValueError(f"case {case.get('id')!r} has an invalid expected target") from exc


def main() -> None:
    reasoning_cfg = load_config("reasoning")
    corpus_path = ROOT / reasoning_cfg["rag"]["corpus_file"]
    corpus = load_regulation_corpus(corpus_path)
    available_targets = {
        (chunk["document_id"], chunk["locator"]) for chunk in corpus["chunks"]
    }
    cases = _load_cases()
    expected_by_case = {case["id"]: _expected_targets(case) for case in cases}
    invalid = {
        case_id: targets - available_targets
        for case_id, targets in expected_by_case.items()
        if targets - available_targets
    }
    if invalid:
        raise ValueError(f"benchmark targets absent from corpus: {invalid}")

    store = VectorStore(
        reasoning_cfg["rag"],
        OllamaClient(reasoning_cfg["ollama"]),
        create_if_missing=False,
    )
    scores: list[dict] = []
    failures: list[tuple[dict, list[DocumentChunk]]] = []
    for case in cases:
        retrieved = store.query(case["query"], top_k=max(KS))
        score = score_case(expected_by_case[case["id"]], retrieved, KS)
        scores.append(score)
        if not score[f"recall_at_{max(KS)}"]:
            failures.append((case, retrieved))

    summary = summarize_scores(scores, KS)
    metadata = store.get_index_metadata()
    print(f"case_count={summary['case_count']}")
    for k in KS:
        print(f"Recall@{k}={summary[f'recall_at_{k}']:.6f}")
    print(f"MRR={summary['mrr']:.6f}")
    print(f"corpus_fingerprint={metadata.get('corpus_fingerprint')}")
    print(f"schema_version={metadata.get('schema_version')}")
    print(f"embedding_model={metadata.get('embedding_model')}")
    print(f"evaluated_top_k={max(KS)}")
    for case, retrieved in failures:
        evidence = [
            {"document_id": chunk.document_id, "locator": chunk.locator}
            for chunk in retrieved
        ]
        print(
            f"failed_case={case['id']} "
            f"retrieved={json.dumps(evidence, ensure_ascii=False)}"
        )


if __name__ == "__main__":
    main()
