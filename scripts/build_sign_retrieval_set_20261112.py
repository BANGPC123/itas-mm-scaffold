"""Build the sign-definition retrieval set (A6) from the regulation corpus.

Two labeling passes, each a fixed rule, run separately:
  pass A: the sign code is followed within 60 characters by its Vietnamese name (accent-insensitive);
  pass B: the Vietnamese name occurs with the sign code within the 200 characters before it.
A locator is kept only when both passes find it (intersection). Codes with no agreed locator are
dropped and listed, not filled in by hand. Labels are AI-reviewed rule output, not human ground truth.

Dev/holdout: codes are ordered by sha256(seed + code); the first half is dev, the second holdout.
The split is written before any retriever is run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.perception.sign_catalog import NAMES_VI  # noqa: E402

CORPUS = ROOT / "data/regulations/corpus.json"
OUT_SET = ROOT / "data/evaluation/legal_retrieval_signs_20261112.json"
OUT_STATS = ROOT / "output/bench_20261112/A6/labeling_stats.json"
SEED = "20261112"


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold().replace("đ", "d"))
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def code_pattern(code: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(code) + r"(?![0-9a-z])")


def pass_a(code: str, name: str, chunks: list[dict]) -> set[tuple[str, str]]:
    name_re = r"\s+".join(re.escape(w) for w in fold(name).split())
    found = set()
    for chunk in chunks:
        for match in code_pattern(code).finditer(chunk["text"]):
            if re.search(name_re, fold(chunk["text"][match.end():match.end() + 60])):
                found.add((chunk["document_id"], chunk["locator"]))
                break
    return found


def pass_b(code: str, name: str, chunks: list[dict]) -> set[tuple[str, str]]:
    name_re = r"\s+".join(re.escape(w) for w in fold(name).split())
    code_folded = re.compile(r"(?<![a-z0-9])" + re.escape(fold(code)) + r"(?![0-9a-z])")
    found = set()
    for chunk in chunks:
        text = fold(chunk["text"])
        for match in re.finditer(name_re, text):
            if code_folded.search(text[max(0, match.start() - 200):match.start()]):
                found.add((chunk["document_id"], chunk["locator"]))
                break
    return found


def main() -> None:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    chunks = corpus["chunks"]
    kept, dropped, stats = [], [], {}
    for code in sorted(NAMES_VI):
        name = NAMES_VI[code]
        a = pass_a(code, name, chunks)
        b = pass_b(code, name, chunks)
        agreed = sorted(a & b)
        stats[code] = {"pass_a": sorted(a), "pass_b": sorted(b), "agreed": agreed}
        if agreed:
            kept.append({"code": code, "name": name, "expected": [{"document_id": d, "locator": l} for d, l in agreed]})
        else:
            dropped.append(code)
    ranked = sorted(kept, key=lambda row: hashlib.sha256((SEED + row["code"]).encode()).hexdigest())
    half = len(ranked) // 2
    cases = []
    for index, row in enumerate(ranked):
        split = "dev" if index < half else "holdout"
        cases.append({"id": f"sign-{row['code']}", "category": "sign", "split": split,
                      "code": row["code"], "query": f"{row['code']}: {row['name']}",
                      "expected": row["expected"]})
    cases.sort(key=lambda c: c["id"])
    agree_rate = len(kept) / len(NAMES_VI)
    payload = {
        "description": "Sign-definition retrieval cases built on 2026-11-12. Answers are corpus locators "
                       "agreed by two fixed labeling rules (see scripts/build_sign_retrieval_set_20261112.py). "
                       "AI-reviewed rule labels, not human ground truth.",
        "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "codes_in_catalog": len(NAMES_VI), "cases_kept": len(kept), "codes_dropped": dropped,
        "agreement_rate_over_catalog": agree_rate,
        "split_counts": {s: sum(1 for c in cases if c["split"] == s) for s in ("dev", "holdout")},
        "cases": cases,
    }
    OUT_SET.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    OUT_STATS.write_text(json.dumps({"per_code": stats, "dropped": dropped}, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "cases"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
