"""Optional sign-aware variants of the offline BM25 legal index (benchmark 2026-11-12).

Every variant is OFF by default. The application keeps using LexicalLegalIndex unchanged.

  code_boost (R1): a result whose excerpt contains the sign code written in the query gets a bonus
                   of ``boost * top_score`` before the final top-k cut (re-ranking of the top 20).
  definition_chunks (R2): every mention of a sign code in the corpus also becomes an extra, shorter
                   chunk (the code plus the following 300 characters) with the same document_id and
                   locator, so that a focused definition can outrank a long article.
  R3 = both.
"""
from __future__ import annotations

import re

from src.reasoning.lexical_retrieval import LexicalLegalIndex, tokens

CODE_RE = re.compile(r"(?<![A-Za-z0-9])(DP|P|R|W|I)\.\d{1,3}[a-z]?(?![0-9a-z])")
DEFINITION_WINDOW = 300


def codes_in_text(text: str) -> list[str]:
    return [m.group(0) for m in CODE_RE.finditer(text)]


class SignAwareLegalIndex(LexicalLegalIndex):
    def __init__(self, corpus_path, *, code_boost: bool = False, boost: float = 0.5,
                 definition_chunks: bool = False):
        super().__init__(corpus_path)
        self.code_boost = bool(code_boost)
        self.boost = float(boost)
        self.definition_chunks = bool(definition_chunks)
        self.extra_chunk_count = 0
        if self.definition_chunks:
            self._add_definition_chunks()

    def _append_chunk(self, chunk: dict) -> None:
        index = len(self.chunks)
        self.chunks.append(chunk)
        counts = {}
        for token in tokens(chunk["text"]):
            counts[token] = counts.get(token, 0) + 1
        self.lengths.append(sum(counts.values()))
        for token, count in counts.items():
            self.postings[token].append((index, count))
        ts = tokens(chunk["text"])
        for width in (2, 3):
            phrase_counts = {}
            for j in range(len(ts) - width + 1):
                key = tuple(ts[j:j + width])
                phrase_counts[key] = phrase_counts.get(key, 0) + 1
            for phrase, count in phrase_counts.items():
                self.phrases[phrase].append((index, count))
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))

    def _add_definition_chunks(self) -> None:
        base = list(self.chunks)
        for chunk in base:
            text = chunk["text"]
            for match in CODE_RE.finditer(text):
                window = text[match.start():match.end() + DEFINITION_WINDOW]
                self._append_chunk({**chunk, "id": f"{chunk['id']}#def{match.start()}",
                                    "text": window, "definition_of": match.group(0)})
                self.extra_chunk_count += 1

    def search(self, query, top_k=4, *, document=None, unique_locators=True):
        if not self.code_boost:
            return super().search(query, top_k, document=document, unique_locators=unique_locators)
        query_codes = set(codes_in_text(query)) if isinstance(query, str) else set()
        pool = super().search(query, 20, document=document, unique_locators=unique_locators)
        if not pool or not query_codes:
            return pool[:top_k]
        top_score = max(hit["score"] for hit in pool)
        for hit in pool:
            if any(re.search(r"(?<![A-Za-z0-9])" + re.escape(code) + r"(?![0-9a-z])", hit["excerpt"])
                   for code in query_codes):
                hit["score"] = hit["score"] + self.boost * top_score
        pool.sort(key=lambda hit: -hit["score"])
        return pool[:top_k]
