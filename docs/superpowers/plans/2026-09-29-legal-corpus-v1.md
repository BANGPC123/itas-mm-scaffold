# ITAS-MM Legal Corpus v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the illustrative traffic-law sample with a reproducible, current Vietnamese legal corpus whose canonical JSON, provenance, retrieval metadata, citations, and retrieval quality are testable end to end.

**Architecture:** Acquire three curated official sources into immutable raw artifacts, normalize them into validated Pydantic canonical JSON, project legal leaf nodes to `DocumentChunk`, and rebuild the existing Chroma index deterministically. Keep `nomic-embed-text` + Chroma, retain retrieved evidence through `RagChain`, and evaluate retrieval separately with Recall@k/MRR.

**Tech Stack:** Python 3.12, Pydantic 2.x, requests, pypdf 6.x, ChromaDB, Ollama `nomic-embed-text` + `llama3.1`, pytest, unittest.mock.

**Spec:** `docs/superpowers/specs/2026-09-28-legal-corpus-v1-design.md`

## Global Constraints

- `main` is read-only: never edit, commit, merge, rebase, force-update, or push it.
- All implementation stays on `backend/legal-corpus-v1` or another explicitly permitted non-`main` branch.
- Reference `lqb464/LuatRAG` only at pinned commit `ae2b1c796503e2a58493771bc341b66fb488e053`; do not vendor it wholesale.
- Do not copy LuatRAG's bootstrap corpus; acquire ITAS traffic-law sources from official endpoints.
- Corpus v1 sources are `36/2024/QH15`, `168/2024/NĐ-CP`, and `QCVN 41:2024/BGTVT`.
- QCVN 41:2019/BGTVT is not used for current guidance.
- Keep existing Chroma + `nomic-embed-text`; no BM25/FTS5/reranker unless later benchmark evidence justifies it.
- No legal ontology, knowledge graph, crawler daemon, UI citation viewer, or public API widening.
- Normalized JSON is reviewable source of truth; Chroma is derived and rebuildable.
- Runtime empty retrieval keeps the existing Vietnamese fail-safe and must not invoke generation.

## Review Focus

1. Source identity mismatch: fetched VBPL payload number must equal the curated expected number; otherwise acquisition fails without overwriting the prior manifest.
2. QCVN PDF without a usable text layer: normalization must fail explicitly rather than silently produce an empty/garbled canonical document.
3. Duplicate or empty legal nodes: schema validation rejects duplicate article/clause/point/section locators and empty leaves before indexing.
4. Stale derived index: a successful rebuild must remove chunks that are absent from the new canonical corpus, while a pre-build validation failure must leave the old collection usable.
5. Citation escape: any source alias emitted by the LLM must refer only to evidence in the current retrieval set; unknown aliases are rejected.

## File Structure

- `config/legal-corpus.json` — curated source identities and official acquisition locators.
- `src/reasoning/legal_models.py` — canonical Pydantic schema and structural validation.
- `src/reasoning/legal_normalizer.py` — extraction/normalization into canonical models.
- `src/reasoning/document_loader.py` — canonical JSON loading and `DocumentChunk` projection.
- `src/reasoning/vector_store.py` — deterministic derived-index rebuild and metadata round-trip.
- `src/reasoning/rag_chain.py` — evidence labels and citation-alias validation.
- `scripts/fetch_regulations.py` — official source acquisition and raw manifest creation.
- `scripts/normalize_regulations.py` — raw → reviewed canonical JSON workflow.
- `scripts/build_vector_index.py` — validate/project/fingerprint/rebuild.
- `scripts/evaluate_legal_retrieval.py` — Recall@1/@3/@4, MRR, failure reporting.
- `data/regulations/raw/`, `data/regulations/normalized/`, `data/regulations/manifest.json` — committed corpus artifacts.
- `data/evaluation/legal_retrieval.json` — curated retrieval benchmark cases.

---
### Task 1: Canonical legal schema and validation

**Files:**
- Create: `src/reasoning/legal_models.py`
- Create: `tests/unit/test_legal_models.py`

**Interfaces:**
- Produces: `SCHEMA_VERSION = "1"`.
- Produces: `LegalSource`, `LegalPoint`, `LegalClause`, `LegalArticle`, `LegalSection`, `LegalDocument` Pydantic models.
- `LegalDocument` fields: `schema_version`, `document_id`, `title`, `document_type`, `source`, `articles`, `sections`.
- `LegalSource` fields: `source_url`, `raw_file`, `sha256`, `retrieved_at`.

- [ ] **Step 1: Write failing schema tests**

Add tests named `test_valid_law_hierarchy`, `test_valid_qcvn_section_hierarchy`, `test_duplicate_articles_rejected`, `test_duplicate_clauses_and_points_rejected`, `test_duplicate_sections_rejected`, `test_empty_leaf_rejected`, and `test_missing_source_provenance_rejected`. Assert exact Pydantic validation failures rather than private validator calls.

- [ ] **Step 2: Verify RED**

Run: `conda run -n WTF python -m pytest tests/unit/test_legal_models.py -v`
Expected: FAIL because `src.reasoning.legal_models` does not exist.

- [ ] **Step 3: Implement minimal models and validators**

Use `BaseModel`, `Field`, and model validators only. A legal leaf must contain non-whitespace text; parent nodes may omit `text` only when they have children. Enforce uniqueness within each parent scope, not globally across unrelated documents.

- [ ] **Step 4: Verify GREEN**

Run: `conda run -n WTF python -m pytest tests/unit/test_legal_models.py -v`
Expected: all Task 1 tests PASS.

- [ ] **Step 5: Run regression suite**

Run: `conda run -n WTF python -m pytest -q`
Expected: current suite plus schema tests PASS.

- [ ] **Step 6: Commit Task 1**

```powershell
git add src/reasoning/legal_models.py tests/unit/test_legal_models.py
git commit -m "feat: add canonical legal schema"
```

### Task 2: Curated official-source acquisition

**Files:**
- Create: `config/legal-corpus.json`
- Create: `scripts/fetch_regulations.py`
- Create: `tests/unit/test_fetch_regulations.py`

**Interfaces:**
- Produces: `fetch_regulations(config_path: str, raw_dir: str, manifest_path: str) -> dict`.
- Produces immutable raw artifacts plus `data/regulations/manifest.json` entries containing `document_id`, source URL, raw file, retrieval timestamp, normalized file, and SHA-256.

- [ ] **Step 1: Write failing acquisition tests**

Test `test_vbpl_number_mismatch_fails_closed`, `test_same_raw_bytes_produce_same_sha256_and_paths`, `test_existing_raw_file_with_different_bytes_is_not_overwritten`, `test_failed_fetch_does_not_replace_existing_manifest`, and `test_qcvn_download_uses_curated_official_attachment`. Mock network responses; do not hit the internet in unit tests.

- [ ] **Step 2: Verify RED**

Run: `conda run -n WTF python -m pytest tests/unit/test_fetch_regulations.py -v`
Expected: FAIL because the fetcher/config do not exist.

- [ ] **Step 3: Add the curated three-source config**

Pin these identities:
- `law-36-2024-qh15`: VBPL item `170620`, expected `36/2024/QH15`, source page `https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=170620`.
- `decree-168-2024-nd-cp`: VBPL item `173920`, expected `168/2024/NĐ-CP`, source page `https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=173920`.
- `qcvn-41-2024-bgtvt`: government page `https://vanban.chinhphu.vn/?classid=1&docid=211908&pageid=27160`, attachment `https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf`, expected `QCVN 41:2024/BGTVT`.

- [ ] **Step 4: Implement acquisition**

For VBPL entries call `https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{item_id}`, verify `docNum`, and store the complete JSON payload as `source.json`. For QCVN store the exact attachment bytes as `source.pdf`. Compute SHA-256 from raw bytes. An existing raw file is immutable: identical bytes are a no-op, different bytes fail closed instead of overwriting it. Replace the manifest only after all configured sources have been acquired and verified; use a temp file + `os.replace`.

- [ ] **Step 5: Verify GREEN and regression safety**

Run: `conda run -n WTF python -m pytest tests/unit/test_fetch_regulations.py -v`
Then: `conda run -n WTF python -m pytest -q`
Expected: acquisition tests and full suite PASS.

- [ ] **Step 6: Commit Task 2**

```powershell
git add config/legal-corpus.json scripts/fetch_regulations.py tests/unit/test_fetch_regulations.py
git commit -m "feat: add official legal source acquisition"
```

### Task 3: Raw extraction and hybrid canonical normalization

**Files:**
- Modify: `requirements.txt`
- Create: `src/reasoning/legal_normalizer.py`
- Create: `scripts/normalize_regulations.py`
- Create: `tests/unit/test_legal_normalizer.py`

**Interfaces:**
- Produces: `extract_vbpl_text(payload: dict) -> str`.
- Produces: `extract_pdf_text(path: str | Path) -> str` using pypdf.
- Produces: `normalize_document(raw_text: str, *, document_id: str, title: str, document_type: str, source: LegalSource, assist: Callable[[str], dict] | None = None) -> tuple[LegalDocument, list[str]]`.
- Deterministic parser handles law/decree `Điều → Khoản → Điểm` and QCVN hierarchical section identifiers; `assist` is invoked only for unresolved blocks.

- [ ] **Step 1: Write failing normalization tests**

Add `test_vbpl_html_normalizes_unicode_and_structure`, `test_law_parser_preserves_article_clause_point`, `test_qcvn_parser_builds_nested_sections`, `test_unresolved_block_is_reported_without_assist`, `test_assist_is_called_only_for_unresolved_block`, `test_invalid_assist_payload_is_rejected`, and `test_pdf_without_usable_text_fails_explicitly`.

- [ ] **Step 2: Verify RED**

Run: `conda run -n WTF python -m pytest tests/unit/test_legal_normalizer.py -v`
Expected: FAIL because normalizer functions do not exist.

- [ ] **Step 3: Add PDF extraction dependency and minimal normalizer**

Add `pypdf>=6.19,<7` to `requirements.txt`. Normalize Unicode to NFC, strip HTML/scripts/styles for VBPL content, and use `PdfReader` for QCVN text-layer extraction. Do not add OCR; fail with a clear error when extracted text is empty/implausibly short.

- [ ] **Step 4: Implement structure parsing and optional block assistance**

Parse explicit headings deterministically. Return unresolved blocks rather than guessing. If `assist` is supplied, call it one unresolved block at a time and validate its candidate node with the Task 1 models before merging; a schema-valid candidate is still written only to reviewable normalized JSON.

- [ ] **Step 5: Implement normalization CLI**

`scripts/normalize_regulations.py` reads both `config/legal-corpus.json` and `manifest.json`, verifies each raw SHA-256 before extraction, writes one normalized JSON file per configured document via temp file + `os.replace`, and exits non-zero if unresolved blocks remain without `--assist-unresolved`. With the flag, adapt existing `OllamaClient.generate()` to return candidate JSON; do not send an entire source document to the LLM.

- [ ] **Step 6: Verify and commit Task 3**

Run: `conda run -n WTF python -m pytest tests/unit/test_legal_normalizer.py tests/unit/test_legal_models.py -v`
Then: `conda run -n WTF python -m pip check` and `conda run -n WTF python -m pytest -q`.
Expected: all PASS.

```powershell
git add requirements.txt src/reasoning/legal_normalizer.py scripts/normalize_regulations.py tests/unit/test_legal_normalizer.py
git commit -m "feat: normalize legal sources to canonical json"
```

### Task 4: Canonical loading and legal-node projection

**Files:**
- Modify: `src/reasoning/document_loader.py`
- Replace: `tests/unit/test_document_loader.py`

**Interfaces:**
- Extends `DocumentChunk` with `document_id: str`, `locator_type: str`, `locator: str`.
- Produces: `load_canonical_documents(normalized_dir: str) -> list[LegalDocument]`.
- Produces: `build_legal_chunks(document: LegalDocument, max_chunk_chars: int | None = None) -> list[DocumentChunk]`.
- Produces: `build_corpus_chunks(normalized_dir: str, max_chunk_chars: int | None = None) -> list[DocumentChunk]`.

- [ ] **Step 1: Replace fixed-window tests with failing legal projection tests**

Test `test_point_becomes_chunk_with_exact_locator`, `test_clause_without_points_becomes_chunk`, `test_article_without_clauses_becomes_chunk`, `test_qcvn_leaf_section_becomes_chunk`, `test_projection_order_is_deterministic`, `test_long_leaf_split_preserves_locator_when_limit_is_explicit`, and `test_invalid_canonical_json_fails_before_projection`.

- [ ] **Step 2: Verify RED**

Run: `conda run -n WTF python -m pytest tests/unit/test_document_loader.py -v`
Expected: FAIL because current loader expects Markdown/text and old `DocumentChunk` shape.

- [ ] **Step 3: Implement canonical loader/projection**

Remove `chunk_text()` as the production corpus API. Use structure-first leaves only. Construct human-readable locators such as `Điều 6 Khoản 1 Điểm a` and QCVN section locators from canonical IDs. `chunk_index` is deterministic within each legal locator. When `max_chunk_chars` is explicitly provided, split only the oversized leaf and preserve the exact locator on every subchunk; the production default stays `None` until Task 7 inspects the real corpus. Do not add arbitrary fixed overlap.

- [ ] **Step 4: Verify GREEN and regression safety**

Run: `conda run -n WTF python -m pytest tests/unit/test_document_loader.py -v`
Then: `conda run -n WTF python -m pytest -q`
Expected: all PASS after updating existing tests/callers to the new `DocumentChunk` fields.

- [ ] **Step 5: Commit Task 4**

```powershell
git add src/reasoning/document_loader.py tests/unit/test_document_loader.py
git commit -m "refactor: project canonical legal evidence"
```

### Task 5: Deterministic fingerprint and Chroma rebuild

**Files:**
- Modify: `src/reasoning/document_loader.py`
- Modify: `src/reasoning/vector_store.py`
- Modify: `scripts/build_vector_index.py`
- Modify: `tests/unit/test_vector_store.py`
- Create: `tests/unit/test_corpus_fingerprint.py`
- Create: `tests/unit/test_build_vector_index.py`

**Interfaces:**
- Produces: `compute_corpus_fingerprint(documents: list[LegalDocument], schema_version: str, embedding_model: str) -> str`.
- Produces: `VectorStore.rebuild(chunks: list[DocumentChunk], *, fingerprint: str, schema_version: str, embedding_model: str) -> None`.

- [ ] **Step 1: Write failing fingerprint/rebuild tests**

Test `test_fingerprint_stable_for_same_semantic_documents`, `test_fingerprint_changes_with_schema_or_embedding_model`, `test_fingerprint_ignores_retrieved_at_and_local_raw_path`, `test_rebuild_uses_deterministic_chunk_ids`, `test_query_round_trips_full_legal_metadata`, `test_embed_failure_keeps_existing_collection`, `test_successful_rebuild_removes_stale_chunks`, and `test_invalid_corpus_does_not_call_rebuild`.

- [ ] **Step 2: Verify RED**

Run: `conda run -n WTF python -m pytest tests/unit/test_corpus_fingerprint.py tests/unit/test_vector_store.py tests/unit/test_build_vector_index.py -v`
Expected: FAIL on missing fingerprint/rebuild behavior and new metadata fields.

- [ ] **Step 3: Implement canonical fingerprint**

Serialize semantic model dumps with sorted keys and compact separators. Exclude retrieval timestamps and local raw-file paths, include source identity/content hash, then hash `schema_version + embedding_model + canonical_documents` with SHA-256.

- [ ] **Step 4: Implement safe full rebuild**

Embed every candidate chunk before deleting the current collection. Chunk IDs are `document_id::locator_type::locator::chunk_index`. Only after all embeddings succeed, replace the collection and upsert all candidate records with `document_id`, `source_file`, `chunk_index`, `locator_type`, and `locator`; store fingerprint/schema/model in collection metadata.

- [ ] **Step 5: Update build script**

`scripts/build_vector_index.py` loads/validates all canonical documents before constructing/calling `VectorStore.rebuild`, refuses an empty corpus, computes fingerprint, checks Ollama embedding readiness, then rebuilds and prints document count, chunk count, fingerprint, schema version, and embedding model. `test_invalid_corpus_does_not_call_rebuild` monkeypatches canonical loading to fail and asserts rebuild is never called.

- [ ] **Step 6: Verify and commit Task 5**

Run targeted tests, then `conda run -n WTF python -m pytest -q` and `git diff --check`. All must pass/clean before commit.

```powershell
git add src/reasoning/document_loader.py src/reasoning/vector_store.py scripts/build_vector_index.py tests/unit/test_vector_store.py tests/unit/test_corpus_fingerprint.py tests/unit/test_build_vector_index.py
git commit -m "feat: rebuild versioned legal index"
```

### Task 6: Grounding labels and fail-closed citation aliases

**Files:**
- Modify: `src/reasoning/rag_chain.py`
- Modify: `tests/unit/test_rag_chain.py`

**Interfaces:**
- Produces: `format_evidence_label(chunk: DocumentChunk, alias: str) -> str`.
- Produces: `validate_citation_aliases(text: str, allowed_aliases: set[str]) -> None`.
- `RagChain.answer()` keeps the same public signature and `ReasoningResult` contract.

- [ ] **Step 1: Write failing grounding tests**

Add `test_prompt_labels_each_chunk_with_deterministic_source_locator`, `test_reasoning_retains_exact_retrieved_evidence`, `test_unknown_generated_source_alias_is_rejected`, `test_known_generated_source_alias_is_allowed`, and keep the existing empty-retrieval/no-generation test.

- [ ] **Step 2: Verify RED**

Run: `conda run -n WTF python -m pytest tests/unit/test_rag_chain.py -v`
Expected: source-label and citation-alias tests FAIL.

- [ ] **Step 3: Implement minimal grounding changes**

Render each retrieved chunk as `[S<n>] <document_id> — <locator>\n<text>` and tell the LLM those aliases are data labels, not authority it may invent. Scan generated text only for bracketed aliases matching `\[S\d+\]`; if any referenced alias is outside the current retrieval set, raise `ValueError("Generated guidance cited evidence outside retrieval set")`. The application still treats `retrieved_chunks` as citation authority; generated prose is not parsed into new legal metadata.

- [ ] **Step 4: Verify GREEN and regression safety**

Run: `conda run -n WTF python -m pytest tests/unit/test_rag_chain.py -v`
Then: `conda run -n WTF python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit Task 6**

```powershell
git add src/reasoning/rag_chain.py tests/unit/test_rag_chain.py
git commit -m "feat: ground guidance in legal source aliases"
```

### Task 7: Acquire, normalize, and migrate the real three-document corpus

**Files:**
- Create/populate: `data/regulations/raw/**`
- Create/populate: `data/regulations/normalized/*.json`
- Create: `data/regulations/manifest.json`
- Delete: `data/regulations/sample_traffic_rules.md`
- Modify: `configs/reasoning.yaml`
- Modify: `README.md`

**Interfaces:**
- `rag.normalized_dir` replaces the old arbitrary Markdown corpus path for index building.
- Remove global `chunk_size_chars` and `chunk_overlap_chars`; structure is the primary chunk boundary.

- [ ] **Step 1: Acquire the three official raw sources**

Run: `conda run -n WTF python scripts/fetch_regulations.py`.
Expected: three verified raw artifacts and a manifest; no sample fallback. Confirm manifest SHA-256 values match the bytes on disk and the VBPL document numbers are exactly the curated expectations.

- [ ] **Step 2: Normalize to canonical JSON**

Run deterministic normalization first. If unresolved blocks remain, rerun only those blocks with `--assist-unresolved`; review every assisted node against the corresponding raw source before accepting it. Do not auto-correct legal wording from model knowledge.

- [ ] **Step 3: Inspect corpus structure before any size ceiling**

Report per-document article/section counts, leaf count, minimum/median/p95/maximum leaf text length, and unresolved-block count. If actual leaves fit the embedding path, do not add a size splitter. If a safety ceiling is demonstrably needed, add it to `build_legal_chunks(..., max_chunk_chars=...)`, test that subchunks preserve the exact legal locator, and document the measured reason.

- [ ] **Step 4: Remove illustrative corpus semantics and align config/docs**

Delete `sample_traffic_rules.md`; set `rag.normalized_dir: "data/regulations/normalized"`; remove `chunk_size_chars` and `chunk_overlap_chars`. Update README with the three-step workflow `fetch_regulations.py → normalize_regulations.py → build_vector_index.py`, current three-source scope, provenance limitations, and no-OCR limitation.

- [ ] **Step 5: Build the real derived index**

Run: `conda run -n WTF python scripts/build_vector_index.py`.
Expected: non-empty deterministic Chroma collection with fingerprint/schema/model metadata and zero sample chunks.

- [ ] **Step 6: Run full verification and commit corpus migration**

Run `conda run -n WTF python -m pip check`, `conda run -n WTF python -m pytest -q`, and `git diff --check`. Inspect `git status --short` to ensure only intended corpus/config/docs files are included.

```powershell
git add data/regulations configs/reasoning.yaml README.md
git commit -m "data: add current traffic-law corpus"
```

### Task 8: Retrieval benchmark and final documentation

**Files:**
- Create: `data/evaluation/legal_retrieval.json`
- Create: `scripts/evaluate_legal_retrieval.py`
- Create: `tests/unit/test_retrieval_evaluation.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `score_case(expected: set[tuple[str, str]], retrieved: list[DocumentChunk], ks: tuple[int, ...]) -> dict`.
- Produces: `summarize_scores(case_scores: list[dict], ks: tuple[int, ...]) -> dict`.
- CLI reports case count, Recall@1/@3/@4, MRR, fingerprint, embedding model, evaluated top-k, and failed cases with retrieved locators.

- [ ] **Step 1: Write failing metric tests**

Test exact-hit at rank 1, hit at rank 3, complete miss, multiple acceptable locators, and MRR arithmetic. Metrics compare `(document_id, locator)` pairs, never generated answer text.

- [ ] **Step 2: Verify RED**

Run: `conda run -n WTF python -m pytest tests/unit/test_retrieval_evaluation.py -v`
Expected: FAIL because evaluation helpers do not exist.

- [ ] **Step 3: Implement metric helpers and CLI**

Keep evaluation independent from the LLM: load cases, query `VectorStore`, score retrieved evidence, print aggregate metrics and every failed case. Define per-case Recall@k as 1 when any acceptable expected `(document_id, locator)` appears in the first k results and 0 otherwise; MRR is the reciprocal rank of the first acceptable target, or 0 on a miss. Do not add an arbitrary pass/fail threshold in pytest.

- [ ] **Step 4: Curate 24 benchmark cases from the actual canonical corpus**

Create three cases each for speed, signals, prohibitory signs, lanes, stopping/parking, overtaking, penalties, and QCVN sign meaning. Each case stores `id`, Vietnamese `query`, and one or more exact expected `{document_id, locator}` targets verified against normalized JSON.

- [ ] **Step 5: Verify metric tests and run the benchmark**

Run: `conda run -n WTF python -m pytest tests/unit/test_retrieval_evaluation.py -v`.
Then: `conda run -n WTF python scripts/evaluate_legal_retrieval.py`.
Expected: tests PASS and the script prints Recall@1/@3/@4, MRR, fingerprint/model metadata, and inspectable failed cases. Record results in README as an observed benchmark snapshot, not a legal-quality guarantee.

- [ ] **Step 6: Full suite and commit Task 8**

Run: `conda run -n WTF python -m pip check`, `conda run -n WTF python -m pytest -q`, and `git diff --check`.

```powershell
git add data/evaluation/legal_retrieval.json scripts/evaluate_legal_retrieval.py tests/unit/test_retrieval_evaluation.py README.md
git commit -m "test: add legal retrieval benchmark"
```

## Final Acceptance

- Confirm `git branch --show-current` is `backend/legal-corpus-v1` and `git status --short` is empty.
- Run the complete test suite in Conda `WTF`; all tests must pass.
- Recompute SHA-256 for the three committed raw artifacts and compare them to the committed manifest; all three official source identities and hashes must match.
- Load every normalized JSON with `LegalDocument.model_validate`; unresolved block count must be zero for committed corpus artifacts.
- Rebuild Chroma twice from unchanged canonical data and confirm identical fingerprint and deterministic chunk IDs.
- Confirm collection metadata names `schema_version`, `embedding_model`, and `corpus_fingerprint` match the build output.
- Run the 24-case retrieval benchmark and preserve the observed scores/failures in README without inventing a threshold.
- Confirm `data/regulations/sample_traffic_rules.md` is absent and no sample text is present in the active Chroma collection.
- Run `git diff --check`; require no output.
- Inspect `main` only by ref and confirm it remains untouched.
- Run the real Ponytail plugin reviewer read-only against `backend/legal-corpus-v1` vs its base; do not substitute a hallucinating local model if the cloud reviewer is unavailable.
