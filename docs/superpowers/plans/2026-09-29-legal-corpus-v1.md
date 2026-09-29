# ITAS-MM Legal Corpus v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Selectively port LuatRAG's proven corpus-building logic into ITAS-MM, under ITAS-native names, and produce a deterministic three-source traffic-regulation corpus for the existing Chroma + `nomic-embed-text` RAG stack.

**Architecture:** Port only source acquisition, Vietnamese text normalization, heading-aware chunking, stable checksums, and atomic corpus publication from pinned LuatRAG commit `ae2b1c796503e2a58493771bc341b66fb488e053`. Publish one reviewable `data/regulations/corpus.json`, project its chunks directly to the existing `DocumentChunk`, and preserve the already-implemented safe Chroma rebuild and fail-closed citation aliases.

**Tech Stack:** Python 3.12, `requests`, `pypdf>=6.19,<7`, ChromaDB, Ollama `nomic-embed-text`, pytest. LuatRAG is source material only; it is not a runtime dependency.

**Spec:** `docs/superpowers/specs/2026-09-28-legal-corpus-v1-design.md`

**Starting point:** `backend/legal-corpus-v1` at/after `1bc3477`. Earlier work already added `DocumentChunk` provenance, safe Chroma rebuild/fingerprint infrastructure, and citation-alias validation; this plan migrates those pieces to the new corpus artifact instead of reimplementing them.

## Global Constraints

- `main` is strictly read-only: never edit, commit, merge, rebase, reset, force-update, or push it.
- All writes stay on `backend/legal-corpus-v1` or another explicitly approved non-`main` branch.
- Preserve the existing uncommitted real raw artifacts under `data/regulations/raw/`; do not silently discard or overwrite them during migration.
- Selective port source is only `lqb464/LuatRAG@ae2b1c796503e2a58493771bc341b66fb488e053`.
- Do not add `vendor/luatrag/`, `luatrag_adapter.py`, `src/rag/`, submodules, or a parallel LuatRAG runtime namespace.
- Use ITAS naming consistently: `config/regulations.json`, `scripts/fetch_regulations.py`, `data/regulations/corpus.json`, and existing `src/reasoning/*` modules.
- Do not copy LuatRAG's bootstrap corpus, Gemini layer, SQLite/FTS5/BM25 retrieval, frontend, or upload flow.
- Keep Chroma + Ollama `nomic-embed-text`; do not add a reranker or lexical retrieval in v1.
- Do not OCR QCVN sign-image pages and do not use model knowledge to repair legal text.
- Preserve the current runtime empty-retrieval fail-safe and citation-alias rejection behavior.

## Review Focus

1. Port drift: characterization tests must pin the LuatRAG normalization/chunking behaviors we intentionally reuse before adaptation, so later refactors do not silently change corpus semantics.
2. Source disagreement: an official representation that fails configured identity/content checks must stop or switch only to an explicitly configured official source; never auto-correct legal wording.
3. QCVN image-only pages: pages without a font/text resource are skipped deterministically, while text-bearing pages remain included; no OCR fallback is allowed.
4. Publication safety: a failed fetch/extraction/chunk build must not overwrite different raw bytes or replace the last valid `corpus.json`.
5. Retrieval continuity: corpus migration must preserve deterministic `DocumentChunk` provenance, safe Chroma rebuild behavior, empty-retrieval identity, and fail-closed `[S<n>]` aliases.

## File Structure

- Rename: `config/legal-corpus.json` → `config/regulations.json` — curated source identities.
- Modify: `src/reasoning/legal_normalizer.py` — selectively port text normalization/extraction only; remove deep legal hierarchy parsing.
- Modify: `src/reasoning/document_loader.py` — selectively port heading-aware chunking and load/project `corpus.json`.
- Modify: `scripts/fetch_regulations.py` — bounded official acquisition + raw immutability + corpus assembly/publication.
- Modify: `scripts/build_vector_index.py` — load `corpus.json`, fingerprint, rebuild Chroma.
- Preserve: `src/reasoning/vector_store.py`, `src/reasoning/rag_chain.py` — only adapt if corpus projection requires it.
- Delete when migration tests prove unused: `src/reasoning/legal_models.py`, `scripts/normalize_regulations.py`, obsolete canonical-model/parser tests.
- Create: `THIRD_PARTY_NOTICES.md` — LuatRAG MIT attribution for substantially ported code.
- Create/populate: `data/regulations/corpus.json`, `data/evaluation/legal_retrieval.json`.
- Modify: `configs/reasoning.yaml`, `README.md`.

---
### Task 1: Selectively port normalization and chunking primitives

**Files:**
- Modify: `src/reasoning/legal_normalizer.py`
- Modify: `src/reasoning/document_loader.py`
- Replace: `tests/unit/test_legal_normalizer.py`
- Replace: `tests/unit/test_document_loader.py`
- Create: `THIRD_PARTY_NOTICES.md`

**Interfaces:**
- Produces: `normalize_vietnamese(value: str) -> str`.
- Produces: `html_to_text(value: str) -> str`.
- Produces: `extract_vbpl_text(payload: dict) -> str`.
- Produces: `extract_official_article_text(page_html: str) -> str`, limited to `itemprop="articleBody"`.
- Produces: `extract_pdf_text(path: str | Path) -> str`, skipping only pages without `/Font` resources.
- Produces: `stable_checksum(value: str) -> str`, adapted from LuatRAG's pinned FNV-1a UTF-16 `stable_id`.
- Produces: `split_long_paragraph(paragraph: str, max_size: int) -> list[str]`.
- Produces: `chunk_regulation_text(document_id: str, text: str, *, target_size: int = 950, max_size: int = 1400) -> list[dict[str, object]]`.

- [ ] **Step 1: Write characterization tests before replacing the current parser**

Add tests named `test_normalize_vietnamese_matches_pinned_luatrag_behavior`, `test_html_to_text_preserves_block_boundaries_and_removes_hidden_markup`, `test_stable_checksum_matches_pinned_luatrag_behavior`, and `test_chunking_is_stable_for_same_text`. Pin at least `stable_checksum("abc") == "7aigaz"`, NFC normalization, script/style removal, deterministic chunk IDs/checksums, and heading-aware locators.

- [ ] **Step 2: Write ITAS-specific extraction tests**

Add `test_vbpl_extracts_data_document_content`, `test_official_html_extracts_only_article_body`, `test_pdf_skips_fontless_image_pages`, and `test_pdf_without_usable_text_fails_closed`. Use fake PDF pages/resources; unit tests must not hit the network or parse the real 403-page QCVN.

- [ ] **Step 3: Verify RED against the current deep-parser implementation**

Run: `conda run --no-capture-output -n WTF python -m pytest tests/unit/test_legal_normalizer.py tests/unit/test_document_loader.py -v -p no:cacheprovider --basetemp=$env:TEMP\itas-mm-selective-port-t1-red`
Expected: FAIL because the ITAS-named ported primitives/corpus chunk behavior are not implemented yet.

- [ ] **Step 4: Port only the approved LuatRAG algorithms**

Port/adapt `normalize_vietnamese`, HTML cleanup, `split_long_paragraph`, heading detection, stable checksum logic, and chunk construction from pinned LuatRAG into the ITAS modules. `chunk_regulation_text` emits records with `id`, `ordinal`, `document_id`, `locator_type`, `locator`, `heading`, `text`, and `checksum`; use `heading` locators when present and deterministic `Đoạn <n>` fallback otherwise. Set `locator_type="heading"` when a structural heading is active and `locator_type="chunk"` otherwise. Compute `checksum = stable_checksum(f"{document_id}:{locator}:{body}")`; chunk IDs are exactly `<document_id>:<ordinal>:<checksum>`.

- [ ] **Step 5: Replace the deep parser with extraction-only normalization**

Remove article/clause/point/QCVN hierarchy parsing from `legal_normalizer.py`. `extract_official_article_text` must capture only the configured government article body, and `extract_pdf_text` must skip pages whose `/Resources` contain no `/Font`; it must not OCR or infer image content.

- [ ] **Step 6: Add attribution and verify GREEN**

Copy LuatRAG's pinned MIT license notice into `THIRD_PARTY_NOTICES.md` with repository and commit attribution. Run the targeted tests, then `conda run --no-capture-output -n WTF python -m pytest -q -p no:cacheprovider --basetemp=$env:TEMP\itas-mm-selective-port-t1-full` and `git diff --check`.
Expected: targeted and full suite PASS; whitespace check clean.

- [ ] **Step 7: Commit Task 1**

```powershell
git add src/reasoning/legal_normalizer.py src/reasoning/document_loader.py tests/unit/test_legal_normalizer.py tests/unit/test_document_loader.py THIRD_PARTY_NOTICES.md
git commit -m "refactor: port regulation corpus primitives"
```

### Task 2: Unify official acquisition and corpus publication

**Files:**
- Rename: `config/legal-corpus.json` → `config/regulations.json`
- Modify: `scripts/fetch_regulations.py`
- Replace: `tests/unit/test_fetch_regulations.py`

**Interfaces:**
- Produces: `fetch_with_retries(url: str, *, attempts: int = 4) -> requests.Response`.
- Produces: `fetch_regulations(config_path: str | Path, raw_dir: str | Path, corpus_path: str | Path) -> dict`.
- Consumes Task 1 extraction functions and `chunk_regulation_text(...)`.
- Publishes one artifact shaped as `{"manifest": ..., "sources": [...], "chunks": [...]}`.
- CLI produces: `main(argv: list[str] | None = None) -> None` with `--config`, `--raw-dir`, and `--corpus`; direct script execution bootstraps the repo root before importing `src.*`, and `--help` exits before any network call.

- [ ] **Step 1: Rewrite acquisition tests around the unified corpus contract**

Cover `test_retry_policy_retries_408_429_and_5xx_then_succeeds`, `test_source_identity_mismatch_fails_closed`, `test_same_raw_bytes_are_a_noop`, `test_different_raw_bytes_are_not_overwritten`, `test_failed_source_does_not_replace_existing_corpus`, `test_duplicate_document_ids_are_rejected`, `test_corpus_publication_contains_unique_source_and_chunk_ids`, and `test_cli_help_exits_without_network`. Mock all HTTP calls.

- [ ] **Step 2: Verify RED**

Run: `conda run --no-capture-output -n WTF python -m pytest tests/unit/test_fetch_regulations.py -v -p no:cacheprovider --basetemp=$env:TEMP\itas-mm-selective-port-t2-red`
Expected: FAIL because the current fetcher publishes a standalone manifest and has no LuatRAG-style retry/corpus assembly.

- [ ] **Step 3: Rename and pin the curated source config**

Use `config/regulations.json` with exactly these source kinds/identities:
- `law-36-2024-qh15`: `source_kind="vbpl_json"`, VBPL item `170620`, expected `36/2024/QH15`.
- `decree-168-2024-nd-cp`: `source_kind="official_html"`, expected `168/2024/NĐ-CP`, URL `https://xaydungchinhsach.chinhphu.vn/toan-van-nghi-dinh-168-2024-nd-cp-quy-dinh-xu-phat-vi-pham-hanh-chinh-ve-trat-tu-atgt-duong-bo-119241231164556785.htm`.
- `qcvn-41-2024-bgtvt`: `source_kind="official_pdf"`, expected `QCVN 41:2024/BGTVT`, page `https://vanban.chinhphu.vn/?classid=1&docid=211908&pageid=27160`, attachment `https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf`.

- [ ] **Step 4: Port bounded acquisition/publication behavior**

Adapt LuatRAG's four-attempt retry policy (`408`, `429`, and `>=500`, exponential `0.75 * 2**attempt` seconds capped at `10`) using the existing `requests` dependency. Verify VBPL `data.docNum` exactly; verify the official HTML page contains the configured document number and extract only `articleBody`; verify QCVN extracted text contains `QCVN 41:2024/BGTVT`.

- [ ] **Step 5: Assemble deterministic source/chunk records before publication**

Each source record contains `document_id`, `document_number`, `title`, `source_kind`, `source_url`, `raw_file`, raw `sha256`, normalized `text_sha256`, `retrieved_at`, and `chunk_count`. Manifest contains `schema_version="1"`, pinned LuatRAG repository/commit, generation timestamp, source count, and chunk count. Refuse empty text, duplicate source IDs, duplicate chunk IDs, or per-document normalized text above `600_000` chars; refuse total normalized text above `8_000_000` chars.

- [ ] **Step 6: Preserve raw immutability and publish corpus atomically**

Stage/validate all candidates before replacing `corpus.json`; publish via temp sibling + `os.replace`. Existing raw bytes are a no-op only when byte-identical and fail closed otherwise. A failed source may leave newly acquired immutable raw audit bytes, but it must not replace the previous valid corpus artifact.

- [ ] **Step 7: Verify and commit Task 2**

Run the focused acquisition suite, full pytest, `py_compile scripts/fetch_regulations.py`, and `git diff --check`.

```powershell
git add config/regulations.json scripts/fetch_regulations.py tests/unit/test_fetch_regulations.py
git rm config/legal-corpus.json
git commit -m "feat: build unified regulation corpus"
```

### Task 3: Replace canonical-model loading with the corpus artifact contract

**Files:**
- Modify: `src/reasoning/document_loader.py`
- Modify: `tests/unit/test_document_loader.py`
- Modify: `tests/unit/test_corpus_fingerprint.py`
- Delete: `src/reasoning/legal_models.py`
- Delete: `tests/unit/test_legal_models.py`
- Delete: `scripts/normalize_regulations.py`
- Modify: `tests/unit/test_script_entrypoints.py`

**Interfaces:**
- Preserves: existing `DocumentChunk(text, source_file, chunk_index, document_id, locator_type, locator)`.
- Produces: `load_regulation_corpus(corpus_path: str | Path) -> dict`.
- Produces: `build_corpus_chunks(corpus_path: str | Path) -> list[DocumentChunk]`.
- Replaces fingerprint signature with `compute_corpus_fingerprint(corpus: dict, schema_version: str, embedding_model: str) -> str`.

- [ ] **Step 1: Write failing corpus-contract tests**

Cover `test_load_corpus_rejects_empty_sources_or_chunks`, `test_load_corpus_rejects_duplicate_source_ids`, `test_load_corpus_rejects_duplicate_chunk_ids`, `test_load_corpus_rejects_chunk_for_unknown_source`, `test_corpus_chunks_round_trip_provenance`, and `test_corpus_chunk_order_is_deterministic`.

- [ ] **Step 2: Rewrite fingerprint tests for semantic corpus JSON**

Assert that changing only `manifest.generated_at`, `source.retrieved_at`, or `source.raw_file` leaves the fingerprint unchanged; changing source SHA/text hash, chunk text/locator, schema version, or embedding model changes it.

- [ ] **Step 3: Verify RED**

Run: `conda run --no-capture-output -n WTF python -m pytest tests/unit/test_document_loader.py tests/unit/test_corpus_fingerprint.py -v -p no:cacheprovider --basetemp=$env:TEMP\itas-mm-selective-port-t3-red`
Expected: FAIL because the current loader requires per-document `LegalDocument` JSON.

- [ ] **Step 4: Implement strict plain-JSON corpus validation and projection**

Validate required manifest/source/chunk fields, non-empty identities/text, source/chunk uniqueness, and chunk-to-source referential integrity before returning the artifact. Project chunks in artifact order; resolve `source_file` through the matching source, set `chunk_index=ordinal`, and preserve `document_id`, `locator_type`, and `locator` exactly.

- [ ] **Step 5: Retire abandoned canonical-model surfaces**

Remove `legal_models.py`, `normalize_regulations.py`, and their obsolete tests/imports after the new corpus tests are green. Update `test_script_entrypoints.py` to execute `scripts/fetch_regulations.py --help` and `scripts/build_vector_index.py --help` under `python -I -X utf8` from repo root; both must return 0 without network/Ollama work. Do not add compatibility shims.

- [ ] **Step 6: Verify and commit Task 3**

Run targeted tests, then the full suite, `python -m py_compile src/reasoning/document_loader.py`, and `git diff --check`.

```powershell
git add src/reasoning/document_loader.py tests/unit/test_document_loader.py tests/unit/test_corpus_fingerprint.py tests/unit/test_script_entrypoints.py
git rm src/reasoning/legal_models.py tests/unit/test_legal_models.py scripts/normalize_regulations.py
git commit -m "refactor: load unified regulation corpus"
```

### Task 4: Rewire index building and runtime configuration

**Files:**
- Modify: `scripts/build_vector_index.py`
- Modify: `tests/unit/test_build_vector_index.py`
- Modify: `configs/reasoning.yaml`
- Modify: `README.md`
- Verify unchanged: `src/reasoning/vector_store.py`, `src/reasoning/rag_chain.py`
- Verify: `tests/unit/test_vector_store.py`, `tests/unit/test_rag_chain.py`

**Interfaces:**
- `rag.corpus_file` is exactly `data/regulations/corpus.json`.
- `build_index() -> None` loads that artifact with `load_regulation_corpus`, projects `DocumentChunk` values, computes the semantic fingerprint, and calls existing `VectorStore.rebuild(...)`.
- Existing `VectorStore.rebuild(chunks, *, fingerprint, schema_version, embedding_model)` contract remains unchanged.

- [ ] **Step 1: Rewrite build-script tests around `corpus_file`**

Cover `test_build_index_loads_configured_corpus_file`, `test_invalid_corpus_does_not_call_rebuild`, `test_empty_corpus_does_not_call_rebuild`, and `test_build_index_passes_fingerprint_schema_and_embedding_model_to_rebuild`.

- [ ] **Step 2: Verify RED**

Run: `conda run --no-capture-output -n WTF python -m pytest tests/unit/test_build_vector_index.py -v -p no:cacheprovider --basetemp=$env:TEMP\itas-mm-selective-port-t4-red`
Expected: FAIL because the current script still expects canonical normalized documents/config.

- [ ] **Step 3: Rewire build/config without changing retrieval architecture**

Set `rag.corpus_file: "data/regulations/corpus.json"`; remove `regulations_dir`, `chunk_size_chars`, and `chunk_overlap_chars`. Load/validate the artifact before constructing the rebuild candidate. Keep `top_k`, Chroma path/collection, `nomic-embed-text`, and safe full-rebuild semantics unchanged.

- [ ] **Step 4: Update developer workflow documentation**

README production workflow becomes exactly `python scripts/fetch_regulations.py` → `python scripts/build_vector_index.py`. Document the three-source scope, no-OCR limitation, immutable raw provenance, Selective Port attribution, and that Chroma is derived/rebuildable.

- [ ] **Step 5: Run retrieval-regression gates**

Run `tests/unit/test_build_vector_index.py`, `tests/unit/test_vector_store.py`, `tests/unit/test_rag_chain.py`, then full pytest and `git diff --check`. The exact existing empty-retrieval list identity and unknown `[S<n>]` rejection must remain green.

- [ ] **Step 6: Commit Task 4**

```powershell
git add scripts/build_vector_index.py tests/unit/test_build_vector_index.py configs/reasoning.yaml README.md
git commit -m "refactor: index unified regulation corpus"
```

### Task 5: Materialize and verify the real three-source corpus

**Files:**
- Create/populate: `data/regulations/raw/**`
- Create: `data/regulations/corpus.json`
- Delete after successful migration: `data/regulations/manifest.json`
- Delete: `data/regulations/sample_traffic_rules.md`
- Delete after successful replacement: superseded untracked VBPL raw artifact for Decree 168 if it is not referenced by the new corpus.
- Modify tests only if a live-source integration bug requires a TDD regression.

**Interfaces:**
- Consumes Task 2 `fetch_regulations(...)` and Task 4 index build.
- Active raw paths are resolved from `corpus.json`; orphan/superseded Task-7 artifacts are not part of runtime provenance.

- [ ] **Step 1: Verify the pre-pivot raw artifacts before touching migration leftovers**

Recompute SHA-256 for the existing untracked raw files and compare them with the old untracked `data/regulations/manifest.json`. Do not remove that manifest or the old Decree VBPL artifact until the new corpus has been successfully fetched and validated.

- [ ] **Step 2: Run the unified fetch/build against official sources**

Run: `conda run --no-capture-output -n WTF python -X utf8 scripts/fetch_regulations.py`.
Expected: three active source records, non-empty chunk list, immutable raw artifacts, and atomically published `data/regulations/corpus.json`. No Ollama call occurs in this step.

- [ ] **Step 3: Inspect the real artifact before indexing**

Verify exactly three active `document_id` values; all source/raw SHA-256 values match disk; all chunk IDs are unique; every chunk points to an active source; source/chunk counts in the manifest match the arrays. Record per-source normalized character count, chunk count, min/median/p95/max chunk length, and confirm QCVN image-only pages were not OCR'd.

If a live source exposes an unmodeled format/endpoint defect, stop, add the smallest mocked regression test to the owning Task 1/2 test file, watch RED, apply the minimal production fix, and rerun Steps 2–3. Never patch `corpus.json` by hand.

- [ ] **Step 4: Retire migration leftovers only after Step 3 passes**

Remove the old standalone `data/regulations/manifest.json`, the illustrative `sample_traffic_rules.md`, and the superseded Decree VBPL `source.json` if the new corpus references `source.html` instead. Keep the active Law/QCVN raw bytes when their hashes match the new source records.

- [ ] **Step 5: Build the real Chroma index twice**

Run `conda run --no-capture-output -n WTF python -X utf8 scripts/build_vector_index.py` twice without changing corpus/config. Confirm the second build reports the same `corpus_fingerprint`, `schema_version`, and `embedding_model`; query collection metadata and verify there are zero sample-corpus chunks.

- [ ] **Step 6: Full verification and commit Task 5**

Run `conda run --no-capture-output -n WTF python -m pip check`, full pytest, `git diff --check`, and inspect `git status --short`. Only active corpus/raw artifacts plus the intended sample deletion may be staged.

```powershell
git add -A data/regulations
git commit -m "data: add current traffic regulation corpus"
```

### Task 6: Add the real retrieval benchmark and final snapshot

**Files:**
- Create: `data/evaluation/legal_retrieval.json`
- Create: `scripts/evaluate_legal_retrieval.py`
- Create: `tests/unit/test_retrieval_evaluation.py`
- Modify if needed: `src/reasoning/vector_store.py`
- Modify if needed: `tests/unit/test_vector_store.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `score_case(expected: set[tuple[str, str]], retrieved: list[DocumentChunk], ks: tuple[int, ...]) -> dict`.
- Produces: `summarize_scores(case_scores: list[dict], ks: tuple[int, ...]) -> dict`.
- If collection metadata is not already accessible without private-member reach-through, add `VectorStore.get_index_metadata() -> dict[str, object]` for fingerprint/schema/model reporting.

- [ ] **Step 1: Write failing metric tests**

Cover exact hit at rank 1, hit at rank 3, complete miss, multiple acceptable locators, and MRR arithmetic. Metrics compare only `(document_id, locator)` pairs from retrieved evidence.

- [ ] **Step 2: Verify RED and implement metric helpers/CLI**

Run the focused metric test and watch it fail before implementation. The CLI loads 24 cases, queries `VectorStore`, reports case count, Recall@1/@3/@4, MRR, corpus fingerprint, schema version, embedding model, evaluated top-k, and every failed case with retrieved locators. Do not call the generation model.

- [ ] **Step 3: Curate exactly 24 cases from `corpus.json`**

Create three verified Vietnamese queries each for speed, signals, prohibitory signs, lanes, stopping/parking, overtaking, penalties, and QCVN sign meaning/text definitions. Every expected target must be an exact `{document_id, locator}` present in the committed corpus artifact; do not invent a locator from memory.

- [ ] **Step 4: Run and record the real benchmark**

Run: `conda run --no-capture-output -n WTF python -X utf8 scripts/evaluate_legal_retrieval.py`.
Record the observed Recall@1/@3/@4, MRR, fingerprint/model, and notable failed cases in README as a reproducible snapshot, not a legal-quality guarantee and not a pytest threshold.

- [ ] **Step 5: Full suite and commit Task 6**

Run focused metric tests, any new vector-store metadata test, full pytest, `pip check`, `py_compile`, and `git diff --check`.

```powershell
git add data/evaluation/legal_retrieval.json scripts/evaluate_legal_retrieval.py tests/unit/test_retrieval_evaluation.py README.md
git add src/reasoning/vector_store.py tests/unit/test_vector_store.py  # only if metadata accessor was required
git commit -m "test: benchmark regulation retrieval"
```

## Final Acceptance

- Confirm branch is `backend/legal-corpus-v1`; `main` still resolves to `02e3dd02b1de49c3b4f1372c4ddbb443ffe1eb38` and was only inspected.
- Confirm `git status --short` is empty except inaccessible stale pytest temp directories, which must be removed/ignored before calling the branch complete.
- Run the complete Conda `WTF` test suite; all tests must pass.
- Run `python -m pip check`, relevant `py_compile`, and `git diff --check`; all must be clean.
- Recompute SHA-256 for every active raw artifact and compare it to `corpus.json`; all three source identities/hashes must match.
- Validate `corpus.json`: exactly three active sources, non-empty chunks, unique source/chunk IDs, referential integrity, and manifest counts equal array counts.
- Confirm runtime/project code contains no `config/legal-corpus.json`, `normalize_regulations.py`, `legal_models` dependency, `src.rag`, `luatrag_adapter`, or `vendor/luatrag` reference except historical design/plan discussion.
- Confirm `THIRD_PARTY_NOTICES.md` identifies `lqb464/LuatRAG`, pinned commit `ae2b1c796503e2a58493771bc341b66fb488e053`, and preserves the MIT notice.
- Build Chroma twice from unchanged corpus/config and confirm identical fingerprint and deterministic corpus chunk IDs; collection metadata must report matching `schema_version`, `embedding_model`, and `corpus_fingerprint`.
- Confirm `data/regulations/sample_traffic_rules.md` and the old standalone `manifest.json` are absent from the active corpus workflow.
- Run the 24-case retrieval benchmark and preserve the observed metrics/failures in README without inventing a threshold.
- Run the real Ponytail plugin reviewer read-only over `backend/legal-corpus-v1` versus its base; resolve any Critical/Important findings before completion.
