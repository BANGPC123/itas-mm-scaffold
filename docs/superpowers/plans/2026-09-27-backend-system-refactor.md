# ITAS-MM Backend/System Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the backend scaffold with the approved proposal by preserving legal-source provenance through retrieval/reasoning and removing dead configuration without inventing unfinished research features.

**Architecture:** Keep the existing four-stage `Perception -> Context -> Reasoning -> Interaction` pipeline. Reuse the existing `DocumentChunk` as the retrieval evidence type, return it from `VectorStore.query()`, carry it through `RagChain` into `ReasoningResult`, and leave the FastAPI response contract unchanged for now.

**Tech Stack:** Python 3.12, dataclasses, ChromaDB, Ollama client, FastAPI, pytest, unittest.mock.

**Spec:** `docs/superpowers/specs/2026-09-26-backend-system-refactor-design.md`

## Global Constraints

- `main` is read-only: no edit, commit, merge, rebase, force-update, or push to `main`.
- All writes and commits stay on `backend/system-refactor` or another explicitly permitted non-`main` branch.
- Keep Ponytail full / YAGNI: no new factories, providers, adapters, or interface hierarchies.
- Do not implement lane-event thresholds, learned lane detection, YOLO training, real legal corpus ingestion, GPS/OSM fusion, streaming, or CUDA changes.
- Keep `Pipeline.run(..., synthesize_audio=...)` unchanged.
- Existing FastAPI response behavior remains compatible in this refactor.
- Baseline before implementation: 23 tests passing in Conda env `WTF`.

## Review Focus

1. Empty Chroma collection must return `[]` without calling the embedding model.
2. Retrieved text must preserve the matching `source_file` and `chunk_index` metadata.
3. Multiple retrieved results must keep Chroma result order and metadata pairing.
4. Empty RAG retrieval must return the existing fail-safe guidance and must not call generation.
5. Populated RAG retrieval must ground the prompt on chunk text while returning the exact evidence objects unchanged.

---
### Task 1: Preserve retrieval metadata in `VectorStore`

**Files:**
- Modify: `src/reasoning/vector_store.py:45-68`
- Create: `tests/unit/test_vector_store.py`

**Interfaces:**
- Consumes: existing `DocumentChunk(text: str, source_file: str, chunk_index: int)` from `src.reasoning.document_loader`.
- Produces: `VectorStore.query(query_text: str, top_k: int) -> list[DocumentChunk]`.

- [ ] **Step 1: Write failing tests for empty retrieval and metadata-preserving retrieval**

Create tests named:
- `test_query_empty_collection_returns_empty_without_embedding`
- `test_query_returns_document_chunks_with_metadata`
- `test_query_preserves_result_order_and_metadata_pairing`

Use `unittest.mock.MagicMock` for `ollama_client` and `_collection`. Assert the empty case returns `[]` and `ollama_client.embed` is not called. For populated cases, stub Chroma-style `documents` and `metadatas` arrays and assert exact `DocumentChunk` values and order.

- [ ] **Step 2: Run the new tests and verify they fail for the current return type**

Run: `conda run -n WTF python -m pytest tests/unit/test_vector_store.py -v`
Expected: metadata tests FAIL because `VectorStore.query()` currently returns `list[str]`.

- [ ] **Step 3: Change `VectorStore.query()` to return `list[DocumentChunk]`**

Keep the empty-collection early return. After Chroma query, pair `results["documents"][0]` with `results["metadatas"][0]` and construct `DocumentChunk` values using `text`, `source_file`, and `chunk_index`; do not introduce a new evidence type.
- [ ] **Step 4: Run Task 1 tests**

Run: `conda run -n WTF python -m pytest tests/unit/test_vector_store.py -v`
Expected: all Task 1 tests PASS.

- [ ] **Step 5: Run the existing suite for regression safety**

Run: `conda run -n WTF python -m pytest -q`
Expected: all existing tests plus the new vector-store tests PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/reasoning/vector_store.py tests/unit/test_vector_store.py
git commit -m "refactor: preserve retrieval evidence metadata"
```

### Task 2: Carry legal evidence through `RagChain`

**Files:**
- Modify: `src/utils/types.py:39-43`
- Modify: `src/reasoning/rag_chain.py:28-61`
- Create: `tests/unit/test_rag_chain.py`

**Interfaces:**
- Consumes: `VectorStore.query(query_text: str, top_k: int) -> list[DocumentChunk]` from Task 1.
- Produces: `ReasoningResult.retrieved_chunks: list[DocumentChunk]` while keeping `guidance_text: str` and `zone_label: str` unchanged.
- [ ] **Step 1: Write failing RAG evidence tests**

Create tests named:
- `test_answer_returns_fail_safe_without_generation_when_no_chunks_found`
- `test_answer_retains_retrieved_document_chunks`
- `test_answer_builds_prompt_from_chunk_text_in_retrieval_order`

Stub `VectorStore.query()` with `MagicMock`. In the empty case assert the existing Vietnamese fail-safe guidance, `retrieved_chunks == []`, the requested `zone_label`, and `ollama_client.generate.assert_not_called()`. In populated cases use two `DocumentChunk` objects, assert `ReasoningResult.retrieved_chunks` equals the same ordered list, and inspect the generation user prompt to verify both `.text` values appear in retrieval order.

- [ ] **Step 2: Run the new RAG tests and verify failure**

Run: `conda run -n WTF python -m pytest tests/unit/test_rag_chain.py -v`
Expected: populated retrieval tests FAIL because `RagChain` currently joins retrieved values as strings and `ReasoningResult` is typed as `list[str]`.

- [ ] **Step 3: Update the reasoning result contract**

In `src/utils/types.py`, make `ReasoningResult.retrieved_chunks` a `list[DocumentChunk]`. Use a type-only import (`TYPE_CHECKING`) plus the existing future-annotations behavior so `src.utils.types` does not acquire a runtime dependency on the reasoning package.

- [ ] **Step 4: Update `RagChain.answer()` for `DocumentChunk` evidence**

Keep the existing method signature `answer(self, query_text: str, zone_label: str) -> ReasoningResult`. Build the prompt context from `chunk.text` values only, preserve the ordered `DocumentChunk` list unchanged in the returned result, and keep the empty-retrieval fail-safe path unchanged.

- [ ] **Step 5: Run Task 2 tests**

Run: `conda run -n WTF python -m pytest tests/unit/test_rag_chain.py tests/unit/test_vector_store.py -v`
Expected: all Task 1 and Task 2 tests PASS.
- [ ] **Step 6: Run the full suite**

Run: `conda run -n WTF python -m pytest -q`
Expected: all tests PASS; FastAPI integration behavior remains unchanged because the API still exposes only `retrieved_chunk_count`.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/utils/types.py src/reasoning/rag_chain.py tests/unit/test_rag_chain.py
git commit -m "refactor: retain legal evidence in reasoning"
```

### Task 3: Remove dead configuration and align documentation

**Files:**
- Modify: `requirements.txt`
- Modify: `configs/interaction.yaml`
- Delete: `.env.example`
- Modify: `README.md`

**Interfaces:**
- Consumes: runtime configuration already read from YAML plus `LOG_LEVEL` read directly from the process environment.
- Produces: no runtime API change; removes configuration surfaces with no consumer and documents the actual setup path.

- [ ] **Step 1: Remove the unused dependency and dead config keys**

Delete `pytest-mock` from `requirements.txt`. Delete `tts.engine` from `configs/interaction.yaml`; keep `rate`, `volume`, `voice_id`, and `output_dir` unchanged. Delete `.env.example` because the application does not load dotenv files and the listed Ollama/backend variables are not consumed.

- [ ] **Step 2: Update README installation/environment instructions**

Remove `cp .env.example .env`. State that Ollama model/base URL values come from `configs/reasoning.yaml`, Uvicorn host/port come from the launch command, and `LOG_LEVEL` is an optional process environment variable. Correct the `Pipeline.run` documentation to the actual `(image, latitude, longitude, synthesize_audio=True)` shape.
- [ ] **Step 3: Align the README with the provenance refactor without widening scope**

Document that retrieved regulation evidence now retains `source_file` and `chunk_index` internally. Keep the current limitations explicit: illustrative corpus, classical lane baseline, no trained sign weights, and no dataset-level evaluation. Do not claim article/clause-level citations or real-time streaming.

- [ ] **Step 4: Verify dead configuration is gone**

Run:
```powershell
git grep -n "pytest-mock\|OLLAMA_BASE_URL\|OLLAMA_GENERATION_MODEL\|OLLAMA_EMBEDDING_MODEL\|BACKEND_HOST\|BACKEND_PORT\|engine: \"pyttsx3\"" -- . ':!docs/superpowers/*'
```
Expected: no matches from active project configuration or requirements.

- [ ] **Step 5: Verify dependencies and the complete test suite**

Run:
```powershell
conda run -n WTF python -m pip check
conda run -n WTF python -m pytest -q
```
Expected: `No broken requirements found.` and all tests PASS.

- [ ] **Step 6: Verify source-control and whitespace guardrails**

Run:
```powershell
git branch --show-current
git diff --check
git status --short
```
Expected: branch is `backend/system-refactor`; `git diff --check` is silent; only Task 3 files are modified before commit.

- [ ] **Step 7: Commit Task 3**

```bash
git add requirements.txt configs/interaction.yaml README.md
git rm .env.example
git commit -m "chore: remove dead project configuration"
```

## Final Acceptance

- Run `conda run -n WTF python -m pytest -q` once more after all task commits.
- Run `git diff --check` and require no output.
- Run `git status --short` and require a clean worktree.
- Run `git log --oneline -5` and verify all implementation commits are on `backend/system-refactor`.
- Do not checkout, edit, merge, rebase, commit, or push `main` at any point.
