# ITAS-MM Backend/System Refactor Design

Date: 2026-09-26
Branch: `backend/system-refactor`
Worktree: `D:\Projects\.worktrees\itas-mm-scaffold\backend\system-refactor`

## Intent

Refactor the current scaffold so it aligns more cleanly with the approved Research Proposal without pretending that unfinished research work already exists.

The refactor must preserve the four-stage pipeline:

`Perception -> Context -> Reasoning -> Interaction`

and keep the current scaffold runnable while reducing dead configuration and preserving legal-source provenance through the RAG path.

## Constraints

- Keep the implementation minimal (Ponytail full / YAGNI).
- Do not add speculative interfaces, factories, providers, or adapter layers.
- Do not implement fake lane events or guessed thresholds.
- Do not replace the current lane baseline with a learned model in this refactor.
- Do not add the real legal corpus in this refactor.
- Do not widen the frontend scope except where a backend contract change requires it.
- Existing behavior must remain testable and the current test suite must stay green.
## Baseline

Verified before design:

- `backend/system-refactor` is clean at `18fa148`.
- The branch is one commit ahead and one commit behind `main` because `main` contains the previous merge commit.
- Conda environment `WTF` uses Python 3.12.14.
- `pip check` reports no broken requirements.
- Current suite: 23 tests passing.
- PyTorch is CPU-only (`2.14.0+cpu`); CUDA work is outside this refactor.
- Ollama currently has `gemma4:e2b`, while project config names `llama3.1` and `nomic-embed-text`.
- Chroma collection `traffic_regulations` exists but is empty.
- `data/regulations` contains only the illustrative sample corpus.

## Design choice

Use a proposal-alignment refactor rather than either a cleanup-only pass or a speculative full domain redesign.

The main architectural correction is to preserve regulation evidence metadata already produced by `DocumentChunk` instead of discarding it at retrieval time.

No new abstraction hierarchy is required: reuse the existing `DocumentChunk` model as the retrieval value passed from `VectorStore` to `RagChain` and into the reasoning result.
## Target data flow

```text
Perception
  -> PerceptionResult
Context
  -> ZoneResult
Pipeline
  -> query text
VectorStore.query()
  -> list[DocumentChunk]
RagChain
  -> grounded prompt + retained evidence
ReasoningResult
  -> guidance + retrieved evidence + zone
Interaction
  -> synthesized audio
```

`DocumentChunk` remains the evidence record with:

- `text`
- `source_file`
- `chunk_index`

This is sufficient for the current scaffold. Article/clause/effective-date fields are deferred until the real corpus structure proves they are needed.

## Scope of changes

1. Reconcile the feature branch with current `main` before implementation work.
2. Remove dead backend configuration that has no runtime consumer.
3. Remove the unused `pytest-mock` dependency.
4. Change vector retrieval to preserve `DocumentChunk` metadata.
5. Change `ReasoningResult` to retain retrieved evidence rather than only raw strings.
6. Keep `synthesize_audio` because it supports benchmark/test isolation and later latency evaluation.
7. Keep the current `__init__.py` files and generated architecture artifacts in this refactor to avoid unrelated churn.
8. Keep the current classical lane baseline and its internal line representation.
9. Avoid exposing Hough-specific details as a long-term research contract when touching related response code.
10. Update README/config comments only where required to reflect actual runtime behavior.

## Explicit non-goals

- Training YOLOv11.
- Contrastive-learning implementation.
- Learned row-wise lane detection.
- Real GPS/OSM road-segment fusion.
- Real Vietnamese legal-corpus ingestion.
- Automatic audio playback in the UI.
- Continuous camera/video streaming.
- CUDA/PyTorch installation changes.
- New deployment/container/CI work.

These belong to later research slices and should not be invented during system cleanup.

## Error behavior

The existing fail-safe behavior remains:

- Missing sign weights -> empty sign detections, no fabricated result.
- Empty vector collection -> explicit no-regulation-found reasoning result.
- Invalid uploaded image -> HTTP 400.
- Unexpected pipeline failure -> logged and returned as HTTP 500.
- Missing Ollama/models -> existing explicit connection error path.
## Test seams

Tests should verify behavior through existing public seams:

- `VectorStore.query()` returns evidence with source metadata preserved.
- `RagChain.answer()` keeps grounding evidence in `ReasoningResult`.
- `Pipeline.run()` still produces the same four-stage result and optional TTS behavior.
- Existing FastAPI response behavior remains compatible unless provenance is intentionally surfaced later.

Do not add tests for private helpers merely to increase coverage.

## Success criteria

- Branch is reconciled cleanly with `main` before implementation proceeds.
- Existing 23 tests remain green after refactor.
- New minimum tests prove retrieved legal evidence retains `source_file` and `chunk_index`.
- Empty-corpus RAG behavior remains fail-safe.
- Dead backend config/dependency identified in the audit is removed.
- No speculative interfaces/factories are introduced.
- README/config documentation matches the implementation after the change.
- `git diff --check` is clean.

## Deferred decisions

Lane-event semantics remain undefined until the learned lane approach and evaluation criteria are selected. No pixel threshold will be invented in this refactor.

Structured legal fields such as article, clause, penalty, effective date, and canonical URL will be designed when the three official source documents are ingested and their actual structure is available.