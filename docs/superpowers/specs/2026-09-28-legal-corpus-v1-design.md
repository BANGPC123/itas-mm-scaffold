# ITAS-MM Legal Corpus v1 Design

Date: 2026-09-28
Branch: `backend/legal-corpus-v1`
Base branch: `backend/system-refactor` at `1e9d7f9`

## Intent

Replace the illustrative fixed-size Markdown corpus with a reproducible,
structure-aware Vietnamese traffic-law corpus that preserves legal provenance
through normalization, retrieval, and evaluation.

The canonical legal data must be independent of Chroma. Chroma remains a
derived retrieval index that can be rebuilt deterministically from validated
canonical JSON.

## Source-control guardrail

- `main` remains read-only.
- Do not edit, commit, merge, rebase, force-update, or push `main`.
- `main` may only be fetched, inspected, and compared as reference context.
- All Legal Corpus v1 changes stay on `backend/legal-corpus-v1` or another
  explicitly permitted non-`main` branch.
- This branch starts from the completed `backend/system-refactor` baseline.

## Scope

### In scope

- Immutable raw copies of the approved official legal sources.
- Canonical normalized JSON as the legal source of truth inside the repo.
- Schema validation for legal structure and provenance metadata.
- Hybrid normalization: deterministic parsing first, optional LLM assistance
  only for unresolved blocks, followed by schema validation and human review.
- Structure-aware projection from canonical legal nodes to `DocumentChunk`.
- Deterministic Chroma rebuilds with corpus fingerprinting.
- Retrieval evidence with deterministic locator metadata.
- Retrieval evaluation using Recall@k, MRR, and failed-case inspection.

### Explicit non-goals

- Automatic law-update crawler or automatic publication of normalized output.
- Legal ontology, knowledge graph, reranker, BM25 hybrid search, or LLM judge.
- Dedicated penalty calculator or new public legal-search API.
- UI citation viewer or production legal-advice claims.
- Lane-model, sign-model, GPS/OSM, or streaming work.

## Corpus layout and provenance

```text
data/regulations/
├── raw/
│   ├── law-36-2024-qh15/source.*
│   ├── decree-168-2024-nd-cp/source.*
│   └── qcvn-41-2019-bgtvt/source.*
├── normalized/
│   ├── law-36-2024-qh15.json
│   ├── decree-168-2024-nd-cp.json
│   └── qcvn-41-2019-bgtvt.json
└── manifest.json
```

Raw source files are immutable audit artifacts. `manifest.json` records only
reproducibility metadata needed by the pipeline: `document_id`, source URL,
raw file path, retrieval timestamp, normalized file path, and SHA-256.

The initial corpus targets the three approved sources: Luật 36/2024/QH15,
Nghị định 168/2024/NĐ-CP, and QCVN 41:2019/BGTVT. Source acquisition must use
official or otherwise explicitly approved authoritative locations; the repo
must not silently replace a missing source with an illustrative sample.

## Canonical legal model

`src/reasoning/legal_models.py` owns the canonical schema. Use plain Pydantic
models already available in the project; do not add a new schema dependency or
an inheritance hierarchy.

The minimal v1 model contains:

- `LegalDocument`: identity, title, type, source metadata, `articles`, `sections`.
- `LegalArticle`: number, optional title/text, child clauses.
- `LegalClause`: number, optional text, child points.
- `LegalPoint`: label and text.
- `LegalSection`: hierarchical identifier, optional title/text, child sections.

Law and decree documents primarily use `articles -> clauses -> points`. QCVN
may use generic hierarchical `sections`; it must not be forced into an
article/clause shape that the source does not have.

Canonical JSON is the source of truth. The retrieval model is a projection,
not the canonical record. Therefore Chroma metadata must always be sufficient
to trace a hit back to the canonical document and locator, but Chroma does not
store the full canonical object graph.

## Normalization and validation

Normalization flow:

```text
official PDF/HTML -> extract text -> deterministic structure parser
                  -> unresolved blocks -> optional LLM assist
                  -> schema validation -> reviewable canonical JSON
```

Deterministic parsing handles explicit legal markers such as `Điều`, numbered
clauses, lettered points, and QCVN section identifiers. LLM assistance is only
for blocks the deterministic parser cannot resolve; it must not rewrite the
entire document or bypass validation.

A structurally valid LLM response is not proof that the legal content is
correct. Normalized JSON remains a reviewable, committable artifact and is not
auto-published into the index.

Validation must fail on missing source metadata, empty required identities,
empty leaf nodes, duplicate article numbers, duplicate clause numbers within an
article, duplicate point labels within a clause, duplicate section locators,
or raw-file hash mismatches. Validation checks structure and provenance, not
substantive legal correctness.

## Retrieval projection

Keep `DocumentChunk` as the retrieval evidence type and extend it minimally:

```python
DocumentChunk(
    text: str,
    source_file: str,
    chunk_index: int,
    document_id: str,
    locator_type: str,
    locator: str,
)
```

Projection rules are structure-first:

- A legal point becomes one retrieval unit.
- A clause with no points becomes one retrieval unit.
- An article with no clauses becomes one retrieval unit.
- A leaf QCVN section becomes one retrieval unit.
- A legal unit that exceeds the practical embedding ceiling may be split into
  subchunks, but every subchunk retains the same legal locator metadata.

The existing global `800`-character window with `120` overlap is no longer the
primary corpus strategy. Any size ceiling is a safety limit only and should be
chosen after inspecting the actual canonical documents.

## Vector index lifecycle

`VectorStore` continues to use Chroma and `nomic-embed-text`. Metadata expands
to include `document_id`, `source_file`, `chunk_index`, `locator_type`, and
`locator`. IDs must be deterministic, for example:

```text
document_id::locator_type::locator::chunk_index
```

V1 uses full deterministic rebuilds rather than incremental reconciliation.
The build flow is:

```text
load all canonical JSON -> validate all -> project all chunks
-> verify embedding readiness -> compute fingerprint
-> replace/recreate collection -> embed/upsert all chunks
```

Validation and candidate-corpus preparation happen before replacing the current
collection so a malformed corpus does not destroy the last usable index.

The corpus fingerprint is SHA-256 over canonicalized semantic JSON plus schema
version and embedding model name. Canonical serialization sorts keys and omits
non-semantic local details such as absolute paths and retrieval timestamps.
The fingerprint and schema/model identity are stored with the derived index.

## Grounding and citations

`RagChain` still retrieves top-k `DocumentChunk` values and keeps the current
empty-retrieval fail-safe. Prompt context should include deterministic source
labels derived from chunk metadata, for example:

```text
[1] Nghị định 168/2024/NĐ-CP — Điều 6 Khoản 1 Điểm a
<retrieved text>
```

The application formats citations from retrieval metadata. The LLM must not be
trusted to invent or reconstruct legal citations from its generated prose.
`ReasoningResult` continues to retain the exact retrieved evidence internally;
no public API widening is required in this slice.

## Retrieval evaluation

Add `data/evaluation/legal_retrieval.json` with roughly 20–30 curated queries
covering speed, signals, prohibitory signs, lanes, stopping/parking, overtaking,
penalties, and QCVN sign meaning. Each case identifies the expected canonical
legal locator(s).

`scripts/evaluate_legal_retrieval.py` reports Recall@1, Recall@3, Recall@4,
MRR, failed cases, corpus fingerprint, embedding model, and evaluated top-k.
V1 records scores but does not introduce an arbitrary pytest pass threshold.

## Module boundaries

Keep the feature inside the existing reasoning package:

```text
src/reasoning/
├── legal_models.py
├── legal_normalizer.py
├── document_loader.py
├── vector_store.py
├── rag_chain.py
└── ollama_client.py

scripts/
├── normalize_regulations.py
├── build_vector_index.py
└── evaluate_legal_retrieval.py
```

Do not add repository/service/provider/adapter layers. `legal_models.py` owns
canonical structure. `legal_normalizer.py` handles source-text normalization.
`document_loader.py` loads validated canonical JSON and projects retrieval
chunks. `vector_store.py` owns derived-index storage/retrieval only.

Developer workflow stays small: normalize sources, review canonical JSON, then
build the validated index. A separate validation CLI is unnecessary unless an
independent use case appears.

## Migration from the current corpus

The illustrative `sample_traffic_rules.md` is removed from the indexed corpus.
`build_corpus_chunks()` changes from loading arbitrary Markdown/text and fixed
windows to loading canonical JSON and projecting legal leaf nodes. The old
`chunk_text()` function is not retained as the production corpus API; a small
private long-leaf splitter may remain only as a safety mechanism.

`configs/reasoning.yaml` should stop presenting `chunk_size_chars` and
`chunk_overlap_chars` as global corpus semantics. A maximum chunk-size ceiling
may be added only after inspection of the three real normalized documents.

The public FastAPI response remains unchanged. Existing pipeline behavior and
the explicit no-regulation-found result remain compatible.

## Testing strategy

- Schema tests: valid law/decree hierarchy, valid QCVN sections, duplicates,
  empty leaves, missing provenance, and hash mismatch failures.
- Projection tests: point, clause, article, QCVN leaf, and long-leaf metadata
  preservation.
- VectorStore tests: deterministic metadata round-trip and text/metadata pairing.
- RagChain tests: source-labeled grounding while retaining exact evidence.
- Retrieval benchmark: separate executable evaluation, not a unit-test score gate.

## Failure behavior

Normalization/indexing fails fast on missing raw source, raw SHA-256 mismatch,
invalid canonical JSON, schema violations, duplicate legal locators, empty
canonical corpus, unavailable Ollama, or missing embedding model. No sample or
open-domain fallback is allowed.

Runtime querying keeps the current fail-safe behavior: an empty retrieval set
returns `Không tìm thấy quy định phù hợp trong cơ sở dữ liệu hiện có.` and does
not invoke generation.

## Success criteria

- Three approved source documents exist as immutable raw artifacts with manifest
  provenance and validated canonical JSON.
- Canonical models represent both article/clause/point and QCVN section forms.
- Retrieval chunks preserve deterministic document and legal locator identity.
- Rebuilding the same canonical corpus with the same schema/model yields the
  same corpus fingerprint and deterministic chunk IDs.
- Chroma contains no stale chunks from an older corpus after a successful build.
- RagChain context exposes deterministic source labels without LLM-generated
  citation authority.
- The curated benchmark produces Recall@1/@3/@4, MRR, and inspectable failures.
- Existing regression tests remain green and no speculative architecture is added.
- `main` remains untouched throughout the work.
