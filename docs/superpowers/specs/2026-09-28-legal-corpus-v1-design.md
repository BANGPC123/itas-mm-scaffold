# ITAS-MM Legal Corpus v1 Design

Date: 2026-09-29
Branch: `backend/legal-corpus-v1`
Base branch: `backend/system-refactor` at `1e9d7f9`

## Intent

Replace the illustrative traffic-law sample with a reproducible Vietnamese
traffic-regulation corpus by selectively porting the proven corpus-building
logic from `lqb464/LuatRAG` into ITAS-MM.

The implementation must look and behave like native ITAS-MM code: one naming
system, one module layout, one retrieval stack. LuatRAG is a pinned source of
algorithms and implementation patterns, not a runtime dependency, submodule,
or nested application.

## Source-control guardrail

- `main` is strictly read-only.
- Never edit, commit, merge, rebase, reset, force-update, or push `main`.
- All writes remain on `backend/legal-corpus-v1` or another explicitly approved
  non-`main` branch.
- Existing uncommitted raw corpus artifacts from Task 7 must not be silently
  discarded while the pivot is implemented.

## Pinned upstream and licensing

The selective port is based only on LuatRAG commit
`ae2b1c796503e2a58493771bc341b66fb488e053`.

The implementation patterns to port are concentrated in:

- `scripts/fetch_legal_corpus.py`: curated official-source acquisition,
  identity verification, HTML cleanup, content hashing, bounded retries,
  atomic publication, and corpus assembly.
- `src/rag/core.py`: Vietnamese normalization, structural heading detection,
  long-paragraph splitting, deterministic chunk construction, and stable
  checksums.

Do not copy LuatRAG's frontend, FastAPI backend, Gemini integration, SQLite
storage, FTS5/BM25 retrieval, bootstrap corpus, or its application-specific
runtime configuration.

LuatRAG is MIT-licensed. Any substantial copied/modified source must retain the
required upstream copyright/license notice in the repository's attribution
material. ITAS does not copy LuatRAG's bundled legal corpus.

## Architectural choice: selective port

The port is source-level adaptation, not an adapter around an external repo:

```text
LuatRAG pinned source
        |
        | selectively port proven algorithms
        v
ITAS-MM regulation acquisition + normalization + chunking
        |
        v
ITAS DocumentChunk
        |
        v
Chroma + nomic-embed-text + RagChain
```

## ITAS naming contract

Upstream names must not survive as parallel concepts inside ITAS. The port uses
ITAS-native names consistently:

| Upstream LuatRAG | ITAS-MM target |
| --- | --- |
| `scripts/fetch_legal_corpus.py` | `scripts/fetch_regulations.py` |
| `html_to_text()` | `src/reasoning/legal_normalizer.py` |
| `normalize_vietnamese()` | `src/reasoning/legal_normalizer.py` |
| `chunk_segments()` | `src/reasoning/document_loader.py` |
| upstream chunk IDs/checksums | ITAS deterministic chunk identity |
| upstream corpus artifact | `data/regulations/corpus.json` |
| upstream curated config | `config/regulations.json` |

Do not add `luatrag_adapter.py`, `vendor/luatrag/`, `src/rag/core.py`, or an
upstream-shaped runtime package. The goal is one vocabulary in the repository.

The existing `config/legal-corpus.json` is renamed to `config/regulations.json`
during migration so config, script, and data-folder terminology all use
`regulations` consistently.

## Scope

### In scope

- Port and adapt LuatRAG's official-source fetch/normalize/chunk algorithms.
- Curate the three ITAS traffic-regulation sources.
- Preserve source identity, hashes, timestamps, and official URLs.
- Produce a deterministic ITAS corpus artifact with source records and chunks.
- Convert corpus chunks directly into existing `DocumentChunk` values.
- Keep Chroma + `nomic-embed-text` as the retrieval layer.
- Keep deterministic citation aliases and retrieval benchmark infrastructure.

### Explicit non-goals

- Vendoring or running LuatRAG as a second application.
- Reimplementing a complete Vietnamese legal parser or legal ontology.
- OCR for sign-image pages in QCVN v1.
- LuatRAG's FTS5/BM25, Gemini layer, frontend, SQLite store, or upload flow.
- Generic automatic law crawling or auto-update services.

## Corpus artifacts and source flow

The production flow is intentionally shorter than the previous design:

```text
config/regulations.json
        |
        v
scripts/fetch_regulations.py
        |
        +--> data/regulations/raw/<document_id>/source.*
        |
        +--> normalize text + structure-aware chunking
        |
        v
data/regulations/corpus.json
        |
        v
scripts/build_vector_index.py
        |
        v
Chroma
```

`data/regulations/corpus.json` is the reviewable corpus source for retrieval. It
contains a top-level manifest, source records, and deterministic chunk records.
Each source records at least `document_id`, title/number, authoritative URL,
raw artifact path, SHA-256, source kind, and retrieval timestamp. Each chunk
records at least deterministic ID, ordinal, `document_id`, locator, heading,
text, and checksum.

Raw files remain immutable audit artifacts. A different response at the same
curated identity must not overwrite existing raw bytes silently.

The separate production step `scripts/normalize_regulations.py` is removed.
Normalization is a pure implementation boundary reused by the fetch/build flow,
not a second CLI or second corpus format. The old per-document canonical
article/clause/point JSON is no longer required for v1 retrieval.

## Source-specific adaptation

`config/regulations.json` curates exactly the ITAS traffic-law sources and their
expected identities. The fetcher supports only the source kinds required by
this corpus; it is not a generic crawler.

For official HTML/JSON legal text, port LuatRAG's bounded-retry, exact-identity,
HTML-to-text, Unicode normalization, size-limit, hashing, and atomic-publication
behavior. ITAS may select a different official endpoint for a curated document
when the configured endpoint cannot produce structurally valid text; the source
URL and hash must make that choice explicit.

For QCVN PDF, use the official text layer only. Pages with no font/text resource
may be skipped deterministically rather than spending extraction time on pure
sign artwork. V1 does not OCR sign images or infer their meaning from pixels.

## Chunking and retrieval projection

Port LuatRAG's proven normalization and structure-aware chunking behavior, then
adapt it to ITAS evidence metadata. Legal headings (`Chương`, `Mục`, `Điều`,
`Khoản`, and equivalents present in the source) are preferred boundaries.
Long paragraphs may be split by sentence/space boundaries with a bounded maximum
size, but no fixed overlapping 800/120-character window is restored.

`DocumentChunk` remains the ITAS retrieval type:

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

The corpus artifact is converted directly to this type. No LuatRAG chunk class
or adapter type survives in runtime code. Chunk IDs remain deterministic and
must be stable for unchanged source text and configuration.

## Vector index lifecycle

ITAS keeps Chroma and `nomic-embed-text`; LuatRAG retrieval is not ported.
`scripts/build_vector_index.py` loads `data/regulations/corpus.json`, validates
its source/chunk identities, computes the corpus fingerprint, and performs the
existing safe full rebuild.

Fingerprint input includes semantic corpus content, schema/artifact version,
and embedding model. Retrieval timestamps and local absolute paths are excluded.
Collection metadata keeps `corpus_fingerprint`, `schema_version`, and
`embedding_model`.

All candidate embeddings must succeed before the current usable collection is
replaced. A malformed or empty corpus must fail before rebuild.

## Grounding and citations

Keep the existing ITAS source-alias contract in `rag_chain.py`:

```text
[S1] <document_id> — <locator>
<retrieved text>
```

The application derives citations from retrieved metadata. Generated aliases
outside the current retrieval set fail closed. No citation parser from LuatRAG
is required if the current ITAS implementation already enforces this invariant.

## Retrieval evaluation

Keep the existing benchmark design: curated Vietnamese traffic-law queries,
expected `(document_id, locator)` targets, Recall@1/@3/@4, MRR, and failed-case
inspection. Evaluation uses ITAS Chroma retrieval only; it does not compare or
blend LuatRAG lexical ranking in v1.

## Module boundaries after pivot

The intended production surface is:

```text
config/
└── regulations.json

src/reasoning/
├── legal_normalizer.py
├── document_loader.py
├── vector_store.py
├── rag_chain.py
└── ollama_client.py

scripts/
├── fetch_regulations.py
├── build_vector_index.py
└── evaluate_legal_retrieval.py

data/regulations/
├── raw/
└── corpus.json
```

`legal_models.py` and `scripts/normalize_regulations.py` are removed if no
remaining runtime/test contract requires them after the selective port. Do not
retain obsolete files merely for compatibility with the abandoned design.

No `src/rag/`, LuatRAG package namespace, vendor tree, or duplicate corpus
configuration is introduced.

## Migration strategy

Existing completed work that remains useful is preserved: `DocumentChunk`
provenance fields, deterministic/safe Chroma rebuild, corpus fingerprinting,
source aliases, citation validation, and retrieval-evaluation scaffolding.

The custom deep legal-structure parser and its canonical-model dependency are
retired where the selective port replaces them. Tests are rewritten around the
new corpus contract instead of keeping obsolete implementation tests green by
adding compatibility shims.

The illustrative `sample_traffic_rules.md` is removed from active corpus use.
`configs/reasoning.yaml` points index building at `data/regulations/corpus.json`
and no longer advertises the old arbitrary fixed-window chunk settings.

## Testing strategy

Tests must validate behavior, not upstream file names:

- Acquisition: exact curated identity, bounded retries, immutable raw bytes,
  deterministic hashes, and atomic corpus publication.
- Normalization: HTML cleanup, Vietnamese Unicode normalization, source-specific
  extraction, and fail-closed handling of unusable source text.
- Chunking: deterministic heading-aware boundaries, long-paragraph splitting,
  stable locator/checksum/IDs, and no duplicate IDs.
- Corpus artifact: deterministic structure, unique source/chunk identities, and
  provenance round-trip into `DocumentChunk`.
- Vector index: stable fingerprint, safe rebuild, stale-chunk removal, metadata
  round-trip, and preservation of the prior collection on pre-rebuild failure.
- RagChain: exact retrieved evidence identity and fail-closed citation aliases.
- Benchmark: metric arithmetic plus real 24-case retrieval report.

Where upstream LuatRAG behavior is ported, add characterization/regression tests
before adapting it so later refactors can distinguish intentional ITAS changes
from accidental drift.

## Failure behavior

Fail closed on source identity mismatch, source fetch exhaustion, raw-byte
mutation, unusable official text, invalid/empty corpus artifact, duplicate IDs,
embedding unavailability, or malformed index input.

Do not repair legal text using model knowledge. If two official representations
disagree, select and record the configured authoritative artifact or stop for
review; do not silently guess a label or provision.

Runtime empty retrieval keeps the existing Vietnamese no-regulation-found
response and does not invoke generation.

## Success criteria

- LuatRAG is used as a pinned implementation source, not merely as conceptual
  inspiration, and the selectively ported logic is identifiable in code review.
- ITAS exposes only ITAS-native file/module/function naming; no parallel LuatRAG
  runtime namespace remains.
- `config/regulations.json`, `scripts/fetch_regulations.py`,
  `data/regulations/corpus.json`, and the reasoning modules form one coherent
  vocabulary.
- The three curated traffic-regulation sources produce immutable raw artifacts,
  provenance metadata, and a deterministic non-empty corpus artifact.
- The production path has no separate `normalize_regulations.py` CLI and no
  dependency on the abandoned deep canonical article/clause/point model unless
  a concrete remaining use case is discovered and reviewed.
- Chroma remains the only v1 retrieval store and continues to use
  `nomic-embed-text`.
- Rebuilding unchanged corpus content yields the same semantic fingerprint and
  deterministic chunk identities.
- Retrieval evidence preserves `document_id` and locator metadata through
  `RagChain`, and unknown generated citation aliases are rejected.
- The 24-case benchmark reports Recall@1/@3/@4, MRR, and inspectable failures.
- Existing non-obsolete regression tests remain green; obsolete parser tests are
  removed/replaced rather than supported with dead compatibility layers.
- `main` remains untouched throughout implementation.
