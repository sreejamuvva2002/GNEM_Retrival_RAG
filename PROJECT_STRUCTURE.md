# Georgia EV Intelligence Project Structure

This document explains the current project layout and the target folder
responsibilities. The goal is to keep each part of the system isolated so that
future changes to indexing, retrieval, query rewriting, generation, API, or
evaluation do not unexpectedly affect the other parts.

## Current Runtime Flow

The main application flow is:

1. Load configuration from `.env` through `georgia_ev_intelligence.config`.
2. Load the Georgia EV KB spreadsheet.
3. Build schema metadata from the KB columns and values.
4. Build or connect to the semantic retrieval backend.
5. Resolve deterministic keywords and detect analytical operations.
6. Generate semantic probes and KB-grounded rewritten queries.
7. Retrieve rows through keyword filters, semantic search, BM25, column search,
   exact entity search, and reciprocal-rank fusion.
8. Apply count, ranking, aggregation, or single-point-of-failure logic when
   needed.
9. Format evidence rows.
10. Send evidence to the final answer generator.
11. Return API, eval, smoke-test, or Excel output.

## Recommended Folder Responsibilities

```text
georgia_ev_intelligence/
├── config/       # Environment settings, filesystem paths, model/backend config
├── data/         # KB loading, schema metadata, optional data overrides
├── indexing/     # Chunking, embeddings, and Qdrant indexing
├── retrieval/    # Vector search, BM25, filters, RRF fusion, evidence selection
├── query/        # Query rewriting, keyword resolution, term matching, operations
├── reasoning/    # Intent handling, counts, ranks, aggregations, support labels
├── generation/   # Prompts, final answer synthesis, hallucination-risk heuristics
├── pipeline/     # End-to-end orchestration and result models
├── api/          # FastAPI app, routes, request/response schemas
├── evaluation/   # Human-validated QA loading, metrics, reports
└── scripts/      # Thin command wrappers for indexing, eval, smoke tests
```

## What Each Folder Does

### `config/`

Owns runtime configuration. Code in this folder should read `.env`, define
paths, expose model names, Qdrant settings, thresholds, and output locations.
Business logic should not live here.

### `data/`

Owns loading and describing structured KB data. This includes Excel loading,
column normalization, schema/index metadata, and optional row overrides. It
should not call LLMs or vector databases.

### `indexing/`

Owns offline/index-time work: turning KB rows into chunks, embedding those
chunks, creating Qdrant collections, and upserting vectors. This folder is used
by the indexing script, not by final answer generation except through the
indexed Qdrant data it creates.

### `retrieval/`

Owns all evidence retrieval mechanisms: Qdrant retrieval, in-memory dense
retrieval, BM25, dataframe filters, exact entity lookup, reciprocal-rank fusion,
and evidence formatting.

### `query/`

Owns question understanding. This includes deterministic keyword resolution,
term matching, analytical operation detection, and LLM-based query rewriting.
This folder should produce retrieval-ready queries and filters, not final
answers.

### `reasoning/`

Owns deterministic dataframe operations after retrieval: count, rank,
aggregate-sum, single-point-of-failure, and support-level calculation.

### `generation/`

Owns final answer generation. It formats prompts, calls Ollama or Anthropic,
cleans model output, and estimates hallucination risk from evidence coverage.

### `pipeline/`

Owns orchestration only. The pipeline should wire together config, data,
query, retrieval, reasoning, evidence, and generation. It should avoid becoming
the place where every helper function lives.

### `api/`

Owns HTTP concerns only: FastAPI app creation, routes, streaming, and
request/response schemas.

### `evaluation/`

Owns offline evaluation: human-validated question loading, token-F1 scoring,
JSON/Excel report writing, and batch runs.

### `scripts/`

Owns command-line entry points. Scripts should be thin wrappers around package
code. They should not contain core business logic.

## Current Transition State

The source files have been consolidated into the responsibility-focused folders
above. The old flat files were removed to avoid duplicate implementations.

Compatibility aliases in `georgia_ev_intelligence/__init__.py` keep older import
paths working for now, so commands such as `uvicorn georgia_ev_intelligence.api:app`
and imports such as `from georgia_ev_intelligence import pipeline, config` continue
to work.

## Safe Refactor Rule

Move one responsibility at a time:

1. Move the implementation into the new folder.
2. Keep a compatibility import at the old path.
3. Run import and compile checks.
4. Only then update callers to the new path.

This keeps the project usable during the refactor instead of making a large
one-shot move that is hard to debug.
