# GNEM Retrieval RAG Codebase Analysis Report

Date: 2026-05-19

## Executive Summary

This repository is a Python Retrieval-Augmented Generation project for answering questions about the Georgia EV supply-chain knowledge base. The system takes company/facility data from Excel workbooks, normalizes it, converts each KB row into parent and child chunks, stores those chunks in PostgreSQL/Neon with pgvector embeddings, and runs a runtime hybrid retrieval flow that combines sparse BM25 search, dense pgvector search, parent expansion, reranking, and local LLM answer generation.

In plain terms: we are building a grounded question-answering pipeline over a Georgia EV industry dataset. The current active direction is no longer Qdrant; the codebase currently uses PostgreSQL tables plus pgvector.

## Repository Shape

```text
GNEM_Retrival_RAG/
├── kb/                         # Source and evaluation Excel workbooks
├── georgia_ev_intelligence/
│   ├── shared/                 # Config, Excel loading, schema helpers, embeddings
│   ├── offline_pipeline/       # Normalization/chunking/indexing into PostgreSQL + pgvector
│   ├── runtime_pipeline/       # Runtime retrieval and answer generation
│   └── outputs/                # Generated normalized KB, chunks, analysis workbooks
├── scripts/                    # One-off analysis utilities
├── tests/                      # Runtime tests, partly stale
├── PROJECT_STRUCTURE.md        # Existing architecture doc, partly outdated
└── requirements.txt
```

## Data Assets

The important Excel files are:

| File | Purpose | Observed shape |
|---|---|---:|
| `kb/GNEM - Auto Landscape Lat Long Updated.xlsx` | Raw/source KB | 205 rows, 15 columns |
| `georgia_ev_intelligence/outputs/Normalized_kb.xlsx` | Cleaned normalized KB plus debug sheets | 205 rows, 16 columns |
| `georgia_ev_intelligence/outputs/parent_chunks.xlsx` | One full parent record per KB row | 205 rows |
| `georgia_ev_intelligence/outputs/child_chunks.xlsx` | Five retrieval-focused child chunks per parent | 1,025 rows |
| `kb/Rewritten_50_questions.xlsx` | 50-question evaluation set | 50 rows |
| `kb/Human validated 50 questions.xlsx` | Human validation/reference answers | 50 rows |

The chunk ratio is exactly what the code intends: 205 parent records x 5 child chunk types = 1,025 child chunks.

## What The System Is Doing

## 1. Shared Layer

Shared code lives under `georgia_ev_intelligence/shared/`.

### Configuration

`shared/config/settings.py` loads `.env` and exposes required runtime settings:

- `NEON_DATABASE_URL`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_MODEL`
- `OLLAMA_BASE_URL`
- `OLLAMA_LLM_MODEL`
- embedding model settings
- retrieval top-k values
- pgvector batch/indexing settings

Important behavior: many environment variables are required at import time. If one is missing, importing `georgia_ev_intelligence.shared.config` can raise immediately.

### KB Loading And Normalization

`shared/data/loader.py` reads the KB Excel, normalizes column names, cleans missing values to `"Unknown"`, normalizes selected domain columns, converts numeric fields, lowercases company names, and adds `_row_id`.

Key normalized columns include:

- `company`
- `category`
- `industry_group`
- `updated_location`
- `address`
- `latitude`
- `longitude`
- `primary_facility_type`
- `ev_supply_chain_role`
- `primary_oems`
- `supplier_or_affiliation_type`
- `employment`
- `product_service`
- `ev_battery_relevant`
- `classification_method`
- `_row_id`

Current issue: `loader.py` has a hardcoded absolute macOS path:

```text
/Users/sreejamuvva/Desktop/GNEM_Retrival_RAG/kb/GNEM - Auto Landscape Lat Long Updated.xlsx
```

In this Linux workspace, the real file is under `kb/`. As written, fresh normalization/indexing through `loader.load()` may fail unless that absolute path exists.

### Embeddings

`shared/embeddings.py` loads a SentenceTransformer model and applies document/query prefixes from config. Offline indexing uses document prefixes; runtime dense retrieval uses query prefixes.

## 2. Offline Pipeline

Offline code lives under `georgia_ev_intelligence/offline_pipeline/`.

The main entry point is:

```bash
python -m georgia_ev_intelligence.offline_pipeline.index_pgvector
```

Optional modes:

```bash
python -m georgia_ev_intelligence.offline_pipeline.index_pgvector --dry-run --preview 3
python -m georgia_ev_intelligence.offline_pipeline.index_pgvector --recreate-child-table
```

### Parent Chunk Creation

`chunking/parent_chunk.py` creates one `ParentRecord` per normalized KB row. Each parent contains:

- stable `record_id`
- `source_row_id`
- structured KB fields
- raw row JSON
- full `parent_chunk_text` for LLM context

The `record_id` is generated from row identity fields and an MD5 hash, for example:

```text
KB_ROW_0001_<hash>
```

### Child Chunk Creation

`chunking/child_chunk.py` defines five child chunk types:

- `identity`
- `product_role`
- `oem_relationship`
- `location_employment`
- `classification`

Each child chunk is a focused retrieval representation of a parent row. It carries:

- `chunk_id`
- `parent_record_id`
- `chunk_type`
- `source_type`
- `embedding_text`
- lightweight metadata

This design separates retrieval from answer generation:

- child chunks are smaller and targeted for matching queries
- parent chunks preserve full row context for final answers

### PostgreSQL Storage

`postgres_store.py` creates/upserts `parent_chunks`.

`pgvector_store.py` creates/upserts `child_chunks` with:

- `chunk_id`
- `parent_record_id`
- `chunk_type`
- `source_type`
- `source_row_id`
- `metadata` JSONB
- `embedding VECTOR(n)`

The pgvector index uses cosine distance:

```sql
embedding <=> query_vector
```

## 3. Runtime Pipeline

Runtime code lives under `georgia_ev_intelligence/runtime_pipeline/`.

The active implementation is concentrated in `runtime_pipeline/hybrid_retrieval/` plus lower-level retrieval and generation helpers.

### Runtime Retrieval Flow

Current active flow:

```text
User question
→ BM25 child retrieval from PostgreSQL child_chunks
→ dense child retrieval from pgvector child_chunks
→ merge and dedupe child results by chunk_id
→ map child results to parent_record_id
→ fetch parent chunks from PostgreSQL parent_chunks
→ rerank parent contexts with cross-encoder
→ send parent_chunk_text as final context
→ generate answer with local Ollama LLM
→ write answers/traces to Excel for evaluation
```

### BM25 Sparse Retrieval

`runtime_pipeline/retrieval/bm25_retriever.py`:

- loads all `child_chunks` from PostgreSQL
- reconstructs searchable text from `chunk_type` and metadata
- tokenizes with domain-aware handling for hyphens and slashes
- builds an in-memory `BM25Okapi` index
- returns matching `RetrievedChildChunk` objects

This is lazy-loaded and cached inside the retriever instance.

### Dense pgvector Retrieval

`runtime_pipeline/retrieval/dense_pgvector_retriever.py`:

- embeds the user query using the configured SentenceTransformer
- normalizes the query embedding
- searches PostgreSQL `child_chunks` by pgvector cosine distance
- returns matching `RetrievedChildChunk` objects

### Hybrid Orchestration

`runtime_pipeline/hybrid_retrieval/orchestrator.py` runs sparse and dense retrievers in parallel using `ThreadPoolExecutor`.

`factory.py` wires the default pipeline:

- `BM25ChildRetriever`
- `DenseChildRetriever`
- `ChildResultMerger`
- `ParentChildMapper`
- `CrossEncoderReranker`

Current config in `hybrid_retrieval/config.py`:

```text
RETRIEVER_TOP_K = 250
RERANKER_TOP_K = 45
RERANKER_MODEL = cross-encoder/ms-marco-MiniLM-L-6-v2
```

Note: `hybrid_retrieval/README.md` still says retriever top-k is 100, but the code and tests expect 250.

### Parent Mapping

`parent_mapper.py` maps retrieved children to parent records by `parent_record_id`.

`retrieval/parent_fetcher.py` fetches the matching parent records from PostgreSQL and returns `ParentContext` objects:

- `record_id`
- `source_row_id`
- `parent_chunk_text`

### Reranking

`hybrid_retrieval/reranker.py` loads a SentenceTransformers cross-encoder and reranks full parent contexts against the user query. The final set is capped at 45 parents by default.

One important design detail: the earlier written goal mentioned Reciprocal Rank Fusion, but the active code currently merges sparse+dense by order and deduplicates by `chunk_id`; it does not compute RRF scores.

## 4. Answer Generation

`runtime_pipeline/generation/llm_client.py` calls local Ollama:

```text
POST {OLLAMA_BASE_URL}/api/generate
model = OLLAMA_LLM_MODEL
temperature = 0.1
top_p = 0.9
num_predict = 4096
```

The answer is cleaned to remove common local-model artifacts such as `<think>...</think>` blocks and accidental Markdown fences.

There are three answer modes:

| Mode | File | Behavior |
|---|---|---|
| Current / RAG + Pretrained | `run_rewritten_50.py`, `run_rewritten_50_all_modes.py` | Uses retrieved parent context with a detailed formatting prompt |
| Only RAG | `only_rag_pipeline.py` | Strictly context-only answer prompt |
| Only Pretrained | `only_pretrained_pipeline.py` | Sends no retrieved context, useful as a comparison baseline |

## 5. Batch Evaluation Workflows

The project is set up to run the 50 rewritten questions and export Excel workbooks.

### Full Current Pipeline

```bash
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50
```

Reads:

```text
kb/Rewritten_50_questions.xlsx
```

Writes generated answers and retrieved context to:

```text
georgia_ev_intelligence/outputs/hybrid_retrieval_rewritten_50/
```

### Retrieval-Only Trace Export

```bash
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50_retrieval_only
```

Exports:

- final retrieved parent context
- dense child retrieval context
- sparse child retrieval context

### All-Modes Comparison

```bash
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50_all_modes
```

Creates separate workbooks for:

- `only_rag.xlsx`
- `only_pre_trained.xlsx`
- `rag_plus_pre_trained.xlsx`

### Coverage Analysis Script

`scripts/analyze_company_coverage.py` compares retrieved companies against hand-curated golden company lists for a specific retrieval-only workbook. It writes a new workbook with dense/context/sparse coverage columns.

Current limitation: the script has hardcoded source and destination paths for one timestamped output file.

## Current Test Status

I ran:

```bash
pytest -q
```

Result: test collection fails before executing tests.

The failures are stale imports in tests for modules that are not present in the active runtime package:

- `georgia_ev_intelligence.runtime_pipeline.ClarificationOptionGenerator`
- `georgia_ev_intelligence.runtime_pipeline.phrase_classifier`
- `georgia_ev_intelligence.runtime_pipeline.query_analyzer`

The newer hybrid retrieval tests exist and align with the active code, but pytest cannot reach them until the stale tests are removed, skipped, or the missing modules are restored.

## Documentation Drift

Several docs/comments do not match the current code:

- `PROJECT_STRUCTURE.md` still describes Qdrant files and Qdrant flow.
- The active code uses PostgreSQL plus pgvector.
- `hybrid_retrieval/README.md` says `RETRIEVER_TOP_K = 100`, but code uses 250.
- Some comments still mention Qdrant payloads even though storage is pgvector/PostgreSQL.
- The original desired runtime flow mentions RRF fusion and citations, but current active code performs deduplication plus cross-encoder reranking and does not yet implement citation objects.

## Current Risks And Gaps

1. The KB loader has a hardcoded absolute path, so fresh indexing may fail outside the original machine.
2. Full test suite is blocked by stale tests for removed or not-yet-restored runtime modules.
3. Current hybrid retrieval does not implement Reciprocal Rank Fusion despite earlier notes requesting it.
4. Parent reranking happens after child-to-parent mapping, so child-level scores are not preserved into parent context.
5. There is no end-to-end `pipeline.py`, FastAPI app, trace logger, or RAGAS runner in the current active tree, despite older architecture notes mentioning them.
6. Runtime generation is local-Ollama only in the active client; Anthropic config exists but is not used by the current generation path.
7. The analysis script is useful but not reusable yet because it hardcodes one workbook path.
8. Import-time required config makes isolated unit tests fragile unless `.env` is present.

## What We Are Doing Overall

The project is moving from raw spreadsheet data toward a production-style RAG workflow:

1. Normalize the Georgia EV supply-chain knowledge base.
2. Convert every KB row into a full parent chunk.
3. Convert every parent into multiple specialized child chunks for better retrieval.
4. Store parents and child embeddings in PostgreSQL/Neon with pgvector.
5. Retrieve candidate child chunks using both lexical BM25 and semantic vector search.
6. Expand child hits back to full parent rows.
7. Rerank the parent context with a cross-encoder.
8. Ask a local LLM to answer using retrieved context.
9. Evaluate output quality across 50 rewritten benchmark questions.
10. Compare retrieval and answer modes through Excel trace workbooks.

## Recommended Next Steps

1. Fix `shared/data/loader.py` to resolve `kb/GNEM - Auto Landscape Lat Long Updated.xlsx` relative to the project root instead of a hardcoded user path.
2. Update `PROJECT_STRUCTURE.md` and `hybrid_retrieval/README.md` to reflect PostgreSQL/pgvector and current top-k values.
3. Decide whether stale clarification/query-analyzer/phrase-classifier tests should be deleted, skipped, or restored with code.
4. Add true RRF fusion if the design still requires it.
5. Add citation-aware context construction if final answers need auditable source IDs.
6. Make the coverage analysis script accept input/output paths as CLI arguments.
7. Add a small end-to-end smoke test that mocks PostgreSQL, embeddings, reranker, and Ollama so runtime orchestration can be tested without external services.

