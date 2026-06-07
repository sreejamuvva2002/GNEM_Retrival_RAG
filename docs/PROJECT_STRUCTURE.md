# Georgia EV Intelligence Project Structure

This project is split into three code boundaries:

```text
georgia_ev_intelligence/
├── offline_pipeline/   # Index-time normalization/chunking/storage pipeline
├── shared/             # Config, data loading, schema helpers, embeddings
└── runtime_pipeline/   # Question answering and evaluation runtime
```

The active storage backend is PostgreSQL + pgvector. The current pipeline does
not use Qdrant.

## Data

Source and evaluation workbooks live in top-level `kb/`:

```text
kb/GNEM - Auto Landscape Lat Long Updated.xlsx
kb/Rewritten_50_questions.xlsx
kb/Human validated 50 questions.xlsx
```

Generated artifacts live in:

```text
georgia_ev_intelligence/outputs/
```

Core artifacts:

```text
Normalized_kb.xlsx
parent_chunks.xlsx
child_chunks.xlsx
```

## Offline Pipeline

Offline code prepares PostgreSQL tables for runtime retrieval:

```text
georgia_ev_intelligence/offline_pipeline/
├── index_pgvector.py       # CLI entrypoint for chunking + PostgreSQL indexing
├── postgres_store.py       # parent_chunks table writes
├── pgvector_store.py       # child_chunks table/vector writes
└── chunking/               # parent chunks, child chunks, relationships
```

Main commands:

```bash
python -m georgia_ev_intelligence.shared.data.loader
python -m georgia_ev_intelligence.offline_pipeline.index_pgvector --dry-run --preview 3
python -m georgia_ev_intelligence.offline_pipeline.index_pgvector
```

The indexing command stores:

```text
parent_chunks      # full parent context rows for generation
child_chunks       # retrieval-focused child rows with pgvector embeddings
```

## Shared Files

Shared code is imported by both offline and runtime code:

```text
georgia_ev_intelligence/shared/
├── config/settings.py   # .env variables, paths, model/backend settings
├── data/loader.py       # KB Excel loading and normalization
├── data/schema.py       # column metadata helpers
└── embeddings.py        # SentenceTransformer loading and query/doc prefixes
```

Allowed dependency directions:

```text
offline_pipeline -> shared
runtime_pipeline -> shared
```

Avoid:

```text
offline_pipeline -> runtime_pipeline
runtime_pipeline -> offline_pipeline
shared -> offline_pipeline
shared -> runtime_pipeline
```

## Runtime Pipeline

Runtime code answers questions from PostgreSQL-indexed parent and child chunks:

```text
georgia_ev_intelligence/runtime_pipeline/
├── schemas.py
├── retrieval/
│   ├── bm25_retriever.py
│   ├── dense_pgvector_retriever.py
│   └── parent_fetcher.py
├── generation/
│   └── llm_client.py
└── hybrid_retrieval/
    ├── factory.py
    ├── orchestrator.py
    ├── merger.py
    ├── parent_mapper.py
    ├── reranker.py
    ├── run_rewritten_50.py
    ├── run_rewritten_50_retrieval_only.py
    └── run_rewritten_50_all_modes.py
```

Runtime flow:

```text
User question
→ BM25 child retrieval from child_chunks
→ dense pgvector child retrieval from child_chunks
→ merge and deduplicate child chunks by chunk_id
→ map children to parent_record_id
→ deduplicate parent IDs
→ fetch parent_chunk_text from parent_chunks
→ cross-encoder rerank parent chunks
→ send reranked top-k parent_chunk_text to Ollama
→ write answer/evaluation workbook
```

The active runtime does not implement Reciprocal Rank Fusion. It performs
ordered merge/deduplication followed by parent-level cross-encoder reranking.

The final answer workbook includes:

```text
question
golden_answer
retrieved_parent_chunks_after_reranking
final_llm_answer
```

See [README.md](README.md) for the full command runbook.
