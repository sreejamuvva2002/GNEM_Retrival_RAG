# Georgia EV Intelligence Project Structure

The codebase is now split into three explicit boundaries:

```text
georgia_ev_intelligence/
├── offline_pipeline/   # Index-time pipeline only
├── shared/             # Code used by both offline and runtime
└── runtime_pipeline/   # Question-answering runtime only
```

The shared KB workbook remains in top-level `kb/`:

```text
kb/GNEM - Auto Landscape Lat Long Updated.xlsx
```

That workbook is shared data. Offline indexing reads it to build Qdrant chunks,
and runtime retrieval reads it to map Qdrant hits back to KB rows.

## Offline Pipeline

Offline code prepares the searchable vector index. It should not import runtime
query rewriting, retrieval fusion, reasoning, generation, API, or evaluation
code.

```text
georgia_ev_intelligence/offline_pipeline/
├── index_qdrant.py   # CLI entrypoint for dry-run, preview, and Qdrant indexing
├── chunking/         # Parent chunks, child chunks, relationships, operations
└── qdrant_store.py   # Collection creation and vector upserts
```

Main command:

```bash
python -m georgia_ev_intelligence.offline_pipeline.index_qdrant --dry-run --preview 3
python -m georgia_ev_intelligence.offline_pipeline.index_qdrant --recreate
```

## Shared Files

Shared code is allowed to be imported by both offline and runtime. Changes here
can affect both pipelines, so this folder should stay small and stable.

```text
georgia_ev_intelligence/shared/
├── config/settings.py   # Env vars, paths, model/backend settings
├── data/loader.py       # KB Excel loading and normalization
├── data/schema.py       # Column metadata and filterability
├── embeddings.py        # SentenceTransformer loading and query/doc prefixes
└── qdrant_client.py     # Qdrant client construction
```

Current shared data:

```text
kb/GNEM - Auto Landscape Lat Long Updated.xlsx
kb/Human validated 50 questions.xlsx
```

## Runtime Pipeline

Runtime code answers user questions from the indexed data and loaded KB rows. It
should not import offline chunking or index-upsert logic.

```text
georgia_ev_intelligence/runtime_pipeline/
├── pipeline/runner.py       # End-to-end orchestration
├── retrieval/               # Qdrant search, dense fallback, BM25, RRF, evidence
├── query/                   # Rewriting, keyword resolution, term matching
├── reasoning/               # Counts, ranks, aggregates, SPOF, support labels
├── generation/              # Final LLM synthesis and hallucination heuristic
├── api/                     # FastAPI app
├── evaluation/              # Human QA evaluation helpers
└── scripts/                 # Runtime eval/smoke/unit-check commands
```

Main runtime imports:

```python
from georgia_ev_intelligence.runtime_pipeline import pipeline
from georgia_ev_intelligence.runtime_pipeline.api import app
```

## Dependency Rule

Allowed directions:

```text
offline_pipeline -> shared
runtime_pipeline -> shared
```

Disallowed directions:

```text
offline_pipeline -> runtime_pipeline
runtime_pipeline -> offline_pipeline
shared -> offline_pipeline
shared -> runtime_pipeline
```

This keeps normal changes to one pipeline from affecting the other pipeline.
Only changes inside `shared/` are intentionally cross-cutting.

## Runtime Flow

Short form:

```text
API / eval / script
→ runtime_pipeline.pipeline.run()
→ shared KB load
→ shared schema build
→ runtime semantic retriever
→ runtime keyword resolution and query rewrite
→ runtime retrieval fusion
→ runtime deterministic reasoning
→ runtime evidence formatting
→ runtime final answer generation
```

## Offline Flow

Short form:

```text
offline_pipeline.index_qdrant
→ shared KB load
→ offline parent/child chunk creation
→ shared embedding model
→ shared Qdrant client
→ offline Qdrant collection/upsert
```

## Removed Legacy Folders

The old responsibility-based wrapper folders (`data`, `indexing`, `retrieval`,
`query`, `reasoning`, `generation`, `pipeline`, `api`, `evaluation`, `scripts`,
and `config`) were removed after the split. New code should import from
`offline_pipeline`, `runtime_pipeline`, or `shared` directly.
