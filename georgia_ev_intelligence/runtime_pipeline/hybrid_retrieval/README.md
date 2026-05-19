# Isolated Hybrid Retrieval

This folder contains the active runtime retrieval and 50-question batch answer flow.

## Flow

1. `BM25ChildRetriever` retrieves child chunks from the existing BM25 implementation with `top_k=100`.
2. `DenseChildRetriever` retrieves child chunks from the existing pgvector implementation with `top_k=100`.
3. `CrossEncoderReranker` reranks the merged, deduplicated child pool with `cross-encoder/ms-marco-MiniLM-L-6-v2` and keeps the top 45 children.

The BM25 and dense retrievers run in parallel inside `HybridRetrievalOrchestrator`. `ChildResultMerger` deduplicates by `chunk_id`. `ParentChildMapper` then expands the reranked children through the existing `parent_record_id` to `parent_chunks.record_id` convention and returns deduplicated `ParentContext` objects containing parent text.

## SOLID Layout

- `interfaces.py` defines narrow retriever and reranker protocols.
- `bm25_retriever.py` and `dense_retriever.py` adapt existing retrieval code behind the same retriever interface.
- `merger.py` owns result merging and child deduplication.
- `reranker.py` owns cross-encoder scoring.
- `parent_mapper.py` owns child-to-parent expansion.
- `orchestrator.py` ties the injected dependencies together.
- `factory.py` wires the default BM25 + dense + reranker implementation.

## Configuration

Top-k and model values are surfaced in `config.py`:

- `RETRIEVER_TOP_K = 100`
- `RERANKER_TOP_K = 45`
- `RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"`

Use the default entry point like this:

```python
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval import build_default_pipeline

pipeline = build_default_pipeline()
parents = pipeline.retrieve("Which Georgia EV companies supply battery materials?")
```

Run the 50 rewritten questions with the isolated retriever and final-answer prompt:

```bash
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50
```

The command reads only `kb/Rewritten_50_questions.xlsx` and writes an XLSX file with exactly these columns: `s.no`, `question`, `golden answer`, `human validated answer`, and `retrived context`.
