# Hybrid Retrieval Runtime

This folder contains the active runtime retrieval and 50-question batch answer
flows.

## Active Flow

```text
question
→ BM25 child retrieval
→ dense pgvector child retrieval
→ merge child results
→ deduplicate by chunk_id
→ map child chunks to parent_record_id
→ deduplicate parent_record_id values in parent_fetcher
→ fetch full parent_chunk_text values
→ cross-encoder rerank parent chunks
→ return final top-k parent contexts
```

Reranking is parent-level because the final LLM prompt receives parent chunks,
not child chunk metadata.

This module does not implement Reciprocal Rank Fusion. The active merge step is
child `chunk_id` deduplication before parent expansion and reranking.

## Key Files

```text
factory.py                         # builds the default runtime pipeline
orchestrator.py                    # coordinates retrieval, parent mapping, reranking
merger.py                          # child result merge + chunk_id dedupe
parent_mapper.py                   # child-to-parent expansion
reranker.py                        # cross-encoder scoring
run_rewritten_50.py                # retrieval + final answer generation
run_rewritten_50_retrieval_only.py # retrieval traces only
run_rewritten_50_all_modes.py      # only RAG, only pretrained, RAG+pretrained
```

## Defaults

Configured in `config.py`:

```text
RETRIEVER_TOP_K = 250
RERANKER_TOP_K = 45
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L12-v2"
```

Optional environment overrides:

```text
HYBRID_RETRIEVER_TOP_K
HYBRID_RERANKER_TOP_K
HYBRID_RERANKER_MODEL
```

## Commands

Retrieval-only smoke test:

```bash
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50_retrieval_only --limit 5
```

Final answer smoke test:

```bash
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50 --limit 5
```

All modes smoke test:

```bash
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50_all_modes --limit 5
```

Full command runbook:

```text
../../../README.md
```
