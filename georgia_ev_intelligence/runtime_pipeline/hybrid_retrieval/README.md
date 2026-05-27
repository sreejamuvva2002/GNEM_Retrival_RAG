# Hybrid Retrieval Runtime

This folder contains the active runtime retrieval and 50-question batch answer
flows.

## Active Flow (Multi-Query)

When the input workbook includes `rewritten_query_1 … rewritten_query_5` columns,
all query variants are used for retrieval:

```text
original_question + rewritten_query_1 … rewritten_query_5
↓ (per query, in parallel BM25 + dense)
BM25 child retrieval   → top-150 per query
dense child retrieval  → top-150 per query
↓
merge all child results across all queries
deduplicate by chunk_id (keep first occurrence / best score)
↓
map child chunks → parent_record_id
deduplicate parent_record_ids
fetch full parent_chunk_text values
↓
cross-encoder rerank parent chunks  (top-45)
  ── reranking query = original question ──
↓
return final reranked parent contexts → LLM
```

When the workbook has no rewritten-query columns the flow falls back to
single-query retrieval (same dedup + parent mapping + parent reranking).

**No reranker is applied to child chunks at any point.**

## Pipelines

| Pipeline | LLM behaviour |
|---|---|
| `rag_only` | Context only — no pretrained knowledge |
| `hybrid_rag` | Context as PRIMARY source; pretrained knowledge used to supplement gaps (must be labelled `[From general knowledge: ...]`) |
| `pretrained_only` | No retrieval; pure pretrained knowledge |
| `direct_kb` | All 205 KB rows passed verbatim as context |

## Key Files

```text
factory.py               # builds the default runtime pipeline
orchestrator.py          # BM25 + dense retrieval, multi-query method, parent mapping, reranking
merger.py                # child result merge + chunk_id dedupe
parent_mapper.py         # child-to-parent expansion
reranker.py              # cross-encoder parent scoring
rag_only_pipeline.py     # strict context-only answer pipeline
hybrid_rag_pipeline.py   # context-primary + pretrained-supplement pipeline
pretrained_only_pipeline.py  # no-retrieval pipeline
direct_kb_pipeline.py    # full KB context pipeline
run_baseline.py          # multi-model × multi-pipeline runner → JSONL
build_ragas_report.py    # JSONL → RAGAS-compatible Excel
evaluate_ragas.py        # in-repo RAGAS evaluator (Qwen 14B judge)
```

## Retrieval Defaults

Configured in `config.py`:

```text
RETRIEVER_TOP_K = 250   # used only by single-query path
RERANKER_TOP_K  = 45
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L12-v2"
```

Multi-query path always uses **150 hits per retriever per query** (hardcoded in
`run_baseline.py` as `_MULTI_QUERY_TOP_K_PER_QUERY`).

Optional environment overrides:

```text
HYBRID_RETRIEVER_TOP_K
HYBRID_RERANKER_TOP_K
HYBRID_RERANKER_MODEL
```

## Commands

### Step 0 — Verify input file

```bash
python3 -c "
import pandas as pd
df = pd.read_excel('kb/<your_file>.xlsx', sheet_name=0)
print(df.columns.tolist(), df.shape)
"
```

### Step 1 — Smoke test (3 questions, one model)

```bash
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_baseline \
    --input kb/<your_file>.xlsx \
    --models gemma3:27b \
    --limit 3
```

### Step 2 — Full run (50 questions, all 4 pipelines)

```bash
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_baseline \
    --input kb/<your_file>.xlsx \
    --models gemma3:27b
```

### Step 3 — Build RAGAS Excel report

```bash
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.build_ragas_report \
    --run-dir georgia_ev_intelligence/outputs/baselines/<timestamp>
```

### Step 4 — Run RAGAS evaluation (Qwen 14B judge)

```bash
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.evaluate_ragas \
    --run-dir georgia_ev_intelligence/outputs/baselines/<timestamp> \
    --judge-model qwen2.5:14b
```

Optional flags for `evaluate_ragas`:
- `--embed-model nomic-embed-text` (default)
- `--ollama-url http://localhost:11434` (default)
- `--pipelines rag_only hybrid_rag` (filter to specific pipelines)
- `--models gemma3:27b` (filter to specific model's JSONL files)
- `--output /path/to/scores.json` (custom output path)
