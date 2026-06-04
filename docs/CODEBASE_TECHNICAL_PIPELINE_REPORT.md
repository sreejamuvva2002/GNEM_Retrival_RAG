# GNEM Retrieval RAG Codebase Technical Pipeline Report

## 1. Project Overview

This project is a Retrieval-Augmented Generation (RAG) codebase for answering questions about a Georgia EV supply-chain knowledge base. The root [README.md](../README.md) describes it as a runtime and indexing pipeline for answering Georgia EV supply-chain questions from the GNEM knowledge base using PostgreSQL, pgvector, BM25, cross-encoder reranking, and local Ollama generation.

The project solves the problem of turning a structured Excel knowledge base into searchable retrieval units, retrieving relevant records for a user question, reranking full parent records, generating a grounded answer, and writing results to Excel workbooks for review or evaluation.

The targeted domain is Georgia EV supply-chain intelligence. This is visible in:

- [README.md](../README.md), which names Georgia EV supply-chain questions.
- [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50.py), whose `PROMPT_TEMPLATE` says the LLM is answering questions about an EV supply chain knowledge base for the state of Georgia.
- [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/only_rag_pipeline.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/only_rag_pipeline.py), whose `ONLY_RAG_PROMPT_TEMPLATE` has the same Georgia EV supply-chain framing.

The input data is Excel:

- Source KB workbook: [kb/GNEM - Auto Landscape Lat Long Updated.xlsx](../kb/GNEM%20-%20Auto%20Landscape%20Lat%20Long%20Updated.xlsx)
- Evaluation/question workbooks:
  - [kb/Human validated 50 questions.xlsx](../kb/Human%20validated%2050%20questions.xlsx)
  - [kb/Rewritten_50_questions.xlsx](../kb/Rewritten_50_questions.xlsx)
  - [kb/RAG_Data_Management_Framework.xlsx](../kb/RAG_Data_Management_Framework.xlsx)

The outputs are:

- Normalized KB/debug workbook: [georgia_ev_intelligence/outputs/Normalized_kb.xlsx](../georgia_ev_intelligence/outputs/Normalized_kb.xlsx)
- Parent chunk workbook: [georgia_ev_intelligence/outputs/parent_chunks.xlsx](../georgia_ev_intelligence/outputs/parent_chunks.xlsx)
- Child chunk workbook: [georgia_ev_intelligence/outputs/child_chunks.xlsx](../georgia_ev_intelligence/outputs/child_chunks.xlsx)
- Runtime/evaluation result workbooks under [georgia_ev_intelligence/outputs/hybrid_retrieval_human_validated_50/](../georgia_ev_intelligence/outputs/hybrid_retrieval_human_validated_50/)

The active architecture is PostgreSQL + pgvector, not Qdrant. This is stated in [README.md](../README.md) and [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md), and the active code uses `psycopg2`, PostgreSQL tables, and pgvector SQL.

## 2. Repository Structure

```text
GNEM_Retrival_RAG/
├── kb/
├── georgia_ev_intelligence/
│   ├── shared/
│   ├── offline_pipeline/
│   ├── runtime_pipeline/
│   └── outputs/
├── tests/
├── docs/
├── README.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
└── .env.example
```

| Folder | What it contains | Why it exists |
|---|---|---|
| [kb/](../kb/) | Source and evaluation Excel workbooks. | Holds external input data and QA workbooks read by loader and batch runners. |
| [georgia_ev_intelligence/shared/](../georgia_ev_intelligence/shared/) | Config, data loading/normalization, schema helpers, embedding helpers. | Shared by offline indexing and runtime retrieval/generation. |
| [georgia_ev_intelligence/offline_pipeline/](../georgia_ev_intelligence/offline_pipeline/) | Chunk builders, parent PostgreSQL storage, child pgvector indexing, CLI entrypoint. | Builds and indexes the retrieval corpus before runtime queries. |
| [georgia_ev_intelligence/runtime_pipeline/](../georgia_ev_intelligence/runtime_pipeline/) | BM25 retrieval, dense pgvector retrieval, parent fetching, hybrid orchestration, reranking, Ollama generation, batch runners. | Answers user/evaluation questions using indexed tables. |
| [georgia_ev_intelligence/outputs/](../georgia_ev_intelligence/outputs/) | Generated normalization, chunking, and run output workbooks. | Stores debug and evaluation artifacts. |
| `scripts/` | No `scripts/` directory exists in the inspected tree. | Unclear from code; scripts functionality appears implemented as Python module CLIs. |
| [tests/](../tests/) | Runtime pipeline tests. | Verifies orchestrator behavior, merge/dedupe, output workbook columns, and batch runner behavior. |
| [docs/](../docs/) | Lightweight docs pointer plus this report. | Holds documentation. |

## 3. Data Files and Workbooks

| File | Purpose | Expected columns/sheets from code or workbook headers | How used |
|---|---|---|---|
| [kb/GNEM - Auto Landscape Lat Long Updated.xlsx](../kb/GNEM%20-%20Auto%20Landscape%20Lat%20Long%20Updated.xlsx) | Raw/source KB workbook. | `Company`, `Category`, `Industry Group`, `Updated Location`, `Address`, `Latitude`, `Longitude`, `Primary Facility Type`, `EV Supply Chain Role`, `Primary OEMs`, `Supplier or Affiliation Type`, `Employment`, `Product / Service`, `EV / Battery Relevant`, `Classification Method` are the columns used by code. Workbook inspection showed `Sheet1`, 1,170 data rows, 19 columns. Extra columns beyond the code-used columns are unclear from code. | Read by `find_kb_excel()` and `load()` in [georgia_ev_intelligence/shared/data/loader.py](../georgia_ev_intelligence/shared/data/loader.py). |
| [georgia_ev_intelligence/outputs/Normalized_kb.xlsx](../georgia_ev_intelligence/outputs/Normalized_kb.xlsx) | Generated normalized KB workbook/debug report. | Sheets: `Normalized_kb`, `missing_summary`, `category_values`, `primary_facility_type_values`, truncated `supplier_or_affiliation_type_va`, `ev_battery_relevant_values`, `ev_supply_chain_role_values`. `Normalized_kb` has 205 data rows and 16 columns in the inspected artifact. | Written by running [georgia_ev_intelligence/shared/data/loader.py](../georgia_ev_intelligence/shared/data/loader.py) as `__main__`. Also configured as `GNEM_EXCEL` in [georgia_ev_intelligence/shared/config/settings.py](../georgia_ev_intelligence/shared/config/settings.py), but active indexing reads the raw KB through `kb_loader.load()`. |
| [georgia_ev_intelligence/outputs/parent_chunks.xlsx](../georgia_ev_intelligence/outputs/parent_chunks.xlsx) | Debug export of parent chunks. | `record_id`, `source_row_id`, `parent_chunk_text`. Inspected artifact has 205 data rows. | Written by `export_parent_chunks_to_xlsx()` in [georgia_ev_intelligence/offline_pipeline/chunking/operations.py](../georgia_ev_intelligence/offline_pipeline/chunking/operations.py), called by [georgia_ev_intelligence/offline_pipeline/index_pgvector.py](../georgia_ev_intelligence/offline_pipeline/index_pgvector.py). |
| [georgia_ev_intelligence/outputs/child_chunks.xlsx](../georgia_ev_intelligence/outputs/child_chunks.xlsx) | Debug export of child chunks. | `chunk_id`, `parent_record_id`, `chunk_type`, `embedding_text`. Inspected artifact has 1,025 data rows. | Written by `export_child_chunks_to_xlsx()` in [georgia_ev_intelligence/offline_pipeline/chunking/operations.py](../georgia_ev_intelligence/offline_pipeline/chunking/operations.py). |
| [kb/Human validated 50 questions.xlsx](../kb/Human%20validated%2050%20questions.xlsx) | Default evaluation/input workbook for active batch runners. | `Num`, `Use Case Category`, `Question`, `Human validated answers` are workbook headers. Inspected artifact has 50 data rows. | Default input for `run_rewritten_50.py`, `run_rewritten_50_retrieval_only.py`, and `run_rewritten_50_all_modes.py` through `DEFAULT_QUESTIONS_WORKBOOK = "Human validated 50 questions.xlsx"`. |
| [kb/Rewritten_50_questions.xlsx](../kb/Rewritten_50_questions.xlsx) | Rewritten 50-question workbook. | Sheet `Q&A`; columns `s.no`, `question`, `answer`; inspected artifact has 50 data rows. | Not the active default in the inspected runner files. Can be passed through `--input`; sheet must be provided if not `Sheet1`. |
| [kb/RAG_Data_Management_Framework.xlsx](../kb/RAG_Data_Management_Framework.xlsx) | Data management/evaluation framework workbook. | Sheets include `README`, `Document_Registry`, `Category_Taxonomy`, `Data_Catalog`, `Chunk_Registry`, `Processing_Pipeline`, `Processing_Log`, `Quality_Metrics`, `Evaluation_Questions`. | No active Python code references this file by path. |
| [georgia_ev_intelligence/outputs/hybrid_retrieval_human_validated_50/20260519_180929_retrieval_only.xlsx](../georgia_ev_intelligence/outputs/hybrid_retrieval_human_validated_50/20260519_180929_retrieval_only.xlsx) | Generated retrieval-only output workbook. | `s.no`, `question`, `human validated answer`, `retrieved context`, trace columns, `dense retrieved context`, `sparse retrieved context`. Inspected artifact has 50 data rows. | Produced by `run_rewritten_50_retrieval_only.py`. |

## 4. Shared Layer

The shared package is [georgia_ev_intelligence/shared/](../georgia_ev_intelligence/shared/). It contains configuration, KB loading/normalization, schema metadata helpers, and embedding model helpers.

### Configuration

Files:

- [georgia_ev_intelligence/shared/config/__init__.py](../georgia_ev_intelligence/shared/config/__init__.py)
- [georgia_ev_intelligence/shared/config/settings.py](../georgia_ev_intelligence/shared/config/settings.py)

Important constants/functions in `settings.py`:

- `ROOT = Path(__file__).resolve().parents[3]`
- `PACKAGE_DIR = ROOT / "georgia_ev_intelligence"`
- `KB_DIR = ROOT / "kb"`
- `OUTPUTS_DIR = PACKAGE_DIR / "outputs"`
- `GNEM_EXCEL = OUTPUTS_DIR / "Normalized_kb.xlsx"`
- `HUMAN_QA_EXCEL = KB_DIR / "Human validated 50 questions.xlsx"`
- `SMOKE_TEST_OUTPUTS_DIR = OUTPUTS_DIR / "smoke_test"`
- `load_dotenv(ROOT / ".env")`
- `_env(name)`: required env lookup; raises `RuntimeError` if missing.
- `_env_bool(name)`: lowercases and compares to `"true"`.
- `_env_int(name)`: casts required env to `int`.
- `_env_optional_float(name, default)`: optional env with float default.
- `_env_optional_int(name, default)`: optional env with int default.

Other files import config via `from georgia_ev_intelligence.shared import config`.

### Data Loader

File: [georgia_ev_intelligence/shared/data/loader.py](../georgia_ev_intelligence/shared/data/loader.py)

Important functions/classes/constants:

- `find_kb_excel()`
- `_norm_column(name)`
- `KBColumns`
- `MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a"}`
- `_normalize_formatting(value)`
- `_is_missing(value)`
- `clean_text(value)`
- `clean_missing_only(value)`
- `clean_numeric(value)`
- `clean_company(value)`
- `clean_category(value)`
- `clean_oem_footprint(value)`
- `clean_primary_facility_type(value)`
- `normalize_separators(value)`
- `clean_product_service(value)`
- `normalize_dataframe(df)`
- `load()`
- `build_debug_report(df)`

Used by:

- `kb_loader.load()` in [georgia_ev_intelligence/offline_pipeline/index_pgvector.py](../georgia_ev_intelligence/offline_pipeline/index_pgvector.py)
- `KBColumns` in parent/child chunking files.

### Schema Handling

File: [georgia_ev_intelligence/shared/data/schema.py](../georgia_ev_intelligence/shared/data/schema.py)

Important constants/classes/functions:

- `NON_FILTER_COLUMNS = {"classification_method", "supplier_or_affiliation_type"}`
- `SKIP_COLUMNS = {"_row_id", "latitude", "longitude", "address"}`
- `PARTIAL_OVERRIDE_COLUMNS = {"primary_oems"}`
- `ColumnMeta`
- `build(df)`

`build(df)` creates a dictionary of column metadata with `unique_values`, `match_type`, `is_numeric`, `is_filterable`, and optional comma-split `components`. In the inspected active runtime, this schema module is not imported by the hybrid retrieval pipeline.

### Embedding Helpers

File: [georgia_ev_intelligence/shared/embeddings.py](../georgia_ev_intelligence/shared/embeddings.py)

Important functions:

- `load_sentence_transformer(model_name)`
- `as_document_text(text)`
- `as_query_text(text)`
- `_prefix(prefix, text)`

Used by:

- [georgia_ev_intelligence/offline_pipeline/pgvector_store.py](../georgia_ev_intelligence/offline_pipeline/pgvector_store.py) for document embeddings.
- [georgia_ev_intelligence/runtime_pipeline/retrieval/dense_pgvector_retriever.py](../georgia_ev_intelligence/runtime_pipeline/retrieval/dense_pgvector_retriever.py) for query embeddings.

## 5. Offline Pipeline Overview

Entrypoint: [georgia_ev_intelligence/offline_pipeline/index_pgvector.py](../georgia_ev_intelligence/offline_pipeline/index_pgvector.py)

End-to-end flow:

```text
Excel KB
→ georgia_ev_intelligence.shared.data.loader.load()
→ normalize rows and add _row_id
→ build_parent_child_chunks(df)
→ validate_relationships(parents, children)
→ export parent_chunks.xlsx and child_chunks.xlsx
→ store_parents_postgres(parents)
→ index_kb_children(artifacts)
→ create/update parent_chunks and child_chunks tables
```

The CLI arguments in `main()` are:

| Argument | Default | Purpose |
|---|---:|---|
| `--model` | `config.EMBEDDING_MODEL` | Embedding model used by `index_kb_children()`. |
| `--recreate-child-table` | `False` | Drops and recreates `child_chunks`; intended after changing model/vector dimension. |
| `--dry-run` | `False` | Builds chunks and exports debug workbooks without updating stores. |
| `--preview` | `0` | Prints selected child previews. |

The offline pipeline always exports debug Excel files before optional dry-run exit.

## 6. Data Normalization

Data loading happens in [georgia_ev_intelligence/shared/data/loader.py](../georgia_ev_intelligence/shared/data/loader.py).

### Excel Loading

`find_kb_excel()` returns:

```text
kb/GNEM - Auto Landscape Lat Long Updated.xlsx
```

`load()` calls:

```python
df = pd.read_excel(kb_path)
```

Then it normalizes column names:

```python
df.columns = [_norm_column(c) for c in df.columns]
```

### Column Name Normalization

`_norm_column(name)`:

1. Converts to lowercase.
2. Strips whitespace.
3. Replaces non-alphanumeric groups with `_`.
4. Strips leading/trailing `_`.

Examples from comments/code:

- `Product / Service` becomes `product_service`.
- `EV / Battery Relevant` becomes `ev_battery_relevant`.

### Missing Values

`MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a"}`.

Missing text values are converted to `"Unknown"` by:

- `clean_text()`
- `clean_missing_only()`
- final `df.fillna("Unknown")`
- final `df.replace("", "Unknown")`

`load()` also removes rows without company identity:

```python
df = df.dropna(subset=[KBColumns.COMPANY]).reset_index(drop=True)
```

### Numeric Fields

`normalize_dataframe(df)` treats these as numeric columns:

- `employment`
- `latitude`
- `longitude`

`clean_numeric(value)`:

1. Converts missing values to `"Unknown"`.
2. Cleans text.
3. Removes commas.
4. Accepts only `-?\d+(\.\d+)?`.
5. Uses `pd.to_numeric(value, errors="coerce")`.
6. Returns `"Unknown"` if parsing fails.

### Row IDs

`load()` creates:

```python
df[KBColumns.ROW_ID] = df.index
```

`KBColumns.ROW_ID` is `"_row_id"`.

### Retrieval and Generation Fields

The parent and child chunk builders use normalized fields from `KBColumns`, including:

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

## 7. Parent Chunk Creation

Parent chunk code lives in [georgia_ev_intelligence/offline_pipeline/chunking/parent_chunk.py](../georgia_ev_intelligence/offline_pipeline/chunking/parent_chunk.py).

A parent chunk is one full KB row represented as a `ParentRecord` with structured columns, the raw row, and formatted `parent_chunk_text`. `build_parent_chunks(df)` in [georgia_ev_intelligence/offline_pipeline/chunking/operations.py](../georgia_ev_intelligence/offline_pipeline/chunking/operations.py) resets the DataFrame index and returns one `ParentRecord` per row:

```python
return [build_parent_record(row) for _, row in df.iterrows()]
```

Important class/function names:

- `ParentRecord`
- `build_parent_record(row)`
- `build_parent_chunk_text(record_id, source_row_id, row)`
- `_source_row_id(row)`
- `_record_id(source_row_id, parts)`

`ParentRecord` fields:

- `record_id`
- `source_row_id`
- `source_type`
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
- `row_id`
- `raw_row`
- `parent_chunk_text`

`record_id` pattern:

```text
KB_ROW_{source_row_id:04d}_{md5_hash}
```

`_record_id()` computes `md5_hash` as the first 12 hex characters of an MD5 hash over a pipe-joined basis containing selected row fields and `source_row_id`.

Parent chunks are used for final LLM context because runtime fetches `parent_chunk_text` from `parent_chunks` and the active reranker reranks `ParentContext` objects. The child chunks are retrieval units, but the LLM receives reranked parent chunks.

## 8. Child Chunk Creation

Child chunk code lives in:

- [georgia_ev_intelligence/offline_pipeline/chunking/child_chunk.py](../georgia_ev_intelligence/offline_pipeline/chunking/child_chunk.py)
- [georgia_ev_intelligence/offline_pipeline/chunking/relationship.py](../georgia_ev_intelligence/offline_pipeline/chunking/relationship.py)
- [georgia_ev_intelligence/offline_pipeline/chunking/operations.py](../georgia_ev_intelligence/offline_pipeline/chunking/operations.py)

A child chunk is a smaller, retrieval-focused view of a parent row. The code creates exactly five child chunks for each parent, one for each `ChildChunkType` enum member.

Child chunk types:

| Type | Fields included in `embedding_text` and metadata |
|---|---|
| `identity` | `company`, `category`, `industry_group`, `updated_location` |
| `product_role` | `company`, `ev_supply_chain_role`, `product_service`, `ev_battery_relevant` |
| `oem_relationship` | `company`, `primary_oems`, `supplier_or_affiliation_type`, `category` |
| `location_employment` | `company`, `updated_location`, `address`, `latitude`, `longitude`, `employment` |
| `classification` | `company`, `primary_facility_type`, `classification_method`, `category`, `ev_battery_relevant` |

Important class/function names:

- `ChildChunkType`
- `CHILD_CHUNK_FIELDS`
- `ChildChunk`
- `build_embedding_text(row, chunk_type, fields)`
- `build_child_metadata(row, fields)`
- `build_child_chunk(parent, row, chunk_type)`
- `build_child_chunks(parent, row)`
- `build_child_chunks_for_parents(parents, df)`

`chunk_id` pattern:

```text
{parent.record_id}_{chunk_type.value.upper()}
```

Each child carries:

- `chunk_id`
- `parent_record_id`
- `chunk_type`
- `source_type`
- `embedding_text`
- `metadata`

The parent-child relationship is direct: every `ChildChunk.parent_record_id` equals the parent `ParentRecord.record_id` that generated it.

## 9. Parent-Child Relationship Validation

Validation exists in `validate_relationships()` in [georgia_ev_intelligence/offline_pipeline/chunking/relationship.py](../georgia_ev_intelligence/offline_pipeline/chunking/relationship.py).

It checks:

- `len(children) == len(parents) * 5`
- each parent has exactly five children
- every `child.parent_record_id` matches a known `parent.record_id`
- no duplicate `child_id` values, implemented using child `chunk_id` values

It is called by [georgia_ev_intelligence/offline_pipeline/index_pgvector.py](../georgia_ev_intelligence/offline_pipeline/index_pgvector.py) immediately after `build_parent_child_chunks(df)`:

```python
validate_relationships(artifacts.parents, artifacts.children)
```

If validation fails, it raises `ValueError` before exporting debug files or writing to PostgreSQL/pgvector. This protects the runtime pipeline from orphan child hits, duplicate child IDs, and mismatched parent-child counts.

## 10. PostgreSQL and pgvector Storage

Storage files:

- Parent table: [georgia_ev_intelligence/offline_pipeline/postgres_store.py](../georgia_ev_intelligence/offline_pipeline/postgres_store.py)
- Child vector table: [georgia_ev_intelligence/offline_pipeline/pgvector_store.py](../georgia_ev_intelligence/offline_pipeline/pgvector_store.py)

Both files connect through `psycopg2.connect(config.NEON_DATABASE_URL)`.

### `parent_chunks` Table

DDL is `_CREATE_TABLE_SQL` in `postgres_store.py`.

Fields:

| Column | Type |
|---|---|
| `record_id` | `TEXT PRIMARY KEY` |
| `source_row_id` | `INTEGER` |
| `source_type` | `TEXT` |
| `company` | `TEXT` |
| `category` | `TEXT` |
| `industry_group` | `TEXT` |
| `updated_location` | `TEXT` |
| `address` | `TEXT` |
| `latitude` | `NUMERIC` |
| `longitude` | `NUMERIC` |
| `primary_facility_type` | `TEXT` |
| `ev_supply_chain_role` | `TEXT` |
| `primary_oems` | `TEXT` |
| `supplier_or_affiliation_type` | `TEXT` |
| `employment` | `NUMERIC` |
| `product_service` | `TEXT` |
| `ev_battery_relevant` | `TEXT` |
| `classification_method` | `TEXT` |
| `row_id` | `INTEGER` |
| `raw_row` | `JSONB` |
| `parent_chunk_text` | `TEXT` |
| `created_at` | `TIMESTAMPTZ DEFAULT NOW()` |
| `updated_at` | `TIMESTAMPTZ DEFAULT NOW()` |

Upsert uses `_UPSERT_SQL`:

```sql
ON CONFLICT (record_id) DO UPDATE SET ...
updated_at = NOW();
```

`store_parents_postgres(parents)` creates the table, upserts all parents with `psycopg2.extras.execute_values(..., page_size=100)`, commits, and returns the row count.

### `child_chunks` Table

DDL is `_CREATE_TABLE_SQL` in `pgvector_store.py`.

Fields:

| Column | Type |
|---|---|
| `chunk_id` | `TEXT PRIMARY KEY` |
| `parent_record_id` | `TEXT NOT NULL` |
| `chunk_type` | `TEXT NOT NULL` |
| `source_type` | `TEXT NOT NULL` |
| `source_row_id` | `INTEGER NOT NULL` |
| `metadata` | `JSONB` |
| `embedding` | `VECTOR({vector_size})` |
| `created_at` | `TIMESTAMPTZ DEFAULT NOW()` |

Before creating the table, `_create_child_chunks_table()` runs:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

If `recreate=True`, it also runs:

```sql
DROP TABLE IF EXISTS child_chunks;
```

The pgvector index is:

```sql
CREATE INDEX IF NOT EXISTS child_chunks_embedding_idx
ON child_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

Distance metric/operator:

- Index operator class: `vector_cosine_ops`
- Runtime SQL operator: `embedding <=> %s::vector`

The code uses normalized embeddings (`normalize_embeddings=True`) for both documents and queries, so cosine distance is the active dense retrieval metric.

## 11. Embedding Model

Embedding helper file: [georgia_ev_intelligence/shared/embeddings.py](../georgia_ev_intelligence/shared/embeddings.py)

Embedding model configuration:

- Env variable: `EMBEDDING_MODEL`
- Example/default value in [.env.example](../.env.example) and [README.md](../README.md): `nomic-ai/nomic-embed-text-v1.5`
- Required by `settings.py` through `_env("EMBEDDING_MODEL")`; there is no fallback default in code.

Other embedding env variables:

- `EMBEDDING_LOCAL_FILES_ONLY`
- `EMBEDDING_TRUST_REMOTE_CODE`
- `EMBEDDING_DOCUMENT_PREFIX`
- `EMBEDDING_QUERY_PREFIX`

Document/query prefixes:

- Offline document text is wrapped by `as_document_text(c.embedding_text)`.
- Runtime query text is wrapped by `as_query_text(query)`.
- `_prefix(prefix, text)` strips the text, avoids double-prefixing if already prefixed, and inserts a space if the configured prefix does not end with a space.

Embedding dimension:

- Determined dynamically in `index_kb_children()`.
- The code calls `model.get_embedding_dimension()` if available, otherwise `model.get_sentence_embedding_dimension()`.
- The dimension is then used in `VECTOR({vector_size})`.
- Exact dimension is unclear from code alone because it depends on the configured model implementation.

Offline embedding generation:

```python
vectors = model.encode(
    texts,
    batch_size=size,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=False,
)
```

Runtime query embedding generation:

```python
query_vec = self._model.encode(
    [as_query_text(query)],
    convert_to_numpy=True,
    normalize_embeddings=True,
)[0].astype(float).tolist()
```

## 12. Runtime Pipeline Overview

Active factory: [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/factory.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/factory.py)

`build_default_pipeline()` builds:

- `BM25ChildRetriever()`
- `DenseChildRetriever()`
- `CrossEncoderReranker(model_name=cfg.reranker_model)`
- `ChildResultMerger()`
- `ParentChildMapper()`
- `HybridRetrievalOrchestrator(...)`

Runtime flow:

```text
User question
→ sparse BM25 child retrieval
→ dense pgvector child retrieval
→ merge child results
→ dedupe child chunks by chunk_id
→ map child chunks to parent_record_id
→ dedupe parent_record_id values in parent_fetcher
→ fetch parent_chunk_text from parent_chunks
→ cross-encoder rerank parent chunks
→ select final top-k parents
→ send reranked parent contexts to LLM
→ generate final answer
→ write Excel output
```

Main classes/functions:

- `HybridRetrievalOrchestrator.retrieve(query)`
- `HybridRetrievalOrchestrator.retrieve_with_sources(query)`
- `HybridRetrievalOrchestrator._retrieve_children(query)`
- `ChildResultMerger.merge(result_sets)`
- `ParentChildMapper.map_to_parents(children)`
- `fetch_parents(parent_record_ids)`
- `CrossEncoderReranker.rerank_parents(query, parents, top_k)`
- `generate_answer(prompt, timeout=180)`

## 13. Sparse Retrieval: BM25

Wrapper file: [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/bm25_retriever.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/bm25_retriever.py)

Implementation file: [georgia_ev_intelligence/runtime_pipeline/retrieval/bm25_retriever.py](../georgia_ev_intelligence/runtime_pipeline/retrieval/bm25_retriever.py)

Important names:

- `BM25ChildRetriever`
- `BM25Retriever`
- `_LOAD_CHUNKS_SQL`
- `tokenize_bm25(text)`
- `_build_bm25_text(chunk_type, metadata)`
- `BM25Retriever._load()`
- `BM25Retriever.search(query, top_k=100)`

Library:

- `rank_bm25.BM25Okapi`

`_LOAD_CHUNKS_SQL`:

```sql
SELECT chunk_id, parent_record_id, chunk_type, metadata
FROM child_chunks;
```

Searchable text is built from child metadata:

```python
parts = [f"chunk_type: {chunk_type}"]
for field_name, value in metadata.items():
    ...
    parts.append(f"{field_name}: {str_value}")
return " ".join(parts)
```

Values that are empty or case-insensitive `"unknown"` are skipped.

Tokenization in `tokenize_bm25(text)`:

- Lowercases text.
- Removes simple possessives with `re.sub(r"'s\b", "", text)`.
- Finds word tokens including hyphenated/slash compounds using `re.findall(r"[\w]+(?:[/\-][\w]+)*", text)`.
- Strips trailing punctuation artifacts.
- Adds the full compound token.
- Splits hyphenated/slash compounds into sub-tokens and adds those too.

BM25 index creation:

```python
self._bm25 = BM25Okapi(corpus_tokens)
```

Top-k:

- The underlying `BM25Retriever.search()` default is `top_k=100`.
- In the active hybrid orchestrator, `BM25ChildRetriever.retrieve()` receives `self._config.retriever_top_k`, default `250`.

Ranking:

```python
scores = self._bm25.get_scores(query_tokens)
n = min(top_k, len(self._chunks))
top_indices = np.argsort(scores)[-n:][::-1]
top_indices = [int(i) for i in top_indices if scores[i] > 0]
```

Returned object:

- `list[RetrievedChildChunk]`
- Fields returned: `chunk_id`, `parent_record_id`, `chunk_type`, `metadata`
- BM25 scores are computed but not stored in `RetrievedChildChunk`.

Conceptual BM25 scoring:

```text
score(D, Q) = sum over query terms of IDF(q) * term-frequency saturation
```

The exact BM25 formula is provided by `rank_bm25.BM25Okapi`; the code does not reimplement or expose its internals.

## 14. Dense Retrieval: pgvector

Wrapper file: [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/dense_retriever.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/dense_retriever.py)

Implementation file: [georgia_ev_intelligence/runtime_pipeline/retrieval/dense_pgvector_retriever.py](../georgia_ev_intelligence/runtime_pipeline/retrieval/dense_pgvector_retriever.py)

Important names:

- `DenseChildRetriever`
- `DensePgvectorRetriever`
- `_SEARCH_SQL`
- `DensePgvectorRetriever.search(query, top_k=100)`

Query embedding:

```python
self._model.encode(
    [as_query_text(query)],
    convert_to_numpy=True,
    normalize_embeddings=True,
)
```

SQL query:

```sql
SELECT
    chunk_id,
    parent_record_id,
    chunk_type,
    metadata
FROM child_chunks
ORDER BY embedding <=> %s::vector
LIMIT %s;
```

Distance operator:

- `<=>`, used by pgvector for distance search with the configured vector operator class.
- The table index uses `vector_cosine_ops`.

Top-k:

- The underlying `DensePgvectorRetriever.search()` default is `top_k=100`.
- In the active hybrid orchestrator, `DenseChildRetriever.retrieve()` receives `self._config.retriever_top_k`, default `250`.

Returned object:

- `list[RetrievedChildChunk]`
- Fields returned: `chunk_id`, `parent_record_id`, `chunk_type`, `metadata`
- Dense distance/similarity is not selected in SQL and not stored in the returned object.

Conceptually, cosine distance compares the direction of the normalized query embedding and child embedding. Because embeddings are normalized, smaller cosine distance indicates closer semantic similarity.

## 15. Hybrid Retrieval

Hybrid retrieval is implemented in [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/orchestrator.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/orchestrator.py).

`HybridRetrievalOrchestrator._retrieve_children()` runs retrievers in parallel using:

```python
with ThreadPoolExecutor(max_workers=len(self._retrievers)) as executor:
```

The result order is restored by storing each future result at its retriever index. With the default factory, index 0 is BM25 and index 1 is dense retrieval.

Merging/deduplication is implemented in [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/merger.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/merger.py):

```python
if child.chunk_id in seen_chunk_ids:
    continue
seen_chunk_ids.add(child.chunk_id)
merged.append(child)
```

Deduplication key:

- `chunk_id`

The current hybrid step is merge + dedupe, not Reciprocal Rank Fusion (RRF). This is stated in [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/README.md](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/README.md) and asserted by `test_child_merger_preserves_retriever_order_without_rrf()` in [tests/runtime_pipeline/test_hybrid_retrieval_orchestrator.py](../tests/runtime_pipeline/test_hybrid_retrieval_orchestrator.py).

RRF is not part of the current pipeline because no active code computes reciprocal ranks, no RRF score is represented in the models, and the merger only preserves first-seen order while deduplicating by `chunk_id`.

## 16. Parent Mapping and Deduplication

Mapping file: [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/parent_mapper.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/parent_mapper.py)

Fetcher file: [georgia_ev_intelligence/runtime_pipeline/retrieval/parent_fetcher.py](../georgia_ev_intelligence/runtime_pipeline/retrieval/parent_fetcher.py)

`ParentChildMapper.map_to_parents(children)` calls:

```python
fetch_parents([child.parent_record_id for child in children])
```

Duplicate parent IDs can occur because:

- A single parent has five child chunks.
- BM25 and dense retrieval can both retrieve child chunks from the same parent.
- Different child chunk types from the same parent can match one query.

Parent ID deduplication occurs in `fetch_parents()` through `_dedupe(parent_record_ids)`. It happens before SQL fetch and before reranking. This is important because the cross-encoder reranker should score each parent context once, and the final LLM context should not contain repeated copies of the same parent record.

The orchestrator also computes `_unique_parent_record_ids(merged_children)` for trace counts.

## 17. Parent Fetching

File: [georgia_ev_intelligence/runtime_pipeline/retrieval/parent_fetcher.py](../georgia_ev_intelligence/runtime_pipeline/retrieval/parent_fetcher.py)

Important names:

- `_FETCH_PARENTS_SQL`
- `fetch_parents(parent_record_ids)`
- `_dedupe(values)`

SQL:

```sql
SELECT
    record_id, source_row_id, parent_chunk_text
FROM parent_chunks
WHERE record_id = ANY(%s);
```

Fetched fields:

- `record_id`
- `source_row_id`
- `parent_chunk_text`

Order preservation:

1. `_dedupe()` preserves first-seen parent ID order.
2. SQL returns rows in unspecified database order.
3. The code builds `parent_data` lookup by `record_id`.
4. It iterates over `ordered_parent_ids` and appends found parents in that order.

Returned object:

- `list[ParentContext]`
- `ParentContext` is defined in [georgia_ev_intelligence/runtime_pipeline/schemas.py](../georgia_ev_intelligence/runtime_pipeline/schemas.py) with `record_id`, `source_row_id`, `parent_chunk_text`.

## 18. Cross-Encoder Reranking

File: [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/reranker.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/reranker.py)

Config file: [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/config.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/config.py)

Exact default model:

```text
cross-encoder/ms-marco-MiniLM-L12-v2
```

Env override:

```text
HYBRID_RERANKER_MODEL
```

Important names:

- `CrossEncoderReranker`
- `CrossEncoderReranker.rerank(...)`
- `CrossEncoderReranker.rerank_parents(...)`
- `CrossEncoderReranker._load_model(model_name)`
- `_flatten_scores(raw_scores)`
- `_as_float(value)`
- `RerankedChildChunk`

The active flow uses parent-level reranking:

```python
reranked_parent_contexts = self._reranker.rerank_parents(
    query=query,
    parents=parent_contexts,
    top_k=self._config.reranker_top_k,
)
```

Child-level reranking still exists as `rerank()`, returning `RerankedChildChunk`, but the orchestrator does not call it in the active path.

Parent-level scoring:

```python
pairs = [(query, parent.parent_chunk_text) for parent in parents]
raw_scores = self._model.predict(pairs, show_progress_bar=False)
scores = _flatten_scores(raw_scores)
```

Conceptual formula:

```text
score = CrossEncoder(query, parent_chunk_text)
```

The cross-encoder jointly reads the query and candidate parent text, unlike dense embedding similarity where query and document are embedded separately.

Top-k after reranking:

- Default `HYBRID_RERANKER_TOP_K` fallback in code: `45`
- Active final parent context count is at most this value.

Returned object:

- `rerank_parents()` returns `list[ParentContext]` sorted by descending cross-encoder score.
- Scores are not returned in the active parent context list.

## 19. Top-K Values

| Top-k / limit | Default | Env / CLI | File path | Used for |
|---|---:|---|---|---|
| `RETRIEVER_TOP_K` | `250` | `HYBRID_RETRIEVER_TOP_K` | [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/config.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/config.py) | Passed to each child retriever by `HybridRetrievalOrchestrator._retrieve_children()`. |
| `RERANKER_TOP_K` | `45` | `HYBRID_RERANKER_TOP_K` | [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/config.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/config.py) | Final parent contexts after cross-encoder reranking. |
| `BM25Retriever.search(top_k)` | `100` | Function argument | [georgia_ev_intelligence/runtime_pipeline/retrieval/bm25_retriever.py](../georgia_ev_intelligence/runtime_pipeline/retrieval/bm25_retriever.py) | Standalone BM25 top-k default; active hybrid overrides with 250. |
| `DensePgvectorRetriever.search(top_k)` | `100` | Function argument | [georgia_ev_intelligence/runtime_pipeline/retrieval/dense_pgvector_retriever.py](../georgia_ev_intelligence/runtime_pipeline/retrieval/dense_pgvector_retriever.py) | Standalone dense top-k default; active hybrid overrides with 250. |
| `--preview` | `0` | CLI | [georgia_ev_intelligence/offline_pipeline/index_pgvector.py](../georgia_ev_intelligence/offline_pipeline/index_pgvector.py) | Number of child chunks printed in offline preview. |
| `--limit` | `None` | CLI | `run_rewritten_50.py`, `run_rewritten_50_retrieval_only.py`, `run_rewritten_50_all_modes.py` | Optional row limit for batch/smoke runs. |
| `--llm-timeout` | `180` seconds | CLI | `run_rewritten_50.py`, `run_rewritten_50_all_modes.py` | Timeout passed to `generate_answer()`. |
| `generate_answer(timeout)` | `180` seconds | Function argument | [georgia_ev_intelligence/runtime_pipeline/generation/llm_client.py](../georgia_ev_intelligence/runtime_pipeline/generation/llm_client.py) | HTTP timeout for Ollama generation. |
| `PGVECTOR_BATCH_SIZE` | Required env; example `64` in `.env.example` | `PGVECTOR_BATCH_SIZE` | [georgia_ev_intelligence/shared/config/settings.py](../georgia_ev_intelligence/shared/config/settings.py), [georgia_ev_intelligence/offline_pipeline/pgvector_store.py](../georgia_ev_intelligence/offline_pipeline/pgvector_store.py) | Offline embedding/upsert batch size. |
| Final LLM context count | At most `RERANKER_TOP_K` | `HYBRID_RERANKER_TOP_K` | [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/orchestrator.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/orchestrator.py) | Number of reranked parent chunks formatted into final prompt. |

## 20. Retrieval Trace

Trace model: `HybridRetrievalTrace` in [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/models.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/models.py)

Fields:

| Field | Meaning | Why it matters |
|---|---|---|
| `sparse_child_count` | Number of BM25 child results returned. | Shows whether sparse retrieval found candidates. |
| `dense_child_count` | Number of dense pgvector child results returned. | Shows whether dense retrieval found candidates. |
| `merged_child_result_count` | Sum of child result counts before dedupe. | Shows the pre-dedup candidate volume. |
| `unique_child_chunk_count` | Child count after `chunk_id` dedupe. | Shows overlap between BM25 and dense retrieval. |
| `unique_parent_id_count` | Unique parent IDs in merged children. | Shows how many parent records are candidates before fetch/rerank. |
| `parent_context_count_before_rerank` | Number of fetched parent contexts before cross-encoder rerank. | Detects missing parent rows or parent dedupe effects. |
| `parent_context_count_after_rerank` | Number of parent contexts returned after reranker top-k. | Confirms final context count sent to answer generation. |

Trace values are exported by:

- `run_rewritten_50.py`
- `run_rewritten_50_retrieval_only.py`
- `run_rewritten_50_all_modes.py`

## 21. LLM Final Answer Generation

LLM client file: [georgia_ev_intelligence/runtime_pipeline/generation/llm_client.py](../georgia_ev_intelligence/runtime_pipeline/generation/llm_client.py)

Provider:

- Local Ollama HTTP API.

Endpoint:

```text
{config.OLLAMA_BASE_URL}/api/generate
```

Model:

- Env variable: `OLLAMA_LLM_MODEL`
- Code fallback: `qwen2.5:14b`

Generation parameters from [georgia_ev_intelligence/shared/config/settings.py](../georgia_ev_intelligence/shared/config/settings.py):

| Parameter | Env variable | Default in code |
|---|---|---:|
| Model | `OLLAMA_LLM_MODEL` | `qwen2.5:14b` |
| Temperature | `OLLAMA_TEMPERATURE` | `0.1` |
| Top-p | `OLLAMA_TOP_P` | `0.9` |
| Num predict | `OLLAMA_NUM_PREDICT` | `4096` |
| HTTP timeout | Function/CLI argument | `180` seconds |

Request body:

```python
{
    "model": model,
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": config.OLLAMA_TEMPERATURE,
        "top_p": config.OLLAMA_TOP_P,
        "num_predict": config.OLLAMA_NUM_PREDICT,
    },
}
```

Prompt templates:

- Current final-answer prompt: `PROMPT_TEMPLATE` in [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50.py)
- Strict context-only prompt: `ONLY_RAG_PROMPT_TEMPLATE` in [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/only_rag_pipeline.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/only_rag_pipeline.py)
- No-context prompt: `ONLY_PRETRAINED_PROMPT_TEMPLATE` in [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/only_pretrained_pipeline.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/only_pretrained_pipeline.py)

Retrieved parent context formatting:

```python
"\n\n".join(
    parent.parent_chunk_text
    for parent in parent_contexts
    if parent.parent_chunk_text
)
```

This means the LLM receives reranked `parent_chunk_text` values, not raw child chunks.

Answer cleaning:

- `_clean_answer(answer)` strips whitespace.
- Removes `<think>...</think>` blocks with a regex.
- Removes accidental surrounding markdown code fences for `text` or `markdown`.

Returned answer:

- `generate_answer()` returns the cleaned `response` field from the Ollama JSON response.

## 22. Final Excel Output Generation

Main answer script: [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50.py)

Defaults:

- Input workbook: `kb/Human validated 50 questions.xlsx`
- Input sheet: `Sheet1`
- Output directory: `georgia_ev_intelligence/outputs/hybrid_retrieval_human_validated_50/`
- Output filename pattern: `{timestamp}_answers.xlsx`

Output columns:

```text
question
golden_answer
retrieved_parent_chunks_after_reranking
final_llm_answer
sparse_child_count
dense_child_count
merged_child_result_count
unique_child_chunk_count
unique_parent_id_count
parent_context_count_before_rerank
parent_context_count_after_rerank
```

Question column candidates:

- `question`
- `Question`

Golden answer column candidates:

- `golden_answer`
- `Golden Answer`
- `answer`
- `Answer`
- `human_validated_answer`
- `Human Validated Answer`
- `Human validated answers`
- `validated_answer`

Question loading:

- `_load_questions(input_path, sheet_name)` reads the workbook with `pd.read_excel`.
- `_question_column_map()` maps workbook-specific column names.
- `QuestionRow` stores `serial_number`, `question`, `golden_answer`.

Final output fields:

- `golden_answer` comes from the workbook.
- `retrieved_parent_chunks_after_reranking` is created by `_format_retrieved_context(parent_contexts)` after reranking.
- `final_llm_answer` comes from `generate_answer(prompt, timeout=args.llm_timeout)`.
- Trace columns are included.
- Child contexts are not included separately in `run_rewritten_50.py`; they are included by retrieval-only and all-modes scripts.

The script writes incrementally after each question through `_write_output(output_path, output_rows)`.

## 23. Evaluation / Batch Runs

| Pipeline | Script path | Purpose | Input | Output | What it measures/records |
|---|---|---|---|---|---|
| Current answer-generation batch | [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50.py) | Runs active hybrid retrieval plus final LLM answer generation. Despite the filename, the default workbook is the human-validated 50 workbook. | Defaults to [kb/Human validated 50 questions.xlsx](../kb/Human%20validated%2050%20questions.xlsx), `Sheet1`; supports `--input`, `--sheet`, `--limit`. | Timestamped `{timestamp}_answers.xlsx` under `georgia_ev_intelligence/outputs/hybrid_retrieval_human_validated_50/` unless `--output` is provided. | Golden answer, reranked parent context, final LLM answer, trace counts. |
| Retrieval-only mode | [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50_retrieval_only.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50_retrieval_only.py) | Runs retrieval without LLM generation. | Same default question workbook and options as current batch. | Timestamped `{timestamp}_retrieval_only.xlsx`. | Retrieved parent context, dense child context, sparse child context, trace counts, retrieval errors. |
| All-modes comparison | [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50_all_modes.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50_all_modes.py) | Runs retrieval once per question and writes three answer modes. | Same default question workbook and options; `--output-root` creates timestamped child folder. | `only_rag.xlsx`, `only_pre_trained.xlsx`, `rag_plus_pre_trained.xlsx`. | Compares context-only answer generation, no-context answer generation, and current RAG-plus-pretrained prompt. |
| Only RAG | [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/only_rag_pipeline.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/only_rag_pipeline.py) used by all-modes runner | Generates answers constrained to retrieved context only. | Question and retrieved context. | Written by all-modes runner. | Context-grounded answer behavior. |
| Only pretrained | [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/only_pretrained_pipeline.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/only_pretrained_pipeline.py) used by all-modes runner | Generates answers without retrieved context. | Question only. | Written by all-modes runner. | Baseline answer behavior without RAG context. |
| RAG + pretrained/current | `CurrentAnswerPipeline` in [run_rewritten_50_all_modes.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50_all_modes.py) | Uses the current final-answer prompt from `run_rewritten_50.py`. | Question and retrieved context. | `rag_plus_pre_trained.xlsx`. | Current answer-generation behavior. |
| Rewritten 50 workbook | [kb/Rewritten_50_questions.xlsx](../kb/Rewritten_50_questions.xlsx) | Contains 50 rewritten questions. | Sheet `Q&A`, columns `s.no`, `question`, `answer`. | No dedicated active script default found; can be passed via `--input` with matching `--sheet`. | Unclear from active code whether this is currently the preferred evaluation source. |
| Human validated 50 pipeline | `run_rewritten_50.py`, `run_rewritten_50_retrieval_only.py`, `run_rewritten_50_all_modes.py` | Active default evaluation path. | [kb/Human validated 50 questions.xlsx](../kb/Human%20validated%2050%20questions.xlsx). | Answer/retrieval/all-modes workbooks. | Retrieval and answer traces against human-validated answers. |

## 24. Tests

Tests are under [tests/runtime_pipeline/](../tests/runtime_pipeline/).

Pytest status:

```text
10 passed in 0.48s
```

Command used:

```bash
pytest -q -p no:cacheprovider
```

The cache provider was disabled to avoid writing pytest cache metadata during this documentation-only review.

| Test file | What it checks |
|---|---|
| [tests/runtime_pipeline/test_hybrid_retrieval_orchestrator.py](../tests/runtime_pipeline/test_hybrid_retrieval_orchestrator.py) | Active orchestrator retrieves 250 children per retriever by default; reranks deduped parents with top-k 45; final output contains only reranked parent contexts; `_dedupe()` preserves parent fetch order; `ChildResultMerger` preserves retriever order without RRF. |
| [tests/runtime_pipeline/test_run_rewritten_50_retrieval_only.py](../tests/runtime_pipeline/test_run_rewritten_50_retrieval_only.py) | Retrieval-only runner populates final/dense/sparse contexts, loads human-validated workbook column names, creates output parent directory and workbook, records retrieval errors in output fields. |
| [tests/runtime_pipeline/test_run_rewritten_50_all_modes.py](../tests/runtime_pipeline/test_run_rewritten_50_all_modes.py) | All-modes runner populates dense/sparse contexts and writes expected columns for contextual and no-context modes. |

Skipped/ignored tests:

- No skipped tests were reported by the pytest run.

Stale tests:

- No stale tests were identified from the active test files. All discovered tests passed.

## 25. Scripts

No top-level `scripts/` directory exists in the inspected repository tree.

Executable workflows are implemented as Python module entrypoints:

| Module/script path | Purpose | Input | Output | General or evaluation-specific | Hardcoded/default values |
|---|---|---|---|---|---|
| [georgia_ev_intelligence/shared/data/loader.py](../georgia_ev_intelligence/shared/data/loader.py) | Normalize KB and write debug workbook when run as `__main__`. | `kb/GNEM - Auto Landscape Lat Long Updated.xlsx`. | `georgia_ev_intelligence/outputs/Normalized_kb.xlsx`. | General offline preprocessing/debug. | Source path constants `KB_EXCEL_PATH`, `NORMALIZED_KB_PATH`. |
| [georgia_ev_intelligence/offline_pipeline/index_pgvector.py](../georgia_ev_intelligence/offline_pipeline/index_pgvector.py) | Build parent/child chunks, validate, export debug workbooks, store parent and child tables. | KB loaded through `kb_loader.load()`. | `parent_chunks`, `child_chunks`, debug XLSX files. | General indexing. | Debug filenames `parent_chunks.xlsx`, `child_chunks.xlsx`; CLI defaults. |
| [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50.py) | Retrieval + final answer generation batch. | Default human-validated workbook; configurable via CLI. | Timestamped answers workbook. | Evaluation/batch. | `DEFAULT_QUESTIONS_WORKBOOK`, `DEFAULT_QUESTIONS_SHEET`, `DEFAULT_OUTPUT_DIR_NAME`, prompt template, output columns. |
| [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50_retrieval_only.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50_retrieval_only.py) | Retrieval-only traces. | Same default workbook; configurable via CLI. | Timestamped retrieval-only workbook. | Evaluation/debug. | Output columns and default output path. |
| [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50_all_modes.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50_all_modes.py) | Compare Only RAG, Only Pre-Trained, and RAG + Pre-Trained. | Same default workbook; configurable via CLI. | Timestamped folder with three workbooks. | Evaluation/comparison. | Mode filenames, output columns, default output root. |

## 26. Configuration and Environment Variables

Only variables used by active code are included.

| Env variable | Purpose | Required or optional | Default if any | File where used | Runtime/offline/shared |
|---|---|---|---|---|---|
| `NEON_DATABASE_URL` | PostgreSQL/Neon connection URL for parent and child tables. | Required by `_env`. | None in code. | [settings.py](../georgia_ev_intelligence/shared/config/settings.py); used by `postgres_store.py`, `pgvector_store.py`, `bm25_retriever.py`, `dense_pgvector_retriever.py`, `parent_fetcher.py`. | Shared, offline, runtime |
| `OLLAMA_BASE_URL` | Base URL for Ollama API. | Required by `_env`. | None in code. | [settings.py](../georgia_ev_intelligence/shared/config/settings.py), [llm_client.py](../georgia_ev_intelligence/runtime_pipeline/generation/llm_client.py) | Shared, runtime |
| `OLLAMA_LLM_MODEL` | Ollama model for generation. | Optional. | `qwen2.5:14b` | [settings.py](../georgia_ev_intelligence/shared/config/settings.py), [llm_client.py](../georgia_ev_intelligence/runtime_pipeline/generation/llm_client.py) | Shared, runtime |
| `OLLAMA_TEMPERATURE` | Ollama generation temperature. | Optional. | `0.1` | [settings.py](../georgia_ev_intelligence/shared/config/settings.py), [llm_client.py](../georgia_ev_intelligence/runtime_pipeline/generation/llm_client.py) | Shared, runtime |
| `OLLAMA_TOP_P` | Ollama generation top-p. | Optional. | `0.9` | [settings.py](../georgia_ev_intelligence/shared/config/settings.py), [llm_client.py](../georgia_ev_intelligence/runtime_pipeline/generation/llm_client.py) | Shared, runtime |
| `OLLAMA_NUM_PREDICT` | Ollama max prediction/token budget option. | Optional. | `4096` | [settings.py](../georgia_ev_intelligence/shared/config/settings.py), [llm_client.py](../georgia_ev_intelligence/runtime_pipeline/generation/llm_client.py) | Shared, runtime |
| `EMBEDDING_MODEL` | SentenceTransformer model ID for document and query embeddings. | Required by `_env`. | None in code; `.env.example` uses `nomic-ai/nomic-embed-text-v1.5`. | [settings.py](../georgia_ev_intelligence/shared/config/settings.py), [embeddings.py](../georgia_ev_intelligence/shared/embeddings.py), `pgvector_store.py`, `dense_pgvector_retriever.py`, `index_pgvector.py` | Shared, offline, runtime |
| `EMBEDDING_LOCAL_FILES_ONLY` | Passed to `SentenceTransformer(local_files_only=...)`. | Required by `_env_bool`. | None in code. | [settings.py](../georgia_ev_intelligence/shared/config/settings.py), [embeddings.py](../georgia_ev_intelligence/shared/embeddings.py) | Shared |
| `EMBEDDING_TRUST_REMOTE_CODE` | Passed to `SentenceTransformer(trust_remote_code=...)`. | Required by `_env_bool`. | None in code. | [settings.py](../georgia_ev_intelligence/shared/config/settings.py), [embeddings.py](../georgia_ev_intelligence/shared/embeddings.py) | Shared |
| `EMBEDDING_DOCUMENT_PREFIX` | Prefix applied to child document embedding text. | Required by `_env`. | None in code; `.env.example` uses `search_document:`. | [settings.py](../georgia_ev_intelligence/shared/config/settings.py), [embeddings.py](../georgia_ev_intelligence/shared/embeddings.py), [pgvector_store.py](../georgia_ev_intelligence/offline_pipeline/pgvector_store.py) | Shared, offline |
| `EMBEDDING_QUERY_PREFIX` | Prefix applied to runtime query text before embedding. | Required by `_env`. | None in code; `.env.example` uses `search_query:`. | [settings.py](../georgia_ev_intelligence/shared/config/settings.py), [embeddings.py](../georgia_ev_intelligence/shared/embeddings.py), [dense_pgvector_retriever.py](../georgia_ev_intelligence/runtime_pipeline/retrieval/dense_pgvector_retriever.py) | Shared, runtime |
| `PGVECTOR_BATCH_SIZE` | Offline embedding and upsert batch size. | Required by `_env_int`. | None in code; `.env.example` uses `64`. | [settings.py](../georgia_ev_intelligence/shared/config/settings.py), [pgvector_store.py](../georgia_ev_intelligence/offline_pipeline/pgvector_store.py) | Shared, offline |
| `HYBRID_RETRIEVER_TOP_K` | Top-k passed to each child retriever. | Optional via `os.environ.get`. | `250` | [hybrid_retrieval/config.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/config.py), [orchestrator.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/orchestrator.py) | Runtime |
| `HYBRID_RERANKER_TOP_K` | Parent reranker final top-k. | Optional via `os.environ.get`. | `45` | [hybrid_retrieval/config.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/config.py), [orchestrator.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/orchestrator.py) | Runtime |
| `HYBRID_RERANKER_MODEL` | Cross-encoder model name. | Optional via `os.environ.get`. | `cross-encoder/ms-marco-MiniLM-L12-v2` | [hybrid_retrieval/config.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/config.py), [factory.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/factory.py), [reranker.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/reranker.py) | Runtime |

## 27. Models Used

| Model | Type | Used for | File/config location | Local/API | CPU/GPU relevance | Notes |
|---|---|---|---|---|---|---|
| `EMBEDDING_MODEL`, example `nomic-ai/nomic-embed-text-v1.5` | SentenceTransformer embedding model | Offline child embeddings and runtime query embeddings | [settings.py](../georgia_ev_intelligence/shared/config/settings.py), [.env.example](../.env.example), [embeddings.py](../georgia_ev_intelligence/shared/embeddings.py) | Local model through `sentence_transformers` | CPU/GPU behavior is controlled by SentenceTransformers/PyTorch environment; no explicit device is set in code. | Required env variable; exact dimension is discovered dynamically. |
| `cross-encoder/ms-marco-MiniLM-L12-v2` | SentenceTransformers CrossEncoder | Parent reranking in active flow; child reranking method also exists but is not active in orchestrator | [hybrid_retrieval/config.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/config.py), [reranker.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/reranker.py) | Local model through `sentence_transformers.CrossEncoder` | No explicit device is set in code. | Env override `HYBRID_RERANKER_MODEL`. |
| `qwen2.5:14b` | Ollama LLM | Final answer generation | [settings.py](../georgia_ev_intelligence/shared/config/settings.py), [llm_client.py](../georgia_ev_intelligence/runtime_pipeline/generation/llm_client.py), [.env.example](../.env.example) | Local Ollama HTTP API | Hardware use depends on Ollama runtime; code does not set CPU/GPU. | Env override `OLLAMA_LLM_MODEL`; code fallback is `qwen2.5:14b`. |
| Anthropic model | Unclear/not active | None found in active code | `anthropic>=0.40` appears in [requirements.txt](../requirements.txt), but no active import/use was found. | N/A | N/A | No Anthropic model name is configured or used by active code. |

## 28. Important Formulas and Scoring

### BM25

Code location: [georgia_ev_intelligence/runtime_pipeline/retrieval/bm25_retriever.py](../georgia_ev_intelligence/runtime_pipeline/retrieval/bm25_retriever.py)

Implementation:

```python
self._bm25 = BM25Okapi(corpus_tokens)
scores = self._bm25.get_scores(query_tokens)
```

Conceptual formula:

```text
score(D, Q) = sum over query terms of IDF(q) * term-frequency saturation
```

Exact formula details are provided by the `rank_bm25` library, not by this codebase.

### Dense Embedding Similarity / pgvector

Code locations:

- [georgia_ev_intelligence/offline_pipeline/pgvector_store.py](../georgia_ev_intelligence/offline_pipeline/pgvector_store.py)
- [georgia_ev_intelligence/runtime_pipeline/retrieval/dense_pgvector_retriever.py](../georgia_ev_intelligence/runtime_pipeline/retrieval/dense_pgvector_retriever.py)

Documents and queries are encoded with `normalize_embeddings=True`.

Search SQL:

```sql
ORDER BY embedding <=> %s::vector
LIMIT %s;
```

Index:

```sql
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100)
```

Conceptually, lower cosine distance means closer vector similarity.

### Cross-Encoder Reranking

Code location: [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/reranker.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/reranker.py)

Conceptual formula:

```text
score = CrossEncoder(query, parent_chunk_text)
```

Sorting:

```python
sorted(..., key=lambda item: item[1], reverse=True)
```

Top-k selection:

```python
scored_parents[:top_k]
```

### Deduplication Logic

Child dedupe:

```text
first occurrence by chunk_id wins
```

Implemented in `ChildResultMerger.merge()`.

Parent dedupe:

```text
first occurrence by parent_record_id wins
```

Implemented in `fetch_parents()` through `_dedupe()`.

### Normalization

Data normalization:

- Column names normalized by `_norm_column`.
- Missing values normalized to `"Unknown"`.
- Numeric fields parsed by `clean_numeric`.

Embedding normalization:

- `normalize_embeddings=True` offline and runtime.

## 29. Performance Bottlenecks

Likely slow parts from code structure:

- LLM generation: `generate_answer()` calls Ollama synchronously per question with default timeout 180 seconds.
- Prompt/context size: up to `HYBRID_RERANKER_TOP_K` parent chunks are joined into one prompt; default is 45.
- Cross-encoder reranking: `CrossEncoder.predict()` scores every fetched parent candidate before slicing top-k.
- Embedding generation: offline `index_kb_children()` encodes all child chunks in batches; runtime `DensePgvectorRetriever.search()` embeds each query.
- DB query and connection overhead: each BM25 load, dense search, parent fetch, and storage call uses direct `psycopg2.connect()`; no connection pooling is implemented.
- BM25 loading: `BM25Retriever._load()` loads all rows from `child_chunks` and builds an in-memory `BM25Okapi` index on first use.
- Excel writing: batch runners write output workbooks after each row, which can add overhead for full runs.

Timing/logging:

- The code prints progress lines like `[index/total] question` in batch runners.
- No structured timing trace or per-stage latency fields were found.

## 30. Current Limitations

Limitations visible from active code:

- No RRF implementation in active hybrid retrieval; merge is ordered `chunk_id` dedupe followed by parent reranking.
- Child retrieval scores are not preserved in `RetrievedChildChunk`; BM25 scores are filtered/sorted but not returned, and dense SQL does not select distance.
- Parent reranker scores are not returned in final `ParentContext` output.
- No connection pooling; code opens direct `psycopg2` connections per storage/search/fetch operation.
- No structured timing trace for retrieval, reranking, generation, or Excel writing.
- Default batch runner names include `run_rewritten_50`, but the active default input is `Human validated 50 questions.xlsx`; this can confuse presentation/runbook naming.
- `scripts/` directory requested in the desired report structure does not exist.
- `RAG_Data_Management_Framework.xlsx` exists but no active code references it by path.
- `shared/data/schema.py` builds schema metadata, but the active hybrid retrieval path does not import it.
- Dense retrieval uses pgvector `LIMIT` but does not return distance/similarity for diagnostics.
- BM25 loads the full child table into memory on first use.
- Output workbooks are rewritten after each processed row.
- Embedding dimension is discovered dynamically, but changing models requires `--recreate-child-table` to rebuild `child_chunks` with the new vector dimension.

Items not found as limitations:

- Stale tests: no stale tests were identified; all discovered tests passed.
- Old docs: the inspected README and project structure docs describe the active PostgreSQL + pgvector pipeline.

## 31. Future Improvements

Prioritized suggestions, kept dynamic/config-driven:

1. Add structured timing fields to `HybridRetrievalTrace` or a separate run trace: BM25 time, dense time, parent fetch time, rerank time, LLM time, Excel write time.
2. Include optional diagnostic scores in retrieval outputs: BM25 score, dense distance, and parent rerank score, without changing the core retrieval decision unless needed.
3. Add connection management or lightweight pooling around PostgreSQL access if repeated batch runs become slow.
4. Make output writing configurable: write after each row for safety or write every N rows for speed.
5. Clarify runner naming or defaults so `run_rewritten_50.py` and the human-validated default workbook do not appear contradictory.
6. Add tests for `DensePgvectorRetriever` SQL shape and `BM25Retriever` tokenization using mocks, avoiding live database requirements.
7. Add an explicit report/evaluation harness that computes retrieval metrics if a gold set of expected records exists. The expected records should come from data files, not hardcoded answers.
8. Optional only: evaluate RRF as an experimental branch/config option if future metrics show merge + dedupe is insufficient. RRF is not part of the current active pipeline and is not required by current tests.
9. Add documentation for `RAG_Data_Management_Framework.xlsx` if it is intended to become an active governance artifact.

## 32. PPT Slide Outline

### Slide 1: Project Goal

- Georgia EV supply-chain question answering from a structured KB.
- RAG pipeline using PostgreSQL, pgvector, BM25, cross-encoder reranking, and Ollama.
- Final output is Excel-based for review/evaluation.
- Suggested visual: one-line pipeline diagram.

### Slide 2: Data Source

- Source workbook: `kb/GNEM - Auto Landscape Lat Long Updated.xlsx`.
- Loader normalizes column names, missing values, and numeric fields.
- Evaluation inputs include human-validated and rewritten 50-question workbooks.
- Suggested visual: table of input/output workbooks.

### Slide 3: Offline Pipeline

- Load Excel KB.
- Create one parent record per normalized KB row.
- Create five child chunks per parent.
- Validate relationships, export debug XLSX, store in PostgreSQL/pgvector.
- Suggested diagram: offline flow arrow chart.

### Slide 4: Parent-Child Chunking

- Parent chunk = full row context for LLM.
- Child chunks = retrieval-focused slices.
- Child types: identity, product role, OEM relationship, location/employment, classification.
- `record_id` and `chunk_id` preserve linkage.
- Suggested visual: one parent box with five child boxes.

### Slide 5: PostgreSQL + pgvector Storage

- `parent_chunks` stores structured fields and `parent_chunk_text`.
- `child_chunks` stores metadata and `VECTOR({vector_size})` embeddings.
- pgvector index: `ivfflat`, `vector_cosine_ops`, `lists = 100`.
- Dense search uses `embedding <=> query_vector`.
- Suggested visual: two-table schema.

### Slide 6: Runtime Retrieval

- BM25 and dense child retrievers run in parallel.
- Results merge and dedupe by `chunk_id`.
- Children map to deduped parent records.
- Parent contexts are reranked before generation.
- Suggested diagram: runtime flow.

### Slide 7: BM25 Sparse Retrieval

- Loads all `child_chunks` metadata from PostgreSQL.
- Builds structured text from `chunk_type` and metadata.
- Tokenizer handles lowercase, possessives, hyphen/slash compounds.
- Default active top-k: 250 per hybrid config.
- Suggested visual: tokenization and scoring schematic.

### Slide 8: Dense Retrieval

- Query encoded with `EMBEDDING_MODEL` and `EMBEDDING_QUERY_PREFIX`.
- Searches child embeddings in pgvector.
- Uses cosine operator/index.
- Returns child metadata, not distances.
- Suggested visual: query vector nearest-neighbor search.

### Slide 9: Hybrid Merge + Dedupe

- Current hybrid strategy is merge + dedupe, not RRF.
- First-seen `chunk_id` wins.
- Retriever order is BM25 then dense in default factory.
- Tests assert no-RRF ordered behavior.
- Suggested visual: two candidate lists merging into unique child list.

### Slide 10: Parent Mapping

- Every child carries `parent_record_id`.
- Duplicate parents can arise from multiple matching child chunks.
- `fetch_parents()` dedupes parent IDs while preserving first-seen order.
- Fetches `record_id`, `source_row_id`, `parent_chunk_text`.
- Suggested visual: child IDs converging to parent IDs.

### Slide 11: Cross-Encoder Reranking

- Model default: `cross-encoder/ms-marco-MiniLM-L12-v2`.
- Active reranking is parent-level.
- Score concept: `CrossEncoder(query, parent_chunk_text)`.
- Final top-k default: 45 parent contexts.
- Suggested visual: query paired with candidate parent chunks.

### Slide 12: LLM Generation

- Provider: local Ollama.
- Model default: `qwen2.5:14b`.
- Receives reranked parent chunks, not raw child chunks.
- Cleans thinking tags and accidental code fences.
- Suggested visual: prompt assembly into answer.

### Slide 13: Final Excel Output

- `run_rewritten_50.py` writes answer workbook.
- Columns include question, golden answer, reranked context, final answer, trace counts.
- Golden answer comes from workbook.
- Final answer comes from LLM.
- Suggested visual: output workbook column table.

### Slide 14: Evaluation Modes

- Retrieval-only.
- Only RAG.
- Only Pre-Trained.
- RAG + Pre-Trained/current.
- All-modes comparison writes three workbooks.
- Suggested visual: mode comparison matrix.

### Slide 15: Current Limitations

- No RRF in active pipeline.
- Scores not preserved in output objects.
- No connection pooling or timing trace.
- Dense distances and rerank scores not exported.
- Suggested visual: limitations table with impact.

### Slide 16: Future Work

- Add timing trace.
- Export optional scores.
- Improve DB connection handling.
- Add retrieval metrics harness if gold record mappings exist.
- Optional experimental RRF only if future metrics justify it.
- Suggested visual: priority roadmap.

## 33. Appendix

### Glossary

| Term | Meaning in this codebase |
|---|---|
| KB | Knowledge base workbook under `kb/`. |
| Parent chunk | Full row-level context stored in `parent_chunks.parent_chunk_text`. |
| Child chunk | Smaller retrieval slice stored in `child_chunks` with metadata and embedding. |
| BM25 | Sparse lexical retrieval over tokenized child metadata text using `rank_bm25.BM25Okapi`. |
| pgvector | PostgreSQL extension used to store/search vector embeddings. |
| Dense retrieval | Semantic retrieval over child embeddings using pgvector cosine distance. |
| Hybrid retrieval | Parallel BM25 + dense child retrieval followed by ordered merge/dedupe and parent reranking. |
| RRF | Reciprocal Rank Fusion; explicitly not implemented in the active pipeline. |
| Cross-encoder | Reranker model that jointly scores `(query, parent_chunk_text)` pairs. |
| Ollama | Local LLM server used by `generate_answer()`. |
| Golden answer | Reference answer read from the evaluation workbook. |

### File Path Index

| Path | Role |
|---|---|
| [README.md](../README.md) | Main project runbook and architecture summary. |
| [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) | Repository architecture guide. |
| [.env.example](../.env.example) | Example environment variables. |
| [requirements.txt](../requirements.txt) | Python dependencies. |
| [kb/GNEM - Auto Landscape Lat Long Updated.xlsx](../kb/GNEM%20-%20Auto%20Landscape%20Lat%20Long%20Updated.xlsx) | Source KB workbook. |
| [kb/Human validated 50 questions.xlsx](../kb/Human%20validated%2050%20questions.xlsx) | Active default QA workbook. |
| [kb/Rewritten_50_questions.xlsx](../kb/Rewritten_50_questions.xlsx) | Rewritten QA workbook. |
| [kb/RAG_Data_Management_Framework.xlsx](../kb/RAG_Data_Management_Framework.xlsx) | Data management framework workbook; no active code reference found. |
| [georgia_ev_intelligence/shared/config/settings.py](../georgia_ev_intelligence/shared/config/settings.py) | Shared settings/env. |
| [georgia_ev_intelligence/shared/data/loader.py](../georgia_ev_intelligence/shared/data/loader.py) | KB loading and normalization. |
| [georgia_ev_intelligence/shared/data/schema.py](../georgia_ev_intelligence/shared/data/schema.py) | Column metadata helper. |
| [georgia_ev_intelligence/shared/embeddings.py](../georgia_ev_intelligence/shared/embeddings.py) | SentenceTransformer loading and prefixes. |
| [georgia_ev_intelligence/offline_pipeline/index_pgvector.py](../georgia_ev_intelligence/offline_pipeline/index_pgvector.py) | Offline indexing entrypoint. |
| [georgia_ev_intelligence/offline_pipeline/postgres_store.py](../georgia_ev_intelligence/offline_pipeline/postgres_store.py) | Parent PostgreSQL storage. |
| [georgia_ev_intelligence/offline_pipeline/pgvector_store.py](../georgia_ev_intelligence/offline_pipeline/pgvector_store.py) | Child pgvector storage. |
| [georgia_ev_intelligence/offline_pipeline/chunking/parent_chunk.py](../georgia_ev_intelligence/offline_pipeline/chunking/parent_chunk.py) | Parent record builder. |
| [georgia_ev_intelligence/offline_pipeline/chunking/child_chunk.py](../georgia_ev_intelligence/offline_pipeline/chunking/child_chunk.py) | Child chunk definitions. |
| [georgia_ev_intelligence/offline_pipeline/chunking/relationship.py](../georgia_ev_intelligence/offline_pipeline/chunking/relationship.py) | Child construction and validation. |
| [georgia_ev_intelligence/offline_pipeline/chunking/operations.py](../georgia_ev_intelligence/offline_pipeline/chunking/operations.py) | High-level chunking/export operations. |
| [georgia_ev_intelligence/runtime_pipeline/schemas.py](../georgia_ev_intelligence/runtime_pipeline/schemas.py) | Runtime dataclasses. |
| [georgia_ev_intelligence/runtime_pipeline/retrieval/bm25_retriever.py](../georgia_ev_intelligence/runtime_pipeline/retrieval/bm25_retriever.py) | BM25 retrieval. |
| [georgia_ev_intelligence/runtime_pipeline/retrieval/dense_pgvector_retriever.py](../georgia_ev_intelligence/runtime_pipeline/retrieval/dense_pgvector_retriever.py) | Dense pgvector retrieval. |
| [georgia_ev_intelligence/runtime_pipeline/retrieval/parent_fetcher.py](../georgia_ev_intelligence/runtime_pipeline/retrieval/parent_fetcher.py) | Parent context fetching. |
| [georgia_ev_intelligence/runtime_pipeline/generation/llm_client.py](../georgia_ev_intelligence/runtime_pipeline/generation/llm_client.py) | Ollama LLM client. |
| [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/config.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/config.py) | Hybrid top-k/reranker config. |
| [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/factory.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/factory.py) | Default pipeline construction. |
| [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/orchestrator.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/orchestrator.py) | Runtime orchestration. |
| [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/merger.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/merger.py) | Child result merge/dedupe. |
| [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/parent_mapper.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/parent_mapper.py) | Child-to-parent mapping. |
| [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/reranker.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/reranker.py) | Cross-encoder reranking. |
| [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50.py) | Current answer batch. |
| [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50_retrieval_only.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50_retrieval_only.py) | Retrieval-only batch. |
| [georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50_all_modes.py](../georgia_ev_intelligence/runtime_pipeline/hybrid_retrieval/run_rewritten_50_all_modes.py) | All-modes batch. |

### Class/Function Index

| Name | Location | Role |
|---|---|---|
| `KBColumns` | `shared/data/loader.py` | Normalized KB column constants. |
| `load()` | `shared/data/loader.py` | Load and normalize source KB workbook. |
| `normalize_dataframe()` | `shared/data/loader.py` | Apply field-level normalizers. |
| `build_debug_report()` | `shared/data/loader.py` | Build debug workbook sheets. |
| `ColumnMeta` | `shared/data/schema.py` | Schema metadata dataclass. |
| `build(df)` | `shared/data/schema.py` | Build column metadata index. |
| `load_sentence_transformer()` | `shared/embeddings.py` | Load SentenceTransformer model. |
| `as_document_text()` | `shared/embeddings.py` | Apply document prefix. |
| `as_query_text()` | `shared/embeddings.py` | Apply query prefix. |
| `ParentRecord` | `offline_pipeline/chunking/parent_chunk.py` | Parent record dataclass. |
| `build_parent_record()` | `offline_pipeline/chunking/parent_chunk.py` | Build one parent from row. |
| `build_parent_chunk_text()` | `offline_pipeline/chunking/parent_chunk.py` | Format parent context text. |
| `ChildChunkType` | `offline_pipeline/chunking/child_chunk.py` | Child chunk enum. |
| `ChildChunk` | `offline_pipeline/chunking/child_chunk.py` | Child chunk dataclass. |
| `build_embedding_text()` | `offline_pipeline/chunking/child_chunk.py` | Build child embedding text. |
| `build_child_metadata()` | `offline_pipeline/chunking/child_chunk.py` | Build child metadata. |
| `build_child_chunk()` | `offline_pipeline/chunking/relationship.py` | Build one child chunk. |
| `build_child_chunks()` | `offline_pipeline/chunking/relationship.py` | Build five children for one parent. |
| `validate_relationships()` | `offline_pipeline/chunking/relationship.py` | Validate parent-child integrity. |
| `ChunkingArtifacts` | `offline_pipeline/chunking/operations.py` | Parent/child artifact container. |
| `build_parent_child_chunks()` | `offline_pipeline/chunking/operations.py` | Build all chunks. |
| `store_parents_postgres()` | `offline_pipeline/postgres_store.py` | Upsert parent records. |
| `index_kb_children()` | `offline_pipeline/pgvector_store.py` | Embed/upsert child chunks. |
| `RetrievedChildChunk` | `runtime_pipeline/schemas.py` | Runtime child result dataclass. |
| `ParentContext` | `runtime_pipeline/schemas.py` | Runtime parent context dataclass. |
| `BM25Retriever` | `runtime_pipeline/retrieval/bm25_retriever.py` | Sparse child retriever. |
| `tokenize_bm25()` | `runtime_pipeline/retrieval/bm25_retriever.py` | BM25 tokenization. |
| `_build_bm25_text()` | `runtime_pipeline/retrieval/bm25_retriever.py` | Child metadata to BM25 text. |
| `DensePgvectorRetriever` | `runtime_pipeline/retrieval/dense_pgvector_retriever.py` | Dense child retriever. |
| `fetch_parents()` | `runtime_pipeline/retrieval/parent_fetcher.py` | Fetch parent contexts. |
| `_dedupe()` | `runtime_pipeline/retrieval/parent_fetcher.py` | Ordered dedupe. |
| `generate_answer()` | `runtime_pipeline/generation/llm_client.py` | Ollama generation. |
| `_clean_answer()` | `runtime_pipeline/generation/llm_client.py` | Clean LLM artifacts. |
| `HybridRetrievalConfig` | `runtime_pipeline/hybrid_retrieval/config.py` | Runtime top-k/model config. |
| `BM25ChildRetriever` | `runtime_pipeline/hybrid_retrieval/bm25_retriever.py` | BM25 wrapper for orchestrator. |
| `DenseChildRetriever` | `runtime_pipeline/hybrid_retrieval/dense_retriever.py` | Dense wrapper for orchestrator. |
| `ChildResultMerger` | `runtime_pipeline/hybrid_retrieval/merger.py` | Merge/dedupe child hits. |
| `ParentChildMapper` | `runtime_pipeline/hybrid_retrieval/parent_mapper.py` | Map children to parent contexts. |
| `CrossEncoderReranker` | `runtime_pipeline/hybrid_retrieval/reranker.py` | Child/parent cross-encoder reranking. |
| `HybridRetrievalOrchestrator` | `runtime_pipeline/hybrid_retrieval/orchestrator.py` | Active retrieval orchestration. |
| `HybridRetrievalResult` | `runtime_pipeline/hybrid_retrieval/models.py` | Retrieval result container. |
| `HybridRetrievalTrace` | `runtime_pipeline/hybrid_retrieval/models.py` | Count trace dataclass. |
| `OnlyRagAnswerPipeline` | `runtime_pipeline/hybrid_retrieval/only_rag_pipeline.py` | Context-only answer mode. |
| `OnlyPretrainedAnswerPipeline` | `runtime_pipeline/hybrid_retrieval/only_pretrained_pipeline.py` | No-context answer mode. |
| `CurrentAnswerPipeline` | `runtime_pipeline/hybrid_retrieval/run_rewritten_50_all_modes.py` | Current prompt answer mode. |

### Config Variable Index

| Variable/constant | Location | Value/source |
|---|---|---|
| `ROOT` | `shared/config/settings.py` | Repository root derived from file path. |
| `PACKAGE_DIR` | `shared/config/settings.py` | `ROOT / "georgia_ev_intelligence"`. |
| `KB_DIR` | `shared/config/settings.py` | `ROOT / "kb"`. |
| `OUTPUTS_DIR` | `shared/config/settings.py` | `PACKAGE_DIR / "outputs"`. |
| `GNEM_EXCEL` | `shared/config/settings.py` | `OUTPUTS_DIR / "Normalized_kb.xlsx"`. |
| `HUMAN_QA_EXCEL` | `shared/config/settings.py` | `KB_DIR / "Human validated 50 questions.xlsx"`. |
| `SMOKE_TEST_OUTPUTS_DIR` | `shared/config/settings.py` | `OUTPUTS_DIR / "smoke_test"`. |
| `NEON_DATABASE_URL` | `shared/config/settings.py` | Required env. |
| `OLLAMA_BASE_URL` | `shared/config/settings.py` | Required env. |
| `OLLAMA_LLM_MODEL` | `shared/config/settings.py` | Env or `qwen2.5:14b`. |
| `OLLAMA_TEMPERATURE` | `shared/config/settings.py` | Env or `0.1`. |
| `OLLAMA_TOP_P` | `shared/config/settings.py` | Env or `0.9`. |
| `OLLAMA_NUM_PREDICT` | `shared/config/settings.py` | Env or `4096`. |
| `EMBEDDING_MODEL` | `shared/config/settings.py` | Required env. |
| `EMBEDDING_LOCAL_FILES_ONLY` | `shared/config/settings.py` | Required env bool. |
| `EMBEDDING_TRUST_REMOTE_CODE` | `shared/config/settings.py` | Required env bool. |
| `EMBEDDING_DOCUMENT_PREFIX` | `shared/config/settings.py` | Required env. |
| `EMBEDDING_QUERY_PREFIX` | `shared/config/settings.py` | Required env. |
| `PGVECTOR_BATCH_SIZE` | `shared/config/settings.py` | Required env int. |
| `RETRIEVER_TOP_K` | `hybrid_retrieval/config.py` | Env `HYBRID_RETRIEVER_TOP_K` or `250`. |
| `RERANKER_TOP_K` | `hybrid_retrieval/config.py` | Env `HYBRID_RERANKER_TOP_K` or `45`. |
| `RERANKER_MODEL` | `hybrid_retrieval/config.py` | Env `HYBRID_RERANKER_MODEL` or `cross-encoder/ms-marco-MiniLM-L12-v2`. |
| `DEFAULT_QUESTIONS_WORKBOOK` | `run_rewritten_50.py` | `Human validated 50 questions.xlsx`. |
| `DEFAULT_QUESTIONS_SHEET` | `run_rewritten_50.py` | `Sheet1`. |
| `DEFAULT_OUTPUT_DIR_NAME` | `run_rewritten_50.py` | `hybrid_retrieval_human_validated_50`. |
