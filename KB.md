# GNEM Expanded Knowledge Base — Runbook

This document covers everything needed to crawl web data, store it locally and in Backblaze B2, index it into pgvector, and run the retrieval/answer pipeline.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Configure Credentials](#2-configure-credentials)
3. [One-Time DB Initialisation](#3-one-time-db-initialisation)
4. [Crawler Command Reference](#4-crawler-command-reference)
5. [Backblaze B2 Setup & Verification](#5-backblaze-b2-setup--verification)
6. [Index Web Docs into pgvector](#6-index-web-docs-into-pgvector)
7. [Run the Retrieval & Answer Pipeline](#7-run-the-retrieval--answer-pipeline)
8. [Periodic Re-Crawl (Scheduler)](#8-periodic-re-crawl-scheduler)
9. [File Locations](#9-file-locations)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Environment Setup

```powershell
# Clone / open the repo, then create and activate the venv
python -m venv .venv
.\.venv\Scripts\activate

# Install all dependencies (includes boto3, beautifulsoup4, pandas, pdfplumber, lxml, etc.)
pip install -r requirements.txt
```

---

## 2. Configure Credentials

Copy `.env.example` to `.env` and fill in:

```bash
# Neon PostgreSQL (required for DB indexing)
NEON_DATABASE_URL="postgresql://USER:PASSWORD@HOST/DB?sslmode=require"

# Ollama (required for answer generation only)
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_LLM_MODEL="qwen2.5:32b"

# Embedding model
EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_LOCAL_FILES_ONLY="false"
EMBEDDING_TRUST_REMOTE_CODE="true"
EMBEDDING_DOCUMENT_PREFIX="search_document: "
EMBEDDING_QUERY_PREFIX="search_query: "

PGVECTOR_BATCH_SIZE="64"

# Hybrid retrieval
HYBRID_RETRIEVER_TOP_K="250"
HYBRID_RERANKER_TOP_K="45"
HYBRID_RERANKER_MODEL="cross-encoder/ms-marco-MiniLM-L12-v2"

# Backblaze B2 (optional — leave blank to disable)
B2_KEY_ID="your-25-char-key-id"
B2_APPLICATION_KEY="your-application-key"
B2_BUCKET_NAME="gnem-raw-docs"
B2_ENDPOINT_URL="https://s3.us-east-005.backblazeb2.com"

# Crawler (all optional — these are the defaults)
CRAWLER_CONCURRENCY="5"
CRAWLER_DEPTH="3"
CRAWLER_DELAY_SECONDS="1.0"
CRAWLER_SCHEDULE_CRON="0 2 * * 0"
```

---

## 3. One-Time DB Initialisation

Creates the `raw_documents` table in Neon. Skip if already done.

```powershell
python -m georgia_ev_intelligence.kb_builder --init-db
```

Expected output:
```
raw_documents table ensured in PostgreSQL.
```

---

## 4. Crawler Command Reference

### Source tiers

| `--source` | What it crawls |
|---|---|
| `company` | 205 company websites from the Excel KB `Website` column |
| `news` | 6 curated Georgia EV news / press sites |
| `gov` | 7 government / regulatory sites (DOE, EPA, FHWA, Georgia DCA, etc.) |
| `ddg` | DuckDuckGo search results from `web_queries.md` |
| `all` | All of the above in priority order A → B → C (default) |

### Supported File Formats

The crawler automatically detects and extracts text from the following file types:
- **HTML**: Uses BeautifulSoup to extract body text, preserving links (`[Text](href)`) and images (`[Image: alt](src)`).
- **PDF**: Uses `pdfplumber`.
- **Office**: Uses `python-docx` for Word (`.docx`), and `pandas`/`openpyxl` for Excel (`.xls`, `.xlsx`).
- **Data/Text**: Uses `pandas` for CSV/TSV, and native Python parsers for JSON, XML, and plain text/Markdown.
- **Images**: Automatically identified and stored as `.png` in Backblaze B2.

### Common commands

```powershell
# --- Smoke tests (safe, nothing written) ---

# Dry-run: see what would be fetched from company sites
python -m georgia_ev_intelligence.kb_builder --source company --limit 5 --dry-run

# Dry-run DDG: 3 seeds, depth 1
python -m georgia_ev_intelligence.kb_builder --source ddg --limit 3 --depth 1 --dry-run


# --- Real crawls ---

# Crawl company sites, write JSONL + DB + B2
python -m georgia_ev_intelligence.kb_builder --source company

# Crawl DDG results, JSONL only (skip DB and B2)
python -m georgia_ev_intelligence.kb_builder --source ddg --no-db --no-b2

# Crawl all tiers, depth 2, 10 concurrent requests
python -m georgia_ev_intelligence.kb_builder --source all --depth 2 --concurrency 10

# Crawl government docs only
python -m georgia_ev_intelligence.kb_builder --source gov


# --- Flag reference ---
# --source    company | news | gov | ddg | all
# --depth N   BFS levels per seed domain (default: 3)
# --limit N   Process only first N seed URLs (0 = unlimited)
# --concurrency N  Parallel HTTP requests (default: 5)
# --delay N   Seconds between requests per domain (default: 1.0)
# --dry-run   Fetch+extract but write nothing
# --no-db     Write JSONL only, skip PostgreSQL
# --no-b2     Skip Backblaze B2 upload
# --schedule  Start periodic re-crawl (blocks until Ctrl-C)
# --init-db   Create raw_documents table, then exit
```

---

## 5. Backblaze B2 Setup & Verification

### Create credentials

1. Log in at [backblaze.com](https://www.backblaze.com) → **B2 Cloud Storage**
2. **Buckets** → **Create a Bucket**: name it `gnem-raw-docs`, set to **Private**
3. Note the **Endpoint** shown on the bucket detail page (e.g. `https://s3.us-east-005.backblazeb2.com`)
4. **App Keys** → **Add a New Application Key**:
   - Name: `gnem-crawler`
   - Bucket: `gnem-raw-docs`
   - Permissions: **Read and Write**
5. Copy **keyID** (25 chars) and **applicationKey** immediately — the key is shown only once
6. Paste both into `.env`

### Test the connection

```powershell
.\.venv\Scripts\python.exe -c "
import sys; sys.path.insert(0, '.')
from georgia_ev_intelligence.shared import config
import boto3
c = boto3.client('s3',
    endpoint_url=config.B2_ENDPOINT_URL,
    aws_access_key_id=config.B2_KEY_ID,
    aws_secret_access_key=config.B2_APPLICATION_KEY)
r = c.list_objects_v2(Bucket=config.B2_BUCKET_NAME, MaxKeys=10)
print('Connected! Objects in bucket:', r.get('KeyCount', 0))
for obj in r.get('Contents', []):
    print(' ', obj['Key'], '-', obj['Size'], 'bytes')
"
```

Expected output (empty bucket):
```
Connected! Objects in bucket: 0
```

### What B2 stores

```
gnem-raw-docs/
  raw-html/<sha256>.html     ← full HTML source of every crawled page
  raw-pdf/<sha256>.pdf       ← PDFs
  raw-docx/<sha256>.docx     ← DOCX files
  raw-image/<sha256>.png     ← Image files
  raw-excel/<sha256>.xlsx    ← Excel files
  raw-csv/<sha256>.csv       ← CSV/TSV files
  raw-json/<sha256>.json     ← JSON files
  raw-xml/<sha256>.xml       ← XML files
  raw-text/<sha256>.txt      ← Text/Markdown files
  jsonl/
    company_sites.jsonl      ← synced at end of each crawl run
    ddg_search.jsonl
    gov_docs.jsonl
    news.jsonl
```

Each object has S3 metadata: `url`, `source_type`, `crawled_at`, `linked_company_id`.

### Verify after a crawl

```powershell
# Run a real crawl (3 DDG seeds, depth 1)
python -m georgia_ev_intelligence.kb_builder --source ddg --limit 3 --depth 1
```

You should see in the terminal:
```
[crawl] Done. Documents written: N
[b2]    Synced 1 JSONL shard(s) to Backblaze B2.
```

And in the log lines:
```
INFO  B2 ← raw-html/abc123...html  (45821 bytes)
```

Then check in the B2 console: **Buckets → gnem-raw-docs → Browse Files**.

---

## 6. Index Web Docs into pgvector

After crawling, index the new documents into `parent_chunks` / `child_chunks`:

```powershell
# Index only new web docs (raw_documents rows with ingestion_status='new')
python -m georgia_ev_intelligence.offline_pipeline.index_pgvector --source web

# Index both Excel KB and web docs in one pass
python -m georgia_ev_intelligence.offline_pipeline.index_pgvector --source all

# Index Excel KB only (original default behaviour)
python -m georgia_ev_intelligence.offline_pipeline.index_pgvector

# Process up to 1000 new web docs per run
python -m georgia_ev_intelligence.offline_pipeline.index_pgvector --source web --web-batch 1000
```

Expected output:
```
Fetched 34 new web documents from raw_documents table.
Stored 34 parent chunks in PostgreSQL (parent_chunks table).
Indexed 170 child chunks into pgvector (child_chunks table) with 768-dim vectors.
Marked 34 web docs as indexed in raw_documents.
```

---

## 7. Run the Retrieval & Answer Pipeline

All commands require `.env` with `NEON_DATABASE_URL` set. Answer generation also requires Ollama running.

```powershell
# Start Ollama (separate terminal)
ollama serve
ollama pull qwen2.5:32b
```

### Retrieval only (no LLM, fast)

```powershell
# Smoke test: first 5 questions
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50_retrieval_only --limit 5

# Full 50-question run
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50_retrieval_only
```

Output: `georgia_ev_intelligence/outputs/hybrid_retrieval_human_validated_50/<timestamp>_retrieval_only.xlsx`

Columns: `s.no`, `question`, `human_validated_answer`, `retrieved_context`, `dense_retrieved_context`, `sparse_retrieved_context`

### Full pipeline with LLM answers

```powershell
# Smoke test: 5 questions
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50 --limit 5

# Full 50-question run
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50
```

Output columns: `question`, `golden_answer`, `retrieved_parent_chunks_after_reranking`, `final_llm_answer`, + retrieval trace counts.

### Three-mode comparison

```powershell
# Smoke test
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50_all_modes --limit 5

# Full run — writes three workbooks
python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_rewritten_50_all_modes
```

Writes: `only_rag.xlsx`, `only_pre_trained.xlsx`, `rag_plus_pre_trained.xlsx` under a timestamped folder.

---

## 8. Periodic Re-Crawl (Scheduler)

The scheduler blocks and fires a full crawl on a cron schedule (default: every Sunday at 02:00 ET).

```powershell
# Start with default schedule (every Sunday 02:00 ET)
python -m georgia_ev_intelligence.kb_builder --schedule

# Override schedule via .env
# CRAWLER_SCHEDULE_CRON="0 2 * * 0"   <- Sunday 02:00
# CRAWLER_SCHEDULE_CRON="0 3 * * 1"   <- Monday 03:00

# Use as a Windows background service:
# Run this in a separate PowerShell window or wrap in nssm/Task Scheduler
```

Each scheduled run:
1. Crawls all seed tiers
2. Writes new documents to JSONL
3. Upserts into `raw_documents` (PostgreSQL)
4. Uploads raw bytes to B2 per-document
5. Syncs all JSONL shards to B2 at the end

---

## 9. File Locations

| Path | Contents |
|---|---|
| `kb/raw_docs/company_sites.jsonl` | Crawled company website documents |
| `kb/raw_docs/ddg_search.jsonl` | DDG search result documents |
| `kb/raw_docs/gov_docs.jsonl` | Government/regulatory documents |
| `kb/raw_docs/news.jsonl` | News/press documents |
| `kb/raw_docs/.dedup.db` | SQLite URL + content-hash dedup cache |
| `georgia_ev_intelligence/kb_builder/web_queries.md` | DDG query definitions (edit to add companies) |
| `georgia_ev_intelligence/outputs/parent_chunks.xlsx` | Debug export of all parent chunks |
| `georgia_ev_intelligence/outputs/child_chunks.xlsx` | Debug export of all child chunks |
| `georgia_ev_intelligence/outputs/hybrid_retrieval_human_validated_50/` | Retrieval + answer output workbooks |

---

## 10. Troubleshooting

### `Documents written: 0` after DDG crawl

The crawler skipped all URLs. Check the INFO logs for the reason:
- `Skipping ... — HTTP 403` → site blocks bots. Add domain to `_BLOCKED_DOMAINS` in `crawler.py`
- `Skipping ... — body too short (N chars)` → trafilatura extracted < 100 chars (thin/JS-rendered page)
- `Skipping ... — duplicate content hash` → content already seen in a previous run

```powershell
# Re-run with explicit logging to see every skip reason
python -m georgia_ev_intelligence.kb_builder --source ddg --limit 5 --dry-run 2>&1
```

### `InvalidAccessKeyId: Malformed Access Key Id`

The B2 key ID in `.env` is incorrect or truncated. A valid B2 key ID is exactly **25 characters**. Regenerate in the B2 console under **App Keys**.

### `WARNING: html - backends do not exist or are disabled`

Your version of `ddgs` no longer has the `html` backend. This is already fixed — the crawler uses `backend="auto"`. If you still see it, reinstall: `pip install -r requirements.txt`.

### Runtime tests fail with `Missing required environment variable: NEON_DATABASE_URL`

Run from the repo root with `.env` present, or set the variable in your shell:
```powershell
$env:NEON_DATABASE_URL = "postgresql://..."
python -m pytest tests/runtime_pipeline/ -v
```

### Reset the dedup cache (re-crawl everything)

```powershell
Remove-Item kb\raw_docs\.dedup.db
```

This forces all previously-seen URLs and content hashes to be re-crawled on the next run.