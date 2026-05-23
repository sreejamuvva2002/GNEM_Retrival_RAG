# Expanded Knowledge Base — Walkthrough

## What Was Built

The GNEM RAG system now has a full web data collection pipeline that sits upstream of the existing chunking and indexing infrastructure. The Excel KB is **completely unchanged** — all new web documents flow through a new code path.

---

## Architecture (Implemented)

```
Excel KB (unchanged)               Web Sources (new)
     │                                    │
     ▼                                    ▼
index_pgvector                    kb_builder (crawler)
--source excel (default)          --source company/news/gov/all
     │                                    │
     │                            kb/raw_docs/*.jsonl   ← crash-safe JSONL
     │                                    │
     │                            PostgreSQL: raw_documents
     │                            (ingestion_status: new → indexed)
     │                                    │
     └─────────────────┬──────────────────┘
                       ▼
              parent_chunks + child_chunks
              PostgreSQL + pgvector
                       │
                       ▼
            BM25 + dense retrieval (unchanged)
            cross-encoder rerank   (unchanged)
            Ollama answer gen      (unchanged)
```

---

## Files Changed / Created

### New Files

| File | Purpose |
|---|---|
| `georgia_ev_intelligence/kb_builder/__init__.py` | Package declaration |
| `georgia_ev_intelligence/kb_builder/__main__.py` | `python -m kb_builder` entry |
| `georgia_ev_intelligence/kb_builder/models.py` | `RawDocument` dataclass (auto `doc_id`, `content_hash`) |
| `georgia_ev_intelligence/kb_builder/seed_urls.py` | 3-tier seeds: Excel KB → news → gov |
| `georgia_ev_intelligence/kb_builder/extractors/html_extractor.py` | trafilatura HTML → text |
| `georgia_ev_intelligence/kb_builder/extractors/pdf_extractor.py` | pdfplumber PDF → text |
| `georgia_ev_intelligence/kb_builder/extractors/docx_extractor.py` | python-docx DOCX → text |
| `georgia_ev_intelligence/kb_builder/dedup.py` | SQLite-backed URL + content-hash dedup |
| `georgia_ev_intelligence/kb_builder/writer.py` | JSONL append + PostgreSQL upsert |
| `georgia_ev_intelligence/kb_builder/crawler.py` | Async BFS, robots.txt, rate-limiting |
| `georgia_ev_intelligence/kb_builder/scheduler.py` | APScheduler weekly cron |
| `georgia_ev_intelligence/kb_builder/cli.py` | Full CLI |
| `georgia_ev_intelligence/offline_pipeline/web_chunk_builder.py` | `raw_documents` row → `ParentRecord` |
| `kb/raw_docs/` | Directory for JSONL shards + SQLite dedup DB |
| `tests/kb_builder/test_dedup.py` | 8 dedup unit tests |
| `tests/kb_builder/test_html_extractor.py` | 4 HTML extractor tests |
| `tests/kb_builder/test_seed_urls.py` | 4 seed URL tests |

### Modified Files

| File | Change |
|---|---|
| `georgia_ev_intelligence/shared/config/settings.py` | `RAW_DOCS_DIR` + 5 crawler config vars |
| `georgia_ev_intelligence/offline_pipeline/postgres_store.py` | `raw_documents` table + 4 helper functions |
| `georgia_ev_intelligence/offline_pipeline/index_pgvector.py` | `--source excel\|web\|all`, `--web-batch N` |
| `requirements.txt` | `trafilatura`, `pdfplumber`, `python-docx`, `httpx[http2]`, `apscheduler` |
| `.env.example` | Crawler env var block |

---

## Test Results

```
platform win32 -- Python 3.14.3, pytest-9.0.3

tests/kb_builder/test_dedup.py::test_new_url_is_not_seen              PASSED
tests/kb_builder/test_dedup.py::test_url_marked_seen                  PASSED
tests/kb_builder/test_dedup.py::test_hash_not_seen_initially           PASSED
tests/kb_builder/test_dedup.py::test_hash_marked_seen                  PASSED
tests/kb_builder/test_dedup.py::test_is_duplicate_url_hit              PASSED
tests/kb_builder/test_dedup.py::test_is_duplicate_hash_hit             PASSED
tests/kb_builder/test_dedup.py::test_mark_seen_persists_across_instances PASSED
tests/kb_builder/test_dedup.py::test_no_duplicate_if_nothing_seen      PASSED
tests/kb_builder/test_html_extractor.py::test_extract_returns_nonempty_body PASSED
tests/kb_builder/test_html_extractor.py::test_extract_title            PASSED
tests/kb_builder/test_html_extractor.py::test_extract_empty_html_returns_empty_strings PASSED
tests/kb_builder/test_html_extractor.py::test_extract_body_contains_ev_content PASSED
tests/kb_builder/test_seed_urls.py::test_news_seeds_are_nonempty       PASSED
tests/kb_builder/test_seed_urls.py::test_gov_seeds_are_nonempty        PASSED
tests/kb_builder/test_seed_urls.py::test_all_seeds_priority_order      PASSED
tests/kb_builder/test_seed_urls.py::test_seed_dicts_have_required_keys PASSED

16 passed in 3.37s
```

> [!NOTE]
> The pre-existing `tests/runtime_pipeline/` tests require `NEON_DATABASE_URL` in the environment (live DB connection). They fail during collection without a `.env` file — this behaviour is unchanged from before our work.

---

## How to Run (Quick Reference)

```bash
# Activate venv
.\.venv\Scripts\activate   # Windows

# 1. Create the raw_documents table in Neon (one-time)
python -m georgia_ev_intelligence.kb_builder --init-db

# 2. Smoke test: 5 company seeds, no writes
python -m georgia_ev_intelligence.kb_builder --source company --limit 5 --dry-run

# 3. Full crawl: all tiers (A→B→C), depth 3, JSONL + DB
python -m georgia_ev_intelligence.kb_builder

# 4. Crawl company sites only, skip PostgreSQL (JSONL only)
python -m georgia_ev_intelligence.kb_builder --source company --no-db

# 5. Index new web docs into pgvector (leaves Excel KB rows untouched)
python -m georgia_ev_intelligence.offline_pipeline.index_pgvector --source web

# 6. Index both Excel + web in one pass
python -m georgia_ev_intelligence.offline_pipeline.index_pgvector --source all

# 7. Start periodic re-crawl (every Sunday 02:00 ET, blocks until Ctrl-C)
python -m georgia_ev_intelligence.kb_builder --schedule

# 8. Run kb_builder unit tests
python -m pytest tests/kb_builder/ -v
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| JSONL-first write | Crash-safe; JSONL is the permanent record. DB upsert failures are warnings, not aborts |
| SQLite dedup cache | Persists across runs — essential for periodic re-crawl to not re-index already-seen pages |
| Same-domain BFS | Child links only followed within the seed's own domain to avoid scope creep |
| `robots.txt` per domain | Cached per domain to avoid repeated fetches; allows crawl if robots.txt is unreachable |
| `ingestion_status` lifecycle | `new → indexed / error` allows the indexer to safely batch-process only unprocessed docs |
| `--source excel` default | Existing scripts and CI remain completely unchanged |
