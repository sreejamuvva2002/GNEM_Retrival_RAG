# kb_builder End-to-End Review

This report reviews the current `georgia_ev_intelligence.kb_builder` module as implemented. It intentionally does not enumerate concrete company names, seed URLs, locations, or domain-specific constants; it describes where those values live and how the code uses them.

## 1. Entry Point

Primary command:

```bash
python -m georgia_ev_intelligence.kb_builder
```

The module entry point is `GNEM_Retrival_RAG/georgia_ev_intelligence/kb_builder/__main__.py`, which imports and calls `main()` from `kb_builder/cli.py`.

Main files/functions:

- `kb_builder/__main__.py`
  - Calls `georgia_ev_intelligence.kb_builder.cli.main()`.
- `kb_builder/cli.py`
  - `main()`: parses CLI arguments.
  - `_build_seeds(source)`: selects seed groups.
  - `_make_crawl_fn(args)`: builds a zero-argument crawl function for one-shot or scheduled execution.
- `kb_builder/crawler.py`
  - `run_crawl(seeds, **kwargs)`: synchronous wrapper.
  - `crawl(...)`: async crawler implementation.
- `kb_builder/scheduler.py`
  - `start(crawl_fn, cron_expr)`: blocking periodic scheduler.

CLI arguments in `kb_builder/cli.py`:

| Argument | Behavior |
|---|---|
| `--source {all,company,news,gov}` | Selects which seed tier to crawl. Default is `all`. |
| `--depth N` | Maximum BFS depth per seed domain. Default comes from `config.CRAWLER_DEPTH`. |
| `--concurrency N` | Maximum concurrent HTTP fetches. Default comes from `config.CRAWLER_CONCURRENCY`. |
| `--delay SECONDS` | Minimum delay between requests to the same domain. Default comes from `config.CRAWLER_DELAY_SECONDS`. |
| `--limit N` | Keeps only the first `N` selected seed records before crawling. `0` means unlimited. |
| `--dry-run` | Fetches and extracts pages, logs what would be written, increments the written counter, but does not write JSONL or PostgreSQL. |
| `--no-db` | Writes JSONL only and skips PostgreSQL upsert. Ignored by `--dry-run` because no writes happen. |
| `--schedule` | Starts the blocking scheduler instead of running once. Cron expression comes from `config.CRAWLER_SCHEDULE_CRON`. |
| `--init-db` | Creates the PostgreSQL `raw_documents` table and exits without crawling. |

Default crawler configuration is in `GNEM_Retrival_RAG/georgia_ev_intelligence/shared/config/settings.py`:

- `RAW_DOCS_DIR`
- `CRAWLER_CONCURRENCY`
- `CRAWLER_DEPTH`
- `CRAWLER_DELAY_SECONDS`
- `CRAWLER_USER_AGENT`
- `CRAWLER_SCHEDULE_CRON`

High-level entry flow:

```text
python -m georgia_ev_intelligence.kb_builder
    |
    v
kb_builder/__main__.py
    |
    v
cli.main()
    |
    +-- --init-db ----> postgres_store.ensure_raw_documents_table()
    |
    +-- --schedule ---> scheduler.start(crawl_fn, cron)
    |
    +-- one-shot ----> crawl_fn()
                         |
                         v
                    crawler.run_crawl()
                         |
                         v
                    crawler.crawl()
```

## 2. Seed URL Loading

Seed loading is implemented in `GNEM_Retrival_RAG/georgia_ev_intelligence/kb_builder/seed_urls.py`.

Main objects/functions:

- `STATIC_COMPANY_SEEDS`: static seed list for curated company/source pages.
- `NEWS_SEEDS`: static seed list for news/press-style sources.
- `GOV_SEEDS`: static seed list for government/regulatory-style sources.
- `_excel_kb_path()`: resolves the preferred workbook path, falling back to the raw workbook path.
- `company_site_seeds()`: returns static company seeds plus URLs discovered from the workbook.
- `all_seeds()`: returns company, news, and government seeds in priority order.
- `_dedupe_seeds(seeds)`: removes duplicate seed URLs while preserving the first occurrence.

Source group selection happens in `cli._build_seeds(source)`:

```text
source=company -> seed_urls.company_site_seeds()
source=news    -> seed_urls.NEWS_SEEDS
source=gov     -> seed_urls.GOV_SEEDS
source=all     -> seed_urls.all_seeds()
```

Workbook-derived seeds:

```text
company_site_seeds()
    |
    v
_excel_kb_path()
    |
    v
pd.read_excel(...)
    |
    v
normalize column names
    |
    +-- choose first column containing "website" or "url"
    +-- choose first column containing "company"
    +-- choose first column containing both "row" and "id"
    |
    v
build seed dicts:
    {url, source_type, linked_company_id}
```

The `company_col` is detected but not currently used. `linked_company_id` is derived only from a row-id-like column if present. If a workbook URL does not start with `http`, the code prefixes `https://`.

Current seed strategy is partly static and partly workbook-driven. The static lists are easy to run but should eventually move to dynamic configuration or database-backed source records so source scope, priorities, labels, allowlists, and refresh cadence can be managed without code edits.

## 3. Crawling Flow

The main crawler is `kb_builder/crawler.py`.

Main functions/classes:

- `crawl(...)`
- `run_crawl(...)`
- `_DomainThrottle`
- `_RobotsCache`
- `_guess_file_type(...)`
- `_is_skippable(...)`
- `_same_domain(...)`
- `_extract_links(...)`

Queueing:

```text
input seeds
    |
    v
asyncio.Queue
    |
    v
(url, depth, source_type, linked_company_id, seed_url)
```

Each seed starts at depth `0`. Child URLs are appended with `depth + 1`.

Depth behavior:

- A page is fetched at its current depth.
- Child links are only discovered from HTML pages.
- Children are only enqueued when `depth < max_depth`.
- Therefore `--depth 0` fetches only seed URLs.
- `--depth 1` fetches seeds and their direct same-domain children.
- The maximum fetched depth is `max_depth`; links from that depth are not expanded.

Concurrency behavior:

- `asyncio.Semaphore(concurrency)` limits simultaneous HTTP GET requests.
- The scheduler loop allows up to `concurrency * 2` active tasks, but each HTTP request still passes through the semaphore.
- `_DomainThrottle(delay)` enforces minimum spacing per exact netloc.
- Robots fetching also uses the same shared `httpx.AsyncClient` but does not run inside the fetch semaphore.

Fetch settings:

- Uses `httpx.AsyncClient`.
- `follow_redirects=True`.
- Timeout is `httpx.Timeout(15.0, connect=10.0)`.
- `http2=True`.
- TLS verification is disabled with `verify=False`.
- Headers include configured user agent and broad accept types.

Retries, timeouts, and failures:

- There is no retry/backoff loop.
- Any exception from `client.get(url)` is logged as a warning.
- Failed URLs are marked seen in the SQLite dedup cache, so later runs will skip them unless the dedup DB is cleared or changed.
- Non-2xx HTTP statuses are not rejected by status. If extraction yields at least 100 characters, the document can be stored with that HTTP status.
- Extractor failures generally return empty text and are skipped.
- Task-level exceptions are caught in the BFS loop and logged as errors; they do not abort the whole crawl.

Internal vs external URLs:

- Seed URLs are always crawled.
- Child links are enqueued only if `_same_domain(seed_url, link)` is true.
- `_same_domain` compares netloc after stripping a leading `www.`.
- External child links are ignored.
- There is no separate subdomain policy, public suffix handling, canonical host mapping, or per-source allowlist/denylist.

Crawl flow:

```text
seed queue
    |
    v
fetch_and_store(url, depth, source_type, linked_company_id, seed_url)
    |
    +-- skip static asset extensions
    +-- skip URL if already seen
    +-- robots.txt check
    +-- per-domain throttle
    +-- HTTP GET
    +-- detect file type
    +-- extract text
    +-- skip empty/short body
    +-- content-hash dedup
    +-- RawDocument
    +-- mark URL/hash seen
    +-- write JSONL and optional DB
    +-- extract same-domain HTML children
```

## 4. URL Exploration

Link extraction is implemented by `_extract_links(html, base_url)` in `crawler.py`.

Extraction behavior:

- Uses a regex over raw HTML for `href="..."` or `href='...'`.
- Resolves relative links with `urllib.parse.urljoin(base_url, href)`.
- Keeps only `http` and `https`.
- Removes URL fragments.
- Does not parse the DOM.
- Does not inspect JavaScript-generated links.
- Does not inspect sitemap XML, RSS feeds, canonical tags, alternate language links, or structured metadata.

Normalization behavior:

- Link fragments are stripped.
- Relative links are converted to absolute links.
- Seed dedup uses `url.rstrip("/").lower()`.
- Runtime dedup does exact URL string matching in SQLite.

Duplicate removal:

- Initial seed dedup: `_dedupe_seeds()` normalizes by lowercase and trailing slash removal.
- Runtime URL dedup: `DedupCache.is_url_seen(url)` uses exact stored URL.
- Runtime content dedup: SHA-256 of `body.strip().lower()`.

Useless URL filtering:

- `_is_skippable(url)` skips URLs whose path ends in selected static asset/archive/media extensions.
- This happens before fetch.
- There is no query-parameter normalization or filtering for tracking parameters, sort parameters, pagination loops, calendar pages, login pages, search pages, print pages, or social-share URLs.

PDF/document detection:

- `_guess_file_type(url, content_type)` classifies as:
  - `pdf` if content type contains `pdf` or URL path ends in `.pdf`.
  - `docx` if content type indicates Office/Word or URL path ends in `.doc`/`.docx`.
  - `html` otherwise.
- The code name is `docx`, but the extension regex also matches legacy `.doc`; both are sent to the DOCX extractor, which may fail on true binary `.doc` files.

## 5. Content Extraction

HTML extraction:

- File: `kb_builder/extractors/html_extractor.py`
- Function: `extract(html_bytes, url="")`
- Library: `trafilatura`
- Output: `(title, body_text)`

Behavior:

- Decodes bytes as UTF-8, Latin-1, or cp1252.
- Calls `trafilatura.extract(...)` with:
  - `include_comments=False`
  - `include_tables=True`
  - `no_fallback=False`
  - `output_format="txt"`
- Extracts title with a regex for `<title>`, then `og:title`.
- Returns plain text only.

PDF extraction:

- File: `kb_builder/extractors/pdf_extractor.py`
- Function: `extract(pdf_bytes)`
- Library: `pdfplumber`
- Title comes from PDF metadata.
- Body is page text joined with double newlines.
- No raw PDF file is persisted.
- Page numbers, tables, headings, images, and coordinates are not stored as structured metadata.

DOCX extraction:

- File: `kb_builder/extractors/docx_extractor.py`
- Function: `extract(docx_bytes)`
- Library: `python-docx`
- Title is first heading paragraph, or first paragraph if no heading exists.
- Body is paragraph text joined with newlines.
- Tables, comments, headers/footers, footnotes, images, and formatting are not preserved.

Structured preservation:

- Tables may be included in HTML text because `trafilatura` is called with `include_tables=True`, but they are flattened into plain text.
- Headings are not separately preserved as a hierarchy.
- Sections are not represented as structured blocks.
- The stored `RawDocument` has a single `body_text` string, not page-level, section-level, heading-level, or table-level records.

Metadata captured:

- URL
- Domain
- Source type
- Title
- Crawled timestamp
- HTTP status
- Language default
- Linked company/source ID if provided by seed
- File type
- Content hash
- Generated document ID

Metadata not captured:

- Final redirected URL separately from requested URL
- MIME type
- Content length
- ETag
- Last-Modified
- HTTP headers
- Fetch duration
- Error details for skipped fetches
- Parent URL/referrer
- Crawl depth
- Seed URL
- Link anchor text
- Extractor version/settings
- Raw HTML/document object key
- Text offsets/page numbers

## 6. Storage

JSONL storage is implemented in `kb_builder/writer.py`.

Main functions:

- `_jsonl_path(raw_docs_dir, source_type)`
- `write_document(doc, raw_docs_dir, db=True)`

JSONL files:

- `company_site` source type -> `company_sites.jsonl`
- `news` source type -> `news.jsonl`
- `gov_doc` source type -> `gov_docs.jsonl`
- Unknown source types -> `other.jsonl`

Each JSONL row is `RawDocument.to_dict()` from `kb_builder/models.py`.

Fields written to JSONL:

- `doc_id`
- `url`
- `domain`
- `source_type`
- `title`
- `body_text`
- `crawled_at`
- `content_hash`
- `http_status`
- `language`
- `linked_company_id`
- `ingestion_status`
- `file_type`

PostgreSQL raw document storage is implemented in `offline_pipeline/postgres_store.py`.

Relevant functions:

- `ensure_raw_documents_table()`
- `upsert_raw_document(doc)`
- `fetch_new_raw_documents(limit=500)`
- `update_raw_doc_status(doc_ids, status, error_detail=None)`

Table: `raw_documents`

Columns:

- `doc_id TEXT PRIMARY KEY`
- `url TEXT NOT NULL`
- `domain TEXT`
- `source_type TEXT`
- `title TEXT`
- `body_text TEXT`
- `crawled_at TIMESTAMPTZ`
- `content_hash TEXT`
- `http_status INT`
- `language TEXT DEFAULT 'en'`
- `linked_company_id TEXT`
- `ingestion_status TEXT DEFAULT 'new'`
- `error_detail TEXT`
- `created_at TIMESTAMPTZ DEFAULT NOW()`
- `updated_at TIMESTAMPTZ DEFAULT NOW()`

Indexes:

- `idx_raw_docs_status` on `ingestion_status`
- `idx_raw_docs_domain` on `domain`
- `idx_raw_docs_url` on `url`

Important storage behavior:

- JSONL is written first.
- PostgreSQL upsert failures are logged as warnings and do not fail the crawl.
- PostgreSQL upsert conflicts on `doc_id`, which is derived from the normalized body text.
- `file_type` is written to JSONL but not stored in the current `raw_documents` schema.
- Raw HTML bytes, raw PDF bytes, raw DOCX bytes, and downloaded files are not stored.
- Clean text and metadata are stored together in JSONL and PostgreSQL.
- The SQLite dedup state is stored separately in `RAW_DOCS_DIR/.dedup.db`.

Storage flow:

```text
RawDocument
    |
    +-- writer.write_document()
            |
            +-- append JSONL shard
            |
            +-- optional postgres_store.upsert_raw_document()
                    |
                    v
               raw_documents
```

## 7. Backblaze / Object Storage

There is no Backblaze B2, S3, or generic object storage integration in the current codebase.

Search indicators checked:

- No Backblaze/B2 references.
- No S3/bucket client code.
- No object-key fields in `RawDocument`.
- No storage configuration in `settings.py` or `.env.example`.
- No raw bytes are persisted after extraction.

Where object storage should be integrated:

```text
fetch response
    |
    +-- persist raw bytes to object storage
    |       object key should be deterministic and source-neutral
    |       metadata should include URL, content hash, crawl timestamp, MIME type
    |
    +-- run extractor
    |
    +-- RawDocument includes:
            raw_object_key
            raw_content_type
            raw_content_length
            raw_sha256
            final_url
```

Best integration point:

- Add an object-storage client module separate from `crawler.py`.
- Call it inside `fetch_and_store(...)` after successful fetch and before extraction.
- Store raw bytes for HTML, PDF, DOCX, and other supported documents.
- Add object references to both JSONL and PostgreSQL.
- Make provider details configurable by environment or database source configuration, not hardcoded in crawler logic.

Suggested schema fields:

- `raw_object_uri`
- `raw_content_type`
- `raw_content_length`
- `raw_sha256`
- `extracted_text_sha256`
- `final_url`
- `fetch_status`
- `fetch_error`
- `extractor_name`
- `extractor_version`

## 8. RAG Readiness

The crawler output is partially RAG-ready.

Ready pieces:

- Extracted text is available as `body_text`.
- Source URL, title, domain, source type, crawl time, HTTP status, language, and content hash are available.
- PostgreSQL `raw_documents` has an ingestion lifecycle via `ingestion_status`.
- `offline_pipeline/index_pgvector.py --source web` can fetch `raw_documents` with `ingestion_status='new'`.
- `offline_pipeline/web_chunk_builder.py` converts raw web docs into parent records.
- `offline_pipeline/pgvector_store.py` stores child embeddings in `child_chunks`.

Current web indexing flow:

```text
raw_documents where ingestion_status='new'
    |
    v
web_chunk_builder.build_parent_record_from_raw_doc()
    |
    v
ParentRecord
    |
    v
chunking.relationship.build_child_chunks()
    |
    v
pgvector_store.index_kb_children()
    |
    v
child_chunks
```

Metadata available for retrieval:

- `doc_id`
- `url`
- `domain`
- `source_type`
- `title`
- `crawled_at`
- `content_hash`
- `linked_company_id`
- Full raw row embedded in `ParentRecord.raw_row`

Weakness in current RAG bridge:

- `web_chunk_builder.build_parent_record_from_raw_doc()` puts URL/title/domain/crawled date into the parent text, but the child chunk builder is reused from the structured workbook path. If child chunk construction expects structured fields, web documents may produce weak or sparse child chunks.
- `child_chunks` stores `metadata` JSON, but retrieval quality depends on what `build_child_chunks()` includes.
- `raw_documents.file_type` is absent in PostgreSQL, even though it exists in JSONL.

Missing fields for strong source traceability:

- `final_url`
- `seed_url`
- `parent_url`
- `crawl_depth`
- `anchor_text`
- `content_type`
- `file_type` in PostgreSQL
- `raw_object_uri`
- `text_object_uri` or extracted-text artifact URI
- `extraction_method`
- `extraction_version`
- `last_modified`
- `etag`
- `retrieved_at`
- `source_config_id`
- `source_priority`
- `source_reliability_score`
- `license_or_terms_hint`
- Page/section/table offsets for citations
- PDF page number mapping
- HTML heading path for each chunk

Recommended RAG-ready design:

```text
source_configs
    |
    v
crawl_jobs
    |
    v
raw_documents
    |
    +-- raw object storage
    +-- extracted text artifacts
    |
    v
document_sections
    |
    v
chunks
    |
    v
chunk_embeddings
```

This separates source governance, crawl runs, raw documents, extracted structured sections, chunks, and embeddings.

## 9. Weaknesses and Risks

URL discovery limitations:

- Regex link extraction misses malformed, dynamically inserted, or script-generated links.
- No sitemap/RSS/feed discovery.
- No canonical URL handling.
- No alternate domain/subdomain policy.
- No anchor text capture.
- No source-specific allowlist/denylist rules.

Filtering limitations:

- Static extension skip list only catches obvious assets.
- Query parameters are not normalized.
- Tracking URLs, sort links, search pages, login pages, calendars, paginated archives, tag pages, and social-share links can create noise.
- No content relevance filter before storage.
- Non-2xx pages can be stored if they contain enough boilerplate text.

Duplicate handling risks:

- Runtime URL dedup uses exact URL strings, so semantically identical URLs with different tracking params are separate.
- Content hash uses all lowercased body text, so small template changes can create new documents.
- Failed fetches are marked seen, which can permanently suppress transient failures.
- `doc_id` is content-hash based, so a changed page creates a new document rather than a new version of the same source URL.

Crawling reliability risks:

- No retry/backoff.
- No per-domain error budget.
- No HTTP status policy.
- TLS verification is disabled.
- Robots fetch failures allow crawling by default.
- No crawl job table or durable queue.
- Process crash loses in-memory queue state.

JavaScript/CAPTCHA risks:

- No browser rendering.
- No JS execution.
- CAPTCHA, consent interstitials, bot-protection pages, and client-rendered content are not handled.
- The crawler may store blocker or boilerplate pages if they produce enough text.

Extraction risks:

- HTML title extraction uses regex rather than DOM/metadata parser.
- HTML output is plain text; heading hierarchy and tables are flattened.
- PDF extraction has no page-level citation mapping.
- DOCX extraction ignores tables and many document parts.
- Legacy Word files may be misclassified as DOCX and fail extraction.

Storage risks:

- No raw source artifact preservation.
- No object storage.
- JSONL and PostgreSQL schemas are not identical; JSONL includes `file_type`, PostgreSQL does not.
- PostgreSQL failures do not fail the crawl, which is pragmatic but can create divergence between JSONL and DB.
- Dedup SQLite is outside PostgreSQL, so distributed or multi-worker crawls would not coordinate cleanly.

Source reliability risks:

- Static source lists encode scope in code.
- No source owner, priority, refresh cadence, trust tier, or allowed content-type policy.
- No source health tracking.
- No review workflow before indexing noisy documents.

## 10. Improvement Recommendations

Recommended source configuration changes:

- Move static seed groups into a `source_configs` table or versioned YAML/JSON config.
- Store source records with source type, priority, allowed domains, max depth, max pages, refresh cadence, content-type policy, URL allow/deny patterns, and reliability tier.
- Link workbook-derived seeds through stable source IDs rather than deriving IDs inside crawler logic.
- Keep source values data-driven; do not encode company names, locations, domains, or project-specific constants in crawler logic.

Recommended crawler changes:

- Add durable `crawl_jobs`, `crawl_frontier`, and `crawl_attempts` tables.
- Store `seed_url`, `parent_url`, `crawl_depth`, `anchor_text`, and `source_config_id`.
- Add retry with exponential backoff and retryable status-code policy.
- Do not mark transient fetch failures as permanently seen.
- Add HTTP status filtering; store failed attempts separately from accepted documents.
- Re-enable TLS verification by default, with per-source override when explicitly configured.
- Make robots failure behavior configurable per source.

Recommended URL normalization/filtering:

- Use a URL canonicalization function shared by seed dedup and runtime dedup.
- Normalize scheme/host casing, trailing slash policy, default ports, fragments, selected query parameters, and percent encoding.
- Drop known tracking parameters through configurable rules.
- Add per-source include/exclude patterns.
- Consider sitemap and feed discovery.
- Use an HTML parser for links and capture anchor text.

Recommended extraction changes:

- Store raw bytes in object storage before extraction.
- Keep extracted text as a separate artifact or table.
- Preserve document structure as sections:
  - heading path
  - section text
  - page number for PDFs
  - table text and table metadata
  - byte/text offsets where practical
- Store extractor name, version, settings, and extraction warnings.

Recommended storage/schema changes:

- Add `file_type` to `raw_documents`.
- Add raw artifact fields: object URI, content type, content length, raw hash.
- Add final URL and redirect chain.
- Add crawl provenance fields: seed URL, parent URL, depth, source config ID, crawl job ID.
- Add versioning by URL plus content hash, so the same URL can have multiple observed versions.
- Keep JSONL as a crash-safe append log, but make PostgreSQL the authoritative query/indexing state.

Recommended object storage integration:

- Add a provider-neutral `object_store.py` interface.
- Configure the provider through environment/database settings.
- Use deterministic object keys based on crawl job ID, normalized URL hash, and raw content hash.
- Store raw HTML/PDF/DOCX and optionally extracted text artifacts.
- Persist object URIs in JSONL and PostgreSQL.

Recommended RAG indexing changes:

- Build a web-specific chunker instead of reusing workbook-oriented child chunk construction directly.
- Chunk by document structure when available.
- Include URL, title, domain, source type, crawl date, file type, source reliability, and section/page metadata in child chunk metadata.
- Add citation fields that let answers point to source URL plus section/page context.
- Add a review or quality gate before marking crawled docs ready for embedding.

## Current End-to-End Diagram

```text
CLI
 |
 v
seed_urls.py
 |
 v
crawler.crawl()
 |
 +-- DedupCache (.dedup.db)
 +-- RobotsCache
 +-- DomainThrottle
 +-- httpx fetch
 |
 v
extractors
 |
 +-- html_extractor -> trafilatura text
 +-- pdf_extractor  -> pdfplumber text
 +-- docx_extractor -> paragraph text
 |
 v
RawDocument
 |
 +-- JSONL shards in RAW_DOCS_DIR
 |
 +-- PostgreSQL raw_documents
          |
          v
 offline_pipeline.index_pgvector --source web
          |
          v
 parent_chunks + child_chunks(pgvector)
```

## Proposed Target Diagram

```text
source_configs
    |
    v
crawl_jobs
    |
    v
durable crawl_frontier
    |
    v
fetch attempts
    |
    +-- raw object storage
    |
    v
raw_documents / document_versions
    |
    v
structured extraction
    |
    v
document_sections
    |
    v
web-aware chunking
    |
    v
chunks + embeddings
    |
    v
retrieval with source/page/section citations
```

## File and Symbol Index

| File | Symbols reviewed |
|---|---|
| `kb_builder/__main__.py` | module entry point |
| `kb_builder/cli.py` | `main`, `_build_seeds`, `_make_crawl_fn` |
| `kb_builder/seed_urls.py` | `STATIC_COMPANY_SEEDS`, `NEWS_SEEDS`, `GOV_SEEDS`, `_excel_kb_path`, `company_site_seeds`, `all_seeds`, `_dedupe_seeds` |
| `kb_builder/crawler.py` | `crawl`, `run_crawl`, `_DomainThrottle`, `_RobotsCache`, `_guess_file_type`, `_is_skippable`, `_same_domain`, `_extract_links` |
| `kb_builder/dedup.py` | `DedupCache`, `_get_conn`, `_db_path` |
| `kb_builder/models.py` | `RawDocument` |
| `kb_builder/writer.py` | `write_document`, `_jsonl_path` |
| `kb_builder/scheduler.py` | `start` |
| `kb_builder/extractors/html_extractor.py` | `extract`, `_extract_title`, `_decode` |
| `kb_builder/extractors/pdf_extractor.py` | `extract` |
| `kb_builder/extractors/docx_extractor.py` | `extract` |
| `shared/config/settings.py` | crawler and storage configuration |
| `offline_pipeline/postgres_store.py` | `ensure_raw_documents_table`, `upsert_raw_document`, `fetch_new_raw_documents`, `update_raw_doc_status` |
| `offline_pipeline/web_chunk_builder.py` | `build_parent_record_from_raw_doc`, `_build_web_parent_text` |
| `offline_pipeline/index_pgvector.py` | `_index_web`, `main` |
| `offline_pipeline/pgvector_store.py` | `index_kb_children` |
