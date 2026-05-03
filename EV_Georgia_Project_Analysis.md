# Georgia EV Supply Chain Intelligence — Full Project Analysis
> **Role: Technical Product Owner | Date: May 2026**
> This document analyzes the project end-to-end: what exists, what is missing, and what must be built — with a clear roadmap.

---

## Table of Contents
1. [Business Context — What Are We Really Building?](#1-business-context)
2. [What Data Do We Have?](#2-data-inventory)
3. [What Has Already Been Built (Phase 1)?](#3-phase-1-analysis)
4. [The Critical Problem — Why Phase 1 Took 3 Hours for 10 Companies](#4-the-speed-problem)
5. [The Full Gap — What Is Missing to Get to the Chatbot](#5-the-gap)
6. [The Target Architecture — GraphRAG for EV Supply Chain](#6-target-architecture)
7. [Phase 2 Blueprint — Chunking, Metadata, Embedding](#7-phase-2-blueprint)
8. [Phase 3 Blueprint — Retrieval & Prompt Strategy](#8-phase-3-blueprint)
9. [The Implementation Roadmap](#9-roadmap)
10. [Key Decisions Needed From You](#10-decisions)

---

## 1. Business Context

### What Is the Business Need?
A **chatbot** that anyone — an economic developer, a journalist, a supply chain analyst, a policy maker — can open and ask natural language questions like:

> *"Which Georgia-based EV suppliers have fewer than 200 employees and supply to Hyundai Metaplant?"*
> *"What is Kia Georgia's total investment in Bryan County as of 2024?"*
> *"Which companies announced new facilities in Jackson County in the last 2 years?"*

The chatbot must answer these questions **accurately**, using **real, verified data** — not LLM guessing.

### Why This Is Hard
Standard chatbots fail here because:
- The data is **highly relational** (companies → suppliers → OEMs → counties → investments)
- Answers require **counting, filtering, and traversing relationships**, not just text similarity
- The data spans **structured Excel/CSV**, **unstructured HTML/PDF**, and **semi-structured JSON**
- New data arrives constantly (news, press releases, SEC filings)

### The Solution Direction (Confirmed by Prior Gemini Analysis)
**GraphRAG** — a hybrid of:
- **Neo4j Knowledge Graph** for structured relational facts (who supplies whom, where, at what scale)
- **Vector Search** for unstructured text (press releases, news, reports)
- **LLM Orchestration** to translate user questions → graph/vector queries → natural language answers

---

## 2. Data Inventory

### 2a. What You Own
| Source | File | Status | What It Contains |
|--------|------|--------|-----------------|
| GNEM Excel | `GNEM - Auto Landscape Lat Long Updated.xlsx` | ✅ Ready | 206 companies, lat/long, county, sector, OEM relationships |
| Human Q&A | `Human validated 50 questions.xlsx` | ✅ Ready | 50 real Q&A pairs validated by domain experts — the **ground truth** for testing |
| RAG Framework Ref | `RAG_Data_Management_Framework.xlsx` | ✅ Ready | Dummy metadata schema reference — defines chunking/tagging approach |
| Master Report | `Master_Data_Enrichment_Report.docx` | ✅ Ready | Architect's plan for data enrichment — design reference |
| Tavily RAG Report | `Kb_Enrichment/tavily_georgia_ev_rag_report.docx` | ✅ Built | Large document with internet-gathered intelligence on Georgia EV ecosystem |

### 2b. What Phase 1 Is Building
Phase 1 (`Kb_Enrichment`) is **actively downloading raw documents** from the internet:
- HTML pages, PDFs, DOCX, JSON, CSV files
- From sources: SEC EDGAR, Georgia.org, Governor press releases, EPA ECHO, Kia Georgia, HMGMA, UGA eMobility, etc.
- Stored locally + uploaded to **Backblaze B2** (cloud object storage)
- Tracked in **PostgreSQL** (or SQLite fallback) via the `urls` metadata table

### 2c. What Is NOT Yet Available
- ❌ No knowledge graph (Neo4j) exists yet
- ❌ No text extraction from raw documents
- ❌ No chunks created
- ❌ No vector embeddings
- ❌ No chatbot interface
- ❌ No enriched company data at scale (10 companies done, 196 remaining)

---

## 3. Phase 1 Analysis — What Was Built

### Architecture Overview
```
Companies List (companies.json)
        ↓
Query Generator (searcher.py)
  → 8 Query Families × 9 Temporal Variants per company
  → ~50-80 queries per company
        ↓
DuckDuckGo Search (ddgs library)
  → URL Discovery with relevance scoring
        ↓
Downloader (downloader.py)
  → Scrapling (browser-based) + HTTPX (direct)
  → SHA256 fingerprint dedup
        ↓
Backblaze B2 (cloud storage)
        ↓
PostgreSQL Tracker (tracker.py)
  → URL registry, convergence tracking, query performance
        ↓
Excel Outputs (framework_updater.py)
  → Download log, registry, processing log
```

### What Is Working Well ✅
1. **Deduplication** — SHA256 fingerprinting prevents re-downloading same content
2. **Convergence tracking** — Knows when a company's queries have "converged" (no new URLs found across 2 consecutive runs), automatically skips completed companies
3. **Checkpoint/resume** — Can resume mid-run after a crash
4. **Domain rate-limiting** — Respects per-domain delays to avoid bans
5. **Priority domains** — georgia.org, sec.gov, energy.gov, hmgma.com, etc. get preference
6. **Multi-mode operation** — `pilot`, `full`, `single-company`, `seed-only`, `retry-only` modes
7. **Cloud storage** — Backblaze B2 with verification and metadata tracking
8. **Fallback stack** — PostgreSQL primary, SQLite local fallback, DuckDB export-only analytics

### The Key Constraint
> `extractor.py` is **intentionally stubbed** — Phase 1 collects raw files only. Phase 2 (extraction, chunking, embedding, RAG) is explicitly out of scope.

---

## 4. The Critical Problem — Why It Took 3 Hours for 10 Companies

### Root Cause Analysis

The pipeline for **1 company** does:
```
8 query families × 9 temporal variants = up to 72 queries per company
DDG rate limit delay: 3 seconds per query
72 × 3 = 216 seconds = ~3.6 minutes PER COMPANY just for searching
+ Download time (scrapling is browser-based = slow)
+ B2 upload time
```

For 206 companies: `206 × 3.6 min = ~12.4 hours` just for search, before any downloads.

### Secondary Problems
1. **DuckDuckGo as sole search source** — DDG has strict rate limits and inconsistent results. Not ideal for 206 × 72 queries.
2. **Scrapling (browser fetcher)** — Heavy for every page. Great for JS-rendered pages but overkill for simple HTML/PDFs.
3. **Sequential company processing** — Companies processed one at a time in the outer loop.
4. **No caching of DDG results** — Re-runs repeat all queries unless marked converged.

### What You Said Is Right
> *"For 10 companies it took 3 hours, for 206 it will take a week"*

That is exactly correct. The pipeline, as-is, **cannot scale** to 206 companies in reasonable time without changes.

---

## 5. The Gap — What Is Missing

### The Full Gap Map
```
Phase 1 (EXISTS, ~80% done)      Phase 2 (MISSING)           Phase 3 (MISSING)
────────────────────────────     ───────────────────────     ─────────────────────
URL Discovery            ✅     Text Extraction         ❌   Neo4j Graph Build  ❌
Raw File Download        ✅     Chunking                ❌   Vector Index       ❌
Cloud Storage (B2)       ✅     Metadata Tagging        ❌   Hybrid Retriever   ❌
PostgreSQL Tracking      ✅     Embedding               ❌   LLM Orchestration  ❌
Convergence Tracking     ✅     Chunk Store             ❌   Chat Interface     ❌
Speed (206 companies)    ❌     Entity Extraction       ❌   Fine-tuning (opt)  ❌
```

### Gap Priority Order
| Priority | Gap | Impact |
|----------|-----|--------|
| 🔴 Critical | Fix Phase 1 speed (3hrs/10 → acceptable/206) | Blocks all downstream |
| 🔴 Critical | Phase 2: Text extraction + chunking pipeline | Needed before RAG |
| 🔴 Critical | Phase 2: Metadata tagging per chunk | Quality of retrieval |
| 🟠 High | Phase 3: Neo4j graph build from GNEM Excel | Enables graph queries |
| 🟠 High | Phase 3: Vector embedding + index | Enables semantic search |
| 🟠 High | Phase 3: Hybrid retriever (graph + vector) | Core RAG capability |
| 🟡 Medium | Phase 3: LLM orchestration layer | User-facing chatbot |
| 🟡 Medium | Phase 3: Chat UI | End user experience |
| 🟢 Optional | Fine-tuning LLM on Cypher | Better graph query generation |

---

## 6. Target Architecture — GraphRAG for EV Supply Chain

### System Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                        USER CHATBOT                             │
│              (Web UI / API / Slack / etc.)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │ Natural Language Question
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM ORCHESTRATOR                             │
│              (LangChain / LangGraph Agent)                      │
│   Routes to → Graph Query OR → Vector Search OR → Both         │
└──────┬──────────────────────────────┬───────────────────────────┘
       │                              │
       ▼                              ▼
┌─────────────────┐        ┌──────────────────────────┐
│   NEO4J GRAPH   │        │    VECTOR STORE           │
│   KNOWLEDGE     │        │  (Qdrant / ChromaDB /     │
│   DATABASE      │        │   Pinecone / Weaviate)    │
│                 │        │                           │
│ Companies       │        │  Chunks from:             │
│ Suppliers       │        │  - PDFs / HTML / DOCX     │
│ OEMs            │        │  - Press releases         │
│ Counties        │        │  - News articles          │
│ Investments     │        │  - SEC filings            │
│ Facilities      │        │  - Government reports     │
└─────────────────┘        └──────────────────────────┘
       ▲                              ▲
       │                              │
┌──────┴──────────────────────────────┴───────────────┐
│              DATA PIPELINE (PHASES 1 & 2)            │
│                                                      │
│  Phase 1 (EXISTS): URL Discovery → Download → B2    │
│                                                      │
│  Phase 2 (BUILD):                                    │
│    Extract Text → Chunk → Tag Metadata →             │
│    Embed → Store in Vector DB                        │
│    + Entity Extract → Load into Neo4j                │
└──────────────────────────────────────────────────────┘
```

### Neo4j Graph Schema (Core Entities)
```
(:Company {
  id, name, type, sector, employee_count,
  annual_revenue, founded_year, website
})

(:Facility {
  id, name, address, city, county, state,
  lat, long, sqft, production_capacity,
  operational_status
})

(:County {
  id, name, state, region
})

(:OEM {
  id, name  // Kia, Hyundai, SK On
})

(:Product {
  id, name, category
  // EV battery cells, stamped parts, wiring harness...
})

(:Investment {
  id, amount_usd, year, announcement_date,
  jobs_created, source_url
})

// RELATIONSHIPS
(:Company)-[:HAS_FACILITY]->(:Facility)
(:Facility)-[:LOCATED_IN]->(:County)
(:Company)-[:SUPPLIES_TO]->(:OEM)
(:Company)-[:SUPPLIES_TO]->(:Company)
(:Company)-[:MANUFACTURES]->(:Product)
(:Company)-[:RECEIVED_INVESTMENT]->(:Investment)
(:Investment)-[:ANNOUNCED_BY]->(:GovernmentEntity)
```

---

## 7. Phase 2 Blueprint — Chunking, Metadata & Embedding

### 7a. Fix Phase 1 Speed First

**Recommendation: Replace DDG-only search with a hybrid approach**

| Source | Use Case | Speed | Quality |
|--------|----------|-------|---------|
| **Tavily API** | Real-time web search per company | Fast (parallel API calls) | High — purpose-built for RAG |
| **SerpAPI / Google Search API** | High-coverage fallback | Medium | High |
| **Direct seed scraping** | Government/OEM sites | Fast | Very High (known good) |
| **GNEM Excel** | Starting structured data | Instant | Authoritative |
| Keep **DDG** | Free fallback only | Slow | Medium |

**Concurrency Fix:**
```yaml
# Current settings.yaml
max_concurrent_downloads: 2       # Too low
ddg_query_delay: 3.0             # Per query

# Recommended
max_concurrent_downloads: 8       # Raise carefully
Use Tavily with async/parallel: 5 companies simultaneously
```

**Estimated improvement:** 206 companies in ~3-4 hours instead of a week.

---

### 7b. Text Extraction (Phase 2 — extractor.py)

The `extractor.py` file is **intentionally empty** (stubbed). You need to fill it.

| File Type | Tool | Notes |
|-----------|------|-------|
| PDF | `pymupdf` (fitz) or `pdfplumber` | Fast, handles scanned PDFs with OCR fallback |
| HTML | `BeautifulSoup` (already installed) | Strip nav/footer, keep article body |
| DOCX | `python-docx` | Already in ecosystem |
| CSV / XLSX | `pandas` | Already installed |
| JSON | Native Python | Parse structured data |

**Rule**: Always extract to **plain Unicode text** + preserve document metadata (source URL, doc_id, company, date).

---

### 7c. Chunking Strategy

> ⚠️ This is the most critical design decision. Wrong chunking = wrong retrieval.

**The Strategy: Contextual Hierarchical Chunking**

```
Document
    ↓
Section (H1/H2 boundary or 1000 tokens)
    ↓
Chunk (300-500 tokens with 50-token overlap)
```

**Rules:**
1. **Never split mid-sentence** — use sentence boundary detection (spaCy or NLTK)
2. **Always include a summary prefix** on every chunk:
   ```
   [Company: Hanwha Q Cells] [County: Hall] [Category: Investment]
   [Source: georgia.org press release, 2024-03-12]
   "Hanwha Q Cells announced a $2.5 billion expansion of its Dalton facility..."
   ```
3. **Tables get special treatment** — convert to markdown, keep as one chunk per table
4. **Numeric facts** get duplicate chunks — one in the full paragraph, one as a "fact chunk":
   ```
   "Hanwha Q Cells | Investment: $2.5B | Jobs: 1,500 | Year: 2024"
   ```

**Chunk Size Decision:**
```
For structured fact queries → smaller chunks (200-300 tokens) = more precise
For contextual/open-ended queries → larger chunks (500-800 tokens) = more context
Recommendation: Default 400 tokens, 75-token overlap
```

---

### 7d. Metadata Schema Per Chunk

Every chunk stored in vector DB must have these metadata fields (based on your `RAG_Data_Management_Framework.xlsx` reference):

```python
{
  # Source traceability
  "doc_id": "DOC_042",                    # From Phase 1 tracker
  "chunk_id": "DOC_042_C003",             # doc_id + chunk index
  "source_url": "https://georgia.org/...",
  "source_domain": "georgia.org",
  "file_type": "html",                    # pdf | html | docx | csv
  "downloaded_at": "2024-05-01",
  "content_date": "2024-03-12",           # Date of the actual content
  
  # Company attribution
  "company_name": "Hanwha Q Cells",       # Primary company
  "company_id": "C042_hanwha_q_cells",    # From Phase 1 companies.json
  "related_companies": ["SK On", "Kia"],  # Other companies mentioned
  "oem_customer": "Kia Georgia",          # If mentioned
  
  # Geographic
  "county": "Hall",
  "city": "Dalton",
  "state": "Georgia",
  
  # Content classification
  "category": "Investment",               # From GNEM taxonomy
  "sub_category": "Expansion Announcement",
  "document_type": "Press Release",       # News | Press Release | SEC Filing | Report | Permit | Data
  
  # Facts (for hybrid retrieval)
  "investment_amount_usd": 2500000000,    # Structured fact extracted
  "jobs_created": 1500,
  "year": 2024,
  
  # Quality signals
  "relevance_score": 0.87,               # From Phase 1 scorer
  "priority_domain": true,               # From Phase 1
  "is_validated": false,                 # True only if in the 50 human-validated Q&A
}
```

---

### 7e. Embedding Model Choice

| Model | Where Runs | Cost | Quality |
|-------|-----------|------|---------|
| `text-embedding-3-small` (OpenAI) | Cloud | ~$0.02/1M tokens | ⭐⭐⭐⭐ |
| `nomic-embed-text` | Local (Ollama) | Free | ⭐⭐⭐⭐ |
| `bge-large-en-v1.5` (BAAI) | Local or HuggingFace | Free | ⭐⭐⭐⭐⭐ |
| `all-MiniLM-L6-v2` | Local | Free | ⭐⭐⭐ (fast, lower quality) |

**Recommendation for this project:** `nomic-embed-text` via Ollama for local dev, `text-embedding-3-small` for production. Both 768-dim embeddings.

---

## 8. Phase 3 Blueprint — Retrieval & Prompt Strategy

### 8a. The Hybrid Retriever

```
User Question
      ↓
┌─────────────────────────────────────────┐
│           QUERY CLASSIFIER              │
│  Is this a RELATIONSHIP/COUNT question? │
│  → Yes → Cypher (Neo4j)                 │
│  Is this a FACTUAL/DOCUMENT question?   │
│  → Yes → Vector Search                  │
│  Is this BOTH?                          │
│  → Parallel: Cypher + Vector → Merge    │
└─────────────────────────────────────────┘
```

**Example routing:**

| Question | Route | Why |
|----------|-------|-----|
| "How many suppliers in Bryan County?" | Neo4j Cypher | COUNT query on graph |
| "What did Kia Georgia announce in 2024?" | Vector Search | Unstructured news content |
| "Which suppliers with <200 employees supply Hyundai?" | Neo4j Cypher | Relationship traversal + filter |
| "What is the investment climate in Georgia for EV batteries?" | Vector Search | Contextual/opinion question |
| "Tell me about Hanwha Q Cells' Georgia operations" | BOTH | Mix of structured facts + documents |

---

### 8b. Cypher Query Generation (for Neo4j)

The LLM must translate natural language to Cypher. Example:

```
User: "Which suppliers with fewer than 200 employees supply to Hyundai Metaplant?"

Generated Cypher:
MATCH (s:Company)-[:SUPPLIES_TO]->(o:OEM {name: "Hyundai Metaplant"})
WHERE s.employee_count < 200
RETURN s.name, s.employee_count, s.county
ORDER BY s.employee_count ASC
```

**This is the "Secret Weapon" from your graphrag-finetune repo** — fine-tuning an open-source LLM to generate accurate Cypher for your specific schema.

---

### 8c. Prompt Strategy

**System Prompt (for the chatbot):**
```
You are the Georgia EV Supply Chain Intelligence Assistant. 
You help economic developers, supply chain analysts, and policy makers 
understand the EV manufacturing ecosystem in Georgia.

Rules:
1. Only answer from the retrieved context. Never guess facts.
2. Always cite your source (company name, document type, date, URL).
3. For numeric facts (investment, jobs, employees), state them precisely.
4. If you don't have enough information, say so clearly.
5. When asked about relationships (who supplies whom), use the graph data.
6. When asked about news or announcements, use the document retrieval.

Current date: {current_date}
User location context: Georgia EV Ecosystem
```

**RAG Prompt Template:**
```
Context from Knowledge Graph:
{graph_results}

Context from Documents:
{vector_results}

Human Question: {question}

Answer based ONLY on the above context. 
Cite sources for every factual claim.
Format your answer clearly with bullet points for lists.
```

---

### 8d. The 50 Human-Validated Q&A — How to Use Them

> ⚠️ You correctly said: "don't design prompts or answers based on the 50 Q&A"

**Correct use:**
- Use them as a **test/evaluation set** only
- After building the RAG pipeline, run all 50 questions through it
- Compare the RAG answers to the human-validated answers
- Measure: accuracy, completeness, hallucination rate, citation quality
- Score: ≥ 80% match = system is ready

**Do NOT use them to:**
- Train the system to pattern-match those specific answers
- Hard-code those answers anywhere
- Design retrieval around just those 50 patterns

---

## 9. Implementation Roadmap

### Sprint 0 (NOW): Fix Phase 1 Speed
**Goal:** Complete all 206 companies in < 24 hours

**Actions:**
1. Switch primary search from DDG-only → **Tavily API** (parallel, purpose-built for RAG)
2. Process companies **in parallel batches** (5 at a time vs 1 at a time)
3. Raise `max_concurrent_downloads: 2 → 8`
4. Add **GNEM Excel direct ingestion** — don't search for basic company facts you already have
5. Keep DDG as free fallback only

**Estimated effort:** 3-5 days

---

### Sprint 1 (Week 1-2): Phase 2 — Text Extraction & Chunking
**Goal:** All downloaded documents become structured chunks

**Build:**
- `src/extractor.py` — full implementation (PDF, HTML, DOCX, CSV extractors)
- `src/chunker.py` — contextual chunking with metadata tagging
- `src/entity_extractor.py` — NLP-based entity extraction (company names, $amounts, job counts, dates)
- PostgreSQL `chunks` table to store chunk metadata
- B2 storage for chunk text files

**Estimated effort:** 1 week

---

### Sprint 2 (Week 2-3): Phase 2 — Embedding & Vector Store
**Goal:** All chunks are embedded and searchable

**Build:**
- `src/embedder.py` — batch embedding with `nomic-embed-text` or OpenAI
- Vector DB setup (recommendation: **Qdrant** — open source, self-hostable, excellent metadata filtering)
- `src/vector_store.py` — upload chunks + metadata to Qdrant
- Basic semantic search test

**Estimated effort:** 1 week

---

### Sprint 3 (Week 3-4): Phase 3 — Neo4j Graph Build
**Goal:** Knowledge graph loaded with all structured data

**Build:**
- Neo4j setup (local Docker or Aura free tier)
- `src/graph_loader.py` — load GNEM Excel → Neo4j nodes + relationships
- `src/graph_enricher.py` — enrich graph with facts extracted from chunks in Sprint 1
- Validate graph with sample Cypher queries

**Estimated effort:** 1 week

---

### Sprint 4 (Week 4-5): Phase 3 — Retriever & LLM Orchestration
**Goal:** End-to-end question answering working

**Build:**
- `src/retriever.py` — hybrid retriever (Neo4j + Qdrant)
- `src/query_router.py` — classify question → route to correct retriever
- LLM orchestration with **LangChain** + open-source LLM (Llama 3.1 8B or Mixtral 8x7B via Ollama)
- Basic `src/agent.py` — takes question, retrieves context, generates answer with citations

**Estimated effort:** 1.5 weeks

---

### Sprint 5 (Week 5-6): Evaluation & Chat UI
**Goal:** System evaluated and user-facing chatbot live

**Build:**
- Evaluation harness using the 50 human-validated Q&A
- Score: accuracy, hallucination rate, citation quality
- Simple chat UI (FastAPI backend + React/HTML frontend)
- Test with next 50 Q&A (when you provide them)

**Estimated effort:** 1 week

---

### Sprint 6 (Optional): Fine-tuning for Better Cypher
**Goal:** LLM generates better Cypher queries for your specific schema

**Approach (from graphrag-finetune repo):**
1. Extract schema from Neo4j
2. Generate synthetic Q&A pairs (question → Cypher)
3. Fine-tune Llama 3.1 on these pairs using QLoRA (4-bit quantization)
4. Deploy fine-tuned model via vLLM

**Estimated effort:** 2 weeks

---

## 10. Key Decisions Needed From You

> These need your input before we can proceed to implementation.

| # | Decision | Options | Recommendation |
|---|----------|---------|----------------|
| 1 | **Primary search tool for Phase 1 speed fix** | Tavily API vs SerpAPI vs pure parallel DDG | Tavily — purpose-built for RAG data collection |
| 2 | **Vector database** | Qdrant vs ChromaDB vs Pinecone vs Weaviate | Qdrant — open-source, self-hostable, best metadata filtering |
| 3 | **Embedding model** | OpenAI text-embedding-3-small vs nomic-embed-text (free) | nomic-embed-text for dev, OpenAI for prod |
| 4 | **LLM for orchestration** | Llama 3.1 8B (local) vs Mixtral 8x7B vs OpenAI GPT-4o | Llama 3.1 8B via Ollama for cost, GPT-4o for quality |
| 5 | **Neo4j hosting** | Local Docker vs Neo4j Aura free tier | Aura free for dev, local for prod |
| 6 | **Do we fine-tune?** | Yes (better Cypher, 2 weeks work) vs No (prompt engineering only) | Start without, add later |
| 7 | **Chat UI** | Simple HTML page vs Full React app | Start simple, upgrade later |
| 8 | **Which 50 new Q&A to get next?** | Focus on relationship queries vs document queries vs mix | Mix — cover both graph + vector retrieval |

---

## Summary: What We Have, What We Need

```
PHASE 1 STATUS: ~80% COMPLETE
  ✅ URL discovery engine (searcher.py)  
  ✅ Parallel downloader (downloader.py)
  ✅ Cloud storage (storage.py → Backblaze B2)
  ✅ PostgreSQL tracker (tracker.py, db.py)
  ✅ Convergence tracking
  ✅ Checkpoint/resume
  ❌ SPEED (3hrs/10 companies → need fix for 206)

PHASE 2 STATUS: 0% — NOT STARTED
  ❌ Text extraction (extractor.py is empty stub)
  ❌ Chunking
  ❌ Metadata tagging
  ❌ Embedding
  ❌ Vector store

PHASE 3 STATUS: 0% — NOT STARTED
  ❌ Neo4j knowledge graph
  ❌ Hybrid retriever
  ❌ LLM orchestration
  ❌ Chat UI
  ❌ Evaluation harness
```

**Immediate Next Step:** Fix Phase 1 speed, then begin Phase 2 extraction pipeline.

> Do you want me to start with the Tavily integration to fix the speed problem first, or jump directly to designing Phase 2?
