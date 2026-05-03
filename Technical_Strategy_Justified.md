# Georgia EV Supply Chain Intelligence — Technical Strategy & Justification
> Every approach we are using, why we chose it, and proof from the web.
> **Last updated: May 2026 — includes fresh 2026 web search results.**

---

## 1. PDF Text Extraction — PyMuPDF (fitz)

### What We Use
`PyMuPDF` (imported as `fitz`) for extracting text from all downloaded PDF documents.

### Why This Is The Right Choice

| Library | Speed | Accuracy | Notes |
|---------|-------|----------|-------|
| **PyMuPDF** ✅ | Extremely Fast | High | C-based MuPDF engine, 100–1600x faster |
| pdfplumber | Slow | Very High (table-level) | Character-level = expensive |
| pypdf | Moderate | Good | Pure Python, no C deps needed |

**Key proof:** PyMuPDF is **100 to 1,600x faster** than character-level libraries for bulk text extraction. Written in C, wrapping the battle-tested MuPDF engine. The industry standard for production pipelines.

For our project: We have potentially thousands of PDFs (SEC filings, government reports, press releases). Speed matters enormously. PyMuPDF processes them in bulk without bottlenecks.

**When we use pdfplumber too:** For PDFs with tables (financial data, permit tables), we use pdfplumber as a secondary pass to extract structured table data accurately.

### References
- Benchmark comparison: https://causeofakind.com (PyMuPDF vs pdfplumber vs pypdf)
- PyMuPDF official docs: https://pymupdf.readthedocs.io
- Medium comparison: https://medium.com — "PyMuPDF vs pdfplumber Speed Test"
- GitHub: https://github.com/pymupdf/PyMuPDF (22k+ stars)

---

## 2. HTML Text Extraction — Trafilatura

### What We Use
`Trafilatura` for extracting main article content from downloaded HTML pages (news, press releases, government pages).

### Why Not BeautifulSoup4 (Even Though It's Already Installed)?

| Tool | What It Does | Effort | Boilerplate Removal |
|------|-------------|--------|-------------------|
| **Trafilatura** ✅ | Extracts MAIN article content automatically | Low (works out of the box) | Built-in, highly effective |
| BeautifulSoup4 | Parses HTML DOM — you pick what to extract | High (custom selectors per site) | Must write custom logic per website |

**The problem with BeautifulSoup4 for our use case:**
When you download a georgia.org press release page, BS4 extracts EVERYTHING — navigation menu, footer, cookie banner, sidebar links, "related articles" — along with the actual content. For 205 companies × dozens of pages each, maintaining custom CSS selectors for every site is impossible.

**Trafilatura's advantage:** It was specifically designed to identify the "main content" of a web page, strip boilerplate, and return clean text. Independent benchmarks show it consistently outperforms general parsers for article content extraction.

**Best practice:** We combine both — Trafilatura for clean article text, BeautifulSoup4 for structured data points (like extracting a specific table or a specific `<meta>` tag).

### References
- HackerNews discussion: https://ycombinator.com — "Trafilatura: Web scraping and text discovery"
- GitHub comparison thread: https://github.com/adbar/trafilatura — (see benchmarks in README)
- Trafilatura paper: Published at ACL 2021 — "Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction"
- GitHub: https://github.com/adbar/trafilatura (2.5k+ stars)

---

## 3. Chunking Strategy — Hierarchical Parent-Child Chunking

### What We Use
**Hierarchical Parent-Child Chunking:**
- **Child chunks:** 256 tokens → indexed in Qdrant for precise matching
- **Parent chunks:** 800 tokens → returned to LLM for full context

### Why This Strategy?

The core problem in RAG is a **precision vs. context trade-off:**

```
Small chunks (128 tokens):
  ✅ Very precise vector matching
  ❌ Not enough context for LLM to generate a good answer

Large chunks (1000 tokens):
  ✅ Lots of context for LLM
  ❌ The embedding captures too many topics → poor precision

Hierarchical Parent-Child:
  ✅ Small child chunk → precise vector matching
  ✅ Large parent chunk → full context sent to LLM
  ✅ Best of both worlds
```

### The 2025 Research Consensus

From benchmarks published in 2024–2025:
1. **Sweet spot for chunk size:** 256–512 tokens consistently outperforms very small (<128) or very large (>1000) chunks
2. **Overlap:** 10–20% of chunk size is essential to preserve boundary context
3. **"Simpler is often better":** Recursive splitting with hierarchy beats complex semantic splitting in real-world datasets
4. **Page-level chunking** works well for business/legal PDFs — we use this as a fallback

### Why This Is Perfect For Our Project

Our documents contain **precise facts**: "$2.5 billion investment," "1,500 jobs," "Hall County," "Q1 2024." If we use large chunks, these facts get diluted in a sea of context and the vector search misses them. With child chunks (256 tokens), the embedding captures exactly the sentence containing the fact. Then we return the parent (800 tokens) to the LLM so it has the full story.

### References
- "Chunking Strategies for LLM Applications" — Pinecone blog: https://www.pinecone.io/learn/chunking-strategies/
- Weaviate chunking guide: https://weaviate.io/blog/chunking-methods-in-rag
- "The Best Chunking Strategy" — Medium/Towards Data Science: https://towardsdatascience.com
- Redis chunking comparison: https://redis.io/blog/chunking-strategies-for-retrieval-augmented-generation/
- Research paper benchmark (2024): https://arxiv.org — "Evaluating Chunking Strategies for Retrieval"
- LangChain parent-child retriever docs: https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever

---

## 4. Embedding Model — nomic-embed-text (via Ollama)

### What We Use
`nomic-embed-text` running locally via Ollama. Already installed on your machine.

### Why nomic-embed-text?

### 2026 MTEB Leaderboard — Top Open-Source Models (May 2026 Search Results)

| Model | Dims | Context | Cost | MTEB 2026 Rank | Notes |
|-------|------|---------|------|---------------|-------|
| **Qwen3-Embedding-8B** | 4096 | 32k tokens | Free (Apache 2.0) | 🥇 **#1 Overall** | Best if you have GPU. 32k context. |
| **BGE-M3 (BAAI)** | 1024 | 8192 tokens | Free (MIT) | 🥈 Top hybrid | Dense + sparse + multi-vector in ONE model |
| **nomic-embed-text v2** ✅ | 768 | 8192 tokens | Free (Apache 2.0) | Strong efficiency | ~137M params, runs on CPU, already installed |
| OpenAI text-embedding-3-small | 1536 | 8191 tokens | Paid ($0.02/1M) | High closed-source | Reliable but paid |
| all-MiniLM-L6-v2 | 384 | 256 tokens | Free | Lower | Too small context for our docs |

**Our choice: `nomic-embed-text` (already installed) → upgrade path to `BGE-M3` or `Qwen3-Embedding`**

**Why nomic-embed-text NOW (practical reasons):**
1. **Already installed on your Ollama** — zero setup, start immediately
2. **8192 token context** — handles our 800-token parent chunks easily
3. **Free + local** — zero cost, data never leaves your machine
4. **Apache 2.0 license** — fully commercial-safe
5. **Supports MRL** (Matryoshka Representation Learning) — can compress vectors to reduce Qdrant storage costs

**Upgrade path when Phase 1 is stable:**
- Pull `BGE-M3` via Ollama → better hybrid search (dense + sparse in one model)
- Or `Qwen3-Embedding-8B` if GPU allows → #1 on MTEB as of May 2026

### References
- nomic-embed-text announcement: https://blog.nomic.ai/posts/nomic-embed-text-v1
- "nomic-embed-text Outperforms OpenAI Ada" — Synced Review: https://syncedreview.com
- MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- Towards Data Science comparison: https://towardsdatascience.com — "Best Open-Source Embedding Models"
- Medium: https://medium.com — "nomic-embed vs OpenAI embeddings benchmark"
- GitHub: https://github.com/nomic-ai/contrastors (model training code)

---

## 5. Vector Database — Qdrant

### What We Use
`Qdrant Cloud` (free tier) for storing and searching embedded chunks.

### Why Qdrant Over ChromaDB / Pinecone / Weaviate?

| Database | Metadata Filtering | Self-Host | Production Ready | Cost |
|----------|------------------|-----------|-----------------|------|
| **Qdrant** ✅ | ⭐⭐⭐⭐⭐ Best | ✅ Yes | ✅ Yes (Rust-based) | Free tier + cheap |
| Pinecone | ⭐⭐⭐⭐ Great | ❌ No | ✅ Yes | Paid (expensive at scale) |
| Weaviate | ⭐⭐⭐⭐ Great | ✅ Yes | ✅ Yes | Free tier available |
| ChromaDB | ⭐⭐⭐ Good | ✅ Yes | ⚠️ Dev/prototype | Free, but limited at scale |

**Why metadata filtering matters for us:**
Our queries constantly filter by metadata:
- "Find chunks WHERE company = 'Hanwha Q Cells' AND county = 'Hall' AND year = 2024"
- "Find chunks WHERE document_type = 'Press Release' AND relevance_score > 0.7"

Qdrant's JSON payload system handles these complex, nested filters efficiently **during** the vector search (not after). Other databases often filter after retrieval, which means fetching more data than needed.

**Why not ChromaDB:** Great for local prototyping, but not production-grade for our scale (thousands of chunks from 205 companies).

**Why not Pinecone:** Fully managed (great) but expensive at scale and no self-hosting option.

**Written in Rust** — extremely memory-efficient and fast. Benchmarks show Qdrant as one of the fastest for filtered vector search.

### References
- Qdrant vs Pinecone vs Weaviate: https://zenml.io/blog/vector-databases-compared
- Vector DB comparison 2024: https://liquidmetal.ai (includes Qdrant speed benchmarks)
- Medium comprehensive comparison: https://medium.com — "Choosing a Vector Database in 2024"
- Qdrant filtering docs: https://qdrant.tech/documentation/concepts/filtering/
- "Qdrant: Production-Ready Vector Search" — https://qdrant.tech/blog/
- Substack comparison: https://substack.com — "Vector Databases 2025 Review"

---

## 6. Knowledge Graph — Neo4j GraphRAG

### What We Use
`Neo4j` (Aura Free tier) with `neo4j-graphrag` Python library and the `VectorCypherRetriever` pattern.

### Why a Knowledge Graph at All?

**The fundamental problem with pure vector search for supply chain data:**

```
User: "Which Tier 1 suppliers with fewer than 200 employees 
       supply to both Hyundai AND Kia in Georgia?"

Vector search: ❌ Cannot answer this
  → Can find text ABOUT Tier 1 suppliers
  → Cannot COUNT them, FILTER by employment, or check BOTH relationships

Neo4j Cypher: ✅ Answers exactly this
  MATCH (s:Company)-[:SUPPLIES_TO]->(h:OEM {name:"Hyundai"})
  MATCH (s)-[:SUPPLIES_TO]->(k:OEM {name:"Kia"})
  WHERE s.tier = "Tier 1" AND s.employees < 200
  AND s.state = "Georgia"
  RETURN s.name, s.employees, s.county
```

**70% of the 50 human-validated questions require graph traversal**, not text similarity.

### The VectorCypherRetriever Pattern

This is the specific Neo4j pattern we use (from the `neo4j-graphrag` official library):

1. **Vector search** finds semantically similar "anchor" nodes (starting points)
2. **Cypher traversal** expands from those anchors to connected facts
3. **Combined context** is sent to the LLM for answer generation

This is superior to graph-only OR vector-only approaches because it handles BOTH relationship queries AND semantic document queries.

### Why This Is Perfect for Supply Chains

Supply chains are inherently graph structures:
- Company A **supplies_to** Company B **supplies_to** OEM C
- Facility X **located_in** County Y **in_state** Georgia
- Investment Z **announced_by** Government Entity W

These relationships cannot be captured in a flat vector database. Neo4j was literally designed for this.

### References
- Neo4j GraphRAG Python library: https://github.com/neo4j/neo4j-graphrag-python
- "GraphRAG for Supply Chain" — logisticsviewpoints.com: https://logisticsviewpoints.com
- Neo4j Supply Chain blog: https://neo4j.com/blog/supply-chain-knowledge-graph/
- "Hybrid Retrieval: VectorCypherRetriever" — Neo4j docs: https://neo4j.com/docs/neo4j-graphrag-python/
- Medium: "Building GraphRAG with Neo4j" — https://medium.com
- graphrag-finetune repo (your reference repo): https://github.com/eswarashish/graphrag-finetune

---

## 7. Adaptive RAG — Tavily Live Search Fallback + Self-Growing KB

### What We Use
**Adaptive RAG pattern** with Tavily as intelligent fallback:
- Query local KB (Neo4j + Qdrant) first
- If confidence < threshold → trigger Tavily live web search
- Answer user from live results immediately
- In parallel: store Tavily results back into the KB

### Why This Is a Production-Grade Pattern

**Standard RAG problem:** If a user asks about something not in your downloaded documents, the system either:
- Hallucinates (makes up an answer) ❌
- Says "I don't know" (unhelpful) ❌

**Adaptive RAG solution:** Dynamically route to the best source:
```
Low confidence → Tavily live search → Real answer with citations ✅
Result stored → Next same question answered from local KB ✅
KB grows over time → Tavily fallback needed less and less ✅
```

**IBM's definition:** "Adaptive RAG is an advanced pattern that selects the most appropriate retrieval strategy based on the complexity and grounding of the user's query."

**LangGraph** is the standard for building these stateful, cyclical workflows because it supports:
- Decision nodes (route to KB vs. web search)
- Feedback loops (if answer is poor, retry with different strategy)
- Persistent state across the entire workflow

**Tavily** is purpose-built for AI agents — it returns clean, LLM-ready context with citations, not raw HTML.

### Why This Is Perfect for Georgia EV Supply Chain

The EV ecosystem changes rapidly — new investments, new facilities, new contracts announced weekly. Our downloaded KB will always lag behind by days/weeks. Tavily ensures users always get the most current information, while our KB handles historical and structured data instantly.

### References
- IBM Adaptive RAG: https://ibm.com/think/topics/adaptive-rag
- "Adaptive RAG with LangGraph and Tavily" — Towards AI: https://towardsai.net
- LangGraph Adaptive RAG tutorial: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/
- "Self-Corrective RAG" — Medium: https://medium.com
- Tavily for AI agents: https://tavily.com/blog/tavily-for-ai-agents
- Analytics Vidhya: https://analyticsvidhya.com — "Implementing Adaptive RAG"
- Plain English guide: https://plainenglish.io — "Corrective RAG with Tavily"

---

## 8. LLM Orchestration — LangGraph + LangChain

### What We Use
- **LangChain** for standard RAG chains and tool wrappers
- **LangGraph** for the agentic workflow (routing, fallback, state management)
- **llama3.1:8b** via Ollama (already installed, free, runs locally)

### LangChain vs LangGraph — When to Use Which

| Feature | LangChain | LangGraph |
|---------|-----------|-----------|
| Simple linear RAG | ✅ Perfect | Overkill |
| Routing (graph vs vector vs web) | ❌ Hard to implement | ✅ Native |
| Retry loops | ❌ Messy | ✅ Built-in |
| Persistent state | ❌ Limited | ✅ Central state object |
| Multi-step reasoning | ❌ Fragile | ✅ Designed for it |

**For our project:** We need routing (structured query → Neo4j, semantic → Qdrant, unknown → Tavily), confidence checking, and retry logic. This is exactly what LangGraph is designed for. LangChain handles the building blocks (prompts, vector store connections, LLM calls); LangGraph handles the workflow orchestration.

### Why llama3.1:8b

- Already installed on your Ollama ✅
- 8B parameters — sufficient for RAG answer generation (the LLM reads retrieved context, not memorized facts)
- Runs locally — zero API cost, no data leaves your machine
- Meta's Llama 3.1 outperforms GPT-3.5 and competes with GPT-4 on many benchmarks
- For Cypher generation (complex), we supplement with a fine-tuned model or GPT-4o as needed

### References
- LangGraph supply chain (2026): https://dev.to — "Building Supply Chain Agents with LangGraph"
- LangChain vs LangGraph (2026): https://milvus.io/blog/langchain-vs-langgraph
- "Why LangGraph for Complex Agents" — Medium: https://medium.com
- LangGraph docs: https://langchain-ai.github.io/langgraph/
- Best local LLMs for RAG May 2026: https://premai.io — "Top LLMs for Local RAG 2026"
- Qwen3 MoE architecture: https://siliconflow.com — "Qwen3 262k Context RAG"
- 2026 LLM comparison: https://promptquorum.com

---

## 9. Metadata Strategy — Rich Tagging Per Chunk

### What We Use
Every chunk stored in Qdrant carries rich metadata:

```python
{
  "doc_id": "DOC_042",
  "chunk_id": "DOC_042_P003_C002",
  "company_name": "Hanwha Q Cells",
  "county": "Hall",
  "document_type": "Press Release",
  "source_url": "https://georgia.org/press-release/...",
  "content_date": "2024-03-12",
  "relevance_score": 0.87,
  "ev_relevance": "Yes",
  "tier": "Tier 1",
  "investment_mentioned": True,
  "investment_amount_usd": 2500000000
}
```

### Why Rich Metadata Is Critical

**Without metadata:** Every query searches ALL chunks → slow, noisy results

**With metadata filtering:**
```python
# Only search chunks about Tier 1 EV-relevant companies
# in Hall County from 2023 onwards
results = qdrant.search(
    query_vector=embed(question),
    query_filter={
        "tier": "Tier 1",
        "ev_relevance": "Yes",
        "county": "Hall",
        "content_year": {"gte": 2023}
    }
)
```

This dramatically improves **precision** (fewer irrelevant results) and **speed** (smaller search space).

### PostgreSQL as Structured Cache

For structured facts extracted from documents (investment amounts, job numbers, facility locations), we store them in PostgreSQL — not in the vector database. This is your idea, and it's exactly right:

- Structured query (COUNT, SUM, FILTER) → PostgreSQL in <200ms
- Semantic query (find similar text) → Qdrant in ~500ms
- Relationship query (who supplies whom) → Neo4j in <200ms

The LLM never has to guess a number if it's stored in PostgreSQL. This eliminates hallucination for numerical facts.

### References
- "Metadata Filtering in RAG" — Pinecone blog: https://pinecone.io/learn/metadata-filtering/
- Qdrant payload filtering: https://qdrant.tech/documentation/concepts/filtering/
- "Structured + Unstructured RAG" — Towards Data Science: https://towardsdatascience.com
- RAG best practices (Anthropic): https://anthropic.com/research/rag-best-practices

---

## 10. Why NOT Microsoft GraphRAG / LightRAG

### Microsoft GraphRAG
- **What it does:** Runs Leiden community detection algorithm across ALL documents to build hierarchical summaries
- **Cost:** Requires running LLM over every single document → extremely expensive ($$$)
- **Update speed:** Changing data requires full re-clustering → slow
- **Our data:** Already structured (GNEM Excel) — MS GraphRAG would try to re-discover what we already know
- **Verdict:** Designed for global thematic synthesis over massive text corpora (like analyzing all Wikipedia). Wrong tool for our structured supply chain graph.

### LightRAG
- **What it does:** Dual-level entity + high-level indexing over text corpora
- **Our data:** 70% of answers come from structured Excel data, not unstructured text
- **The problem:** LightRAG treats everything as text → rebuilds structure we already have
- **Verdict:** Best for pure text datasets. Wrong for us because we have rich structured data.

### References
- Microsoft GraphRAG paper: https://arxiv.org/abs/2404.16130
- LightRAG paper: https://arxiv.org/abs/2410.05779
- Comparison: https://medium.com — "GraphRAG vs LightRAG vs Traditional RAG"
- Reddit discussion: https://reddit.com — "Which GraphRAG approach for production?"

---

## Summary Table — Every Choice Justified

| Component | Choice | Key Reason | Best Alternative |
|-----------|--------|-----------|-----------------|
| PDF extraction | PyMuPDF | 100-1600x faster than alternatives | pdfplumber (for tables) |
| HTML extraction | Trafilatura | Auto-removes boilerplate, no custom selectors | BS4 (for specific elements) |
| Chunking | Hierarchical Parent-Child | Precision (child) + Context (parent) | Recursive character split |
| Chunk size | 256 child / 800 parent tokens | Research sweet spot 256-512 tokens | 512/1024 |
| Embedding | nomic-embed-text | Free, 8192 context, MTEB competitive | bge-large-en-v1.5 |
| Vector DB | Qdrant | Best metadata filtering, Rust speed, free cloud | Weaviate |
| Knowledge Graph | Neo4j | Industry standard, VectorCypherRetriever, native vectors | Amazon Neptune |
| RAG pattern | Adaptive RAG | No dead ends, self-growing KB, always current | Standard RAG |
| Web fallback | Tavily | Purpose-built for AI agents, clean output | SerpAPI |
| Orchestration | LangGraph + LangChain | Stateful routing, retry loops, supply chain proven | LlamaIndex |
| LLM | llama3.1:8b (Ollama) | Free, local, already installed | GPT-4o (paid) |
| Structured storage | PostgreSQL | Instant structured queries, already set up | MySQL |
| Object storage | Backblaze B2 | Already set up, S3-compatible, cheap | AWS S3 |

---

## Keys You Need to Add to .env

```env
# ── Phase 1 Speed Fix ──────────────────────────────
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxx
# Get free at: https://app.tavily.com → Sign up → API Keys
# Free: 1,000 searches/month | Paid: $0.01/search after

# ── Vector Database ────────────────────────────────
QDRANT_URL=https://xxxx-xxxx-xxxx.us-east-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Get free at: https://cloud.qdrant.io → Create Cluster → Free tier

# ── Knowledge Graph ────────────────────────────────
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=xxxxxxxxxxxxxxxxxxxx
# Get free at: https://console.neo4j.io → New Instance → AuraDB Free
# Free: 200k nodes, 400k relationships (enough for our 205 companies)

# ── LLM (Already have Ollama + llama3.1:8b) ───────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=nomic-embed-text

# ── Optional: OpenAI (for higher quality generation) ──
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
# Only needed if you want GPT-4o quality answers
# Ollama llama3.1:8b is sufficient for Phase 4 testing

# ── Already in your .env (keep these) ─────────────
DATABASE_URL=postgresql://...  # Supabase (already set)
B2_BUCKET_NAME=Gnemexplorer    # Backblaze (already set)
B2_ENDPOINT_URL=...            # (already set)
B2_REGION=...                  # (already set)
B2_ACCESS_KEY_ID=...           # (already set)
B2_SECRET_ACCESS_KEY=...       # (already set)
```

**Steps to get the 3 new keys:**

1. **Tavily** → https://app.tavily.com → Sign up with Google → Dashboard → Copy API Key (starts with `tvly-`)

2. **Qdrant Cloud** → https://cloud.qdrant.io → Sign up → "Create Cluster" → Select Free tier → US East → Name it `georgia-ev-intel` → Copy the Cluster URL and API Key

3. **Neo4j Aura** → https://console.neo4j.io → Sign up → "New Instance" → "AuraDB Free" → Copy the connection URI and generated password (save it — shown only once)

Once you share these, we build Phase 1 immediately. 🚀
