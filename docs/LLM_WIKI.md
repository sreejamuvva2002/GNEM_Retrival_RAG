# LLM-Wiki Implementation

Building a persistent, synthesized knowledge graph from your document corpus using Claude.

## Overview

The LLM-Wiki follows Karpathy's approach to knowledge management: instead of repeatedly retrieving raw documents at query time, the system uses a **local Ollama LLM** to **synthesize documents into a persistent wiki** with structured pages, cross-references, and entity relationships.

**Key Benefit**: Uses your local LLM (Qwen, Llama, etc.) - no external API calls or costs!

### Three-Layer Architecture

```
Raw Documents (ddg_search.jsonl, etc.)
         ↓
    LLM Analysis & Synthesis
         ↓
Persistent Wiki (markdown pages, index)
         ↓
Query → Wiki Search + Hybrid Retrieval → LLM Answer
```

## Quick Start

### 1. Prerequisites

**Start Ollama** (in a separate terminal):
```bash
ollama serve
```

**Verify your `.env`** has Ollama configuration (already set up):
```bash
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_LLM_MODEL="qwen2.5:32b"  # or any model you prefer
```

**Install dependencies**:
```bash
pip install -r requirements.txt
```

### 2. Ingest Documents

Process your `ddg_search.jsonl` and generate wiki pages:

```bash
# Ingest all documents (can take a while)
python -m georgia_ev_intelligence.kb_builder.wiki_cli ingest \
  --source kb/raw_docs/ddg_search.jsonl

# Ingest with limit (for testing)
python -m georgia_ev_intelligence.kb_builder.wiki_cli ingest \
  --source kb/raw_docs/ddg_search.jsonl \
  --limit 10 \
  --export outputs/wiki_preview.md
```

Progress will show like:
```
[wiki] Ingesting from kb/raw_docs/ddg_search.jsonl
[wiki] Wiki directory: kb/wiki
[wiki] Processing 1: Foreign Investment Travels Nonstop to Georgia...
      → Updated 3 pages: SK Innovation, Duckyang, Georgia
...
[wiki] Done!
[wiki] Documents processed: 100
[wiki] Page updates: 347
[wiki] Total pages: 87
```

### 3. Search and Browse

```bash
# Search for an entity
python -m georgia_ev_intelligence.kb_builder.wiki_cli search "SK Innovation"

# Show a full page
python -m georgia_ev_intelligence.kb_builder.wiki_cli show "SK Innovation"

# List all companies
python -m georgia_ev_intelligence.kb_builder.wiki_cli list --type company

# Get statistics
python -m georgia_ev_intelligence.kb_builder.wiki_cli stats
```

### 4. Export Wiki

```bash
# Export as single markdown file
python -m georgia_ev_intelligence.kb_builder.wiki_cli export \
  --output outputs/gnem_wiki.md
```

## Integration with Hybrid Retrieval

The wiki works **alongside** (not instead of) your existing BM25 + dense + rerank pipeline.

### Using Wiki-Enhanced Retrieval

```python
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.orchestrator import HybridRetrievalOrchestrator
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.wiki_retriever import WikiRetriever
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.wiki_orchestrator import WikiEnhancedOrchestrator

# Initialize components
hybrid = HybridRetrievalOrchestrator(...)
wiki_retriever = WikiRetriever(wiki_dir="kb/wiki")

# Create enhanced orchestrator
enhanced = WikiEnhancedOrchestrator(hybrid, wiki_retriever, wiki_top_k=3)

# Retrieve with both wiki + hybrid
result = enhanced.retrieve_enhanced("What is SK Innovation investing in Georgia?")

print("Wiki results:")
for wiki_res in result.wiki_results:
    print(f"  - {wiki_res.title} ({wiki_res.entity_type})")

print("\nCombined context for LLM:")
print(result.combined_context)
```

## File Structure

```
kb/wiki/
  _index.json                 # Master index (pages, entities, sources)
  sk_innovation.md            # Individual entity pages
  duckyang.md
  georgia_ev_battery.md
  ... (one .md per entity)
```

Each page contains:
- **Frontmatter**: title, entity_type, last_updated, sources, related_entities
- **Body**: markdown with key facts, relationships, citations

Example page (`kb/wiki/sk_innovation.md`):
```markdown
---
{
  "title": "SK Innovation",
  "entity_type": "company",
  "last_updated": "2026-06-02T...",
  "sources": ["sha256:abc123...", "sha256:def456..."],
  "related_entities": ["Duckyang", "SK Battery America", "Georgia"]
}
---

# SK Innovation

## Overview

South Korea's largest energy company investing in US battery manufacturing.

## Key Facts

- Building two EV battery plants ~70 miles northeast of Atlanta
- $2.6 billion investment in US battery business
- Will create 2,600+ permanent jobs by 2024
- First plant scheduled for initial operations in 2021, mass production 2022
- Second plant expected for 2023

## Related Companies

- Duckyang (battery modules supplier, $10M Jackson County investment)
- Ford (BlueOval SK joint venture, 2025)
- ...
```

## How It Works

### Ingestion (What Happens When You Run `ingest`)

For each document:

1. **Extraction**: Claude reads the document and extracts:
   - Main entity (company, product, location)
   - Entity type classification
   - 5-10 key facts
   - Related entities
   - Document category

2. **Page Updates**:
   - Main entity page created/updated with new facts
   - Related entity pages updated with cross-references
   - Document ID added to `sources` list

3. **Index Update**:
   - Page metadata stored in `_index.json`
   - Entity relationships tracked
   - Source document marked as processed (no duplicates)

### Search

```python
wiki = LLMWiki("kb/wiki")
results = wiki.search("investment Georgia")  # Scores title + entity_type matches
```

Returns top-k pages ranked by relevance:
```python
{
  "title": "SK Innovation",
  "entity_type": "company",
  "preview": "South Korea's largest energy...",
  "score": 15,
  "sources": 5  # documents mentioning this entity
}
```

### Integration with LLM Pipeline

When answering questions, you get:

```
Question: "What companies are investing in Georgia EV?"

1. Wiki Search (fast, pre-synthesized)
   ✓ SK Innovation
   ✓ Duckyang
   ✓ Vanderlande

2. Hybrid Retrieval (parallel BM25 + dense)
   ✓ 250 child chunks (BM25)
   ✓ 250 child chunks (dense)
   ↓ Merge & deduplicate
   ↓ Cross-encoder rerank
   ✓ Top 45 parent chunks

3. Combined Context → LLM
   - Synthesized wiki facts (consistent, cross-referenced)
   - Raw document context (detailed, comprehensive)
```

## Performance Notes

### Speed

- **Wiki Search**: ~10ms (in-memory index)
- **Hybrid Retrieval**: ~500-1000ms (BM25 + pgvector + rerank)
- **Wiki Ingest**: ~100-500ms per document (depends on LLM latency)

### Cost

- **Ingest**: Local Ollama (zero API costs!) 
  - Computation cost: CPU/GPU on your machine
  - No external API charges
- **Query**: No incremental cost (wiki is pre-built)

### Quality

- **Completeness**: Hybrid retrieval still finds all raw facts
- **Clarity**: Wiki provides high-level summaries and cross-references
- **Freshness**: Re-ingest to update pages with new documents

## Advanced Usage

### Custom Entity Types

Modify the prompt in `llm_wiki.py:ingest_document()` to extract different entity types:

```python
# Current: company, product, location, concept
# Could add: person, event, funding_round, regulation, etc.
```

### Periodic Updates

Set up a cron job to periodically re-ingest new documents:

```bash
# Ingest only new documents (marked ingestion_status='new')
python -m georgia_ev_intelligence.kb_builder.wiki_cli ingest \
  --source kb/raw_docs/ddg_search.jsonl
```

### Wiki Validation

Add a "lint" function to detect contradictions:

```python
wiki.lint()  # Finds pages with conflicting facts
```

### Multi-Source Wiki

Ingest from multiple JSONL files:

```bash
python -m georgia_ev_intelligence.kb_builder.wiki_cli ingest \
  --source kb/raw_docs/company_sites.jsonl

python -m georgia_ev_intelligence.kb_builder.wiki_cli ingest \
  --source kb/raw_docs/news.jsonl
```

Pages automatically merge and cross-reference.

## Troubleshooting

### "Error: Connection refused" or "Cannot connect to Ollama"

Make sure Ollama is running:
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run wiki ingestion
python -m georgia_ev_intelligence.kb_builder.wiki_cli ingest ...
```

### "Model not found: qwen2.5:32b"

Pull the model first:
```bash
ollama pull qwen2.5:32b
# Or use a different model:
# ollama pull llama2
# ollama pull mistral
```

### Wiki is empty after ingest

Check the logs. Common reasons:
- Documents were already processed (check `_index.json`)
- LLM extraction failed (check error messages)
- JSONL file path incorrect

To reset and re-ingest:
```bash
rm -rf kb/wiki/
python -m georgia_ev_intelligence.kb_builder.wiki_cli ingest ...
```

### Too many API calls

The wiki makes 1 Claude call per document. To reduce:
- Use `--limit 10` for testing
- Batch ingest: process in chunks, save time between runs
- Skip already-processed docs (automatic via index)

## Comparison: Wiki vs. Hybrid Retrieval

| Aspect | Wiki | Hybrid Retrieval |
|---|---|---|
| **Source** | Pre-synthesized pages | Raw documents |
| **Query time** | ~10ms | ~1000ms |
| **Completeness** | Summaries, cross-refs | Full text, all details |
| **Update cost** | High (re-ingest) | Low (append to index) |
| **Best for** | Entity lookup, overview | Detailed evidence |
| **Integration** | Complementary | Primary |

**Recommendation**: Use wiki results first (fast overview), then hybrid for detail.

## References

- [Karpathy's LLM-Wiki Gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
- [KB.md](../KB.md) - Full crawler and pipeline documentation
