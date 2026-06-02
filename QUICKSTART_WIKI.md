# LLMWiki Quick Start

Get started with the LLM-based wiki in 5 minutes.

## Prerequisites

✅ Ollama running: `ollama serve` (in separate terminal)
✅ `.env` with Ollama config (already set up):
   - `OLLAMA_BASE_URL="http://localhost:11434"` 
   - `OLLAMA_LLM_MODEL="qwen2.5:32b"` (or your preferred model)
✅ `ddg_search.jsonl` in `kb/raw_docs/`

## 1. Ingest Documents (Choose One)

### Test First (2-5 minutes)
```bash
python -m georgia_ev_intelligence.kb_builder.wiki_cli ingest \
  --source kb/raw_docs/ddg_search.jsonl \
  --limit 20
```

Output:
```
[wiki] Processing 1: Foreign Investment Travels Nonstop to Georgia...
      → Updated 3 pages: SK Innovation, Duckyang, Georgia
...
[wiki] Done!
[wiki] Documents processed: 20
[wiki] Page updates: 67
[wiki] Total pages: 31
```

### Full Ingestion (Variable time)
```bash
python -m georgia_ev_intelligence.kb_builder.wiki_cli ingest \
  --source kb/raw_docs/ddg_search.jsonl
```

## 2. Explore Wiki

```bash
# Show stats
python -m georgia_ev_intelligence.kb_builder.wiki_cli stats

# Search
python -m georgia_ev_intelligence.kb_builder.wiki_cli search "SK Innovation"

# View page
python -m georgia_ev_intelligence.kb_builder.wiki_cli show "SK Innovation"

# List all companies
python -m georgia_ev_intelligence.kb_builder.wiki_cli list --type company

# Export to markdown
python -m georgia_ev_intelligence.kb_builder.wiki_cli export \
  --output outputs/gnem_wiki.md
```

## 3. Use in Your Code

```python
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.wiki_retriever import WikiRetriever

wiki = WikiRetriever(wiki_dir="kb/wiki")

# Search
results = wiki.retrieve("SK Innovation", top_k=3)
for result in results:
    print(f"{result.title}: {result.content[:100]}...")

# Get a specific page
page = wiki.get_page("SK Innovation")
print(page.content)

# Format for LLM context
context = wiki.format_for_context(results)
print(context)
```

## 4. Integrate with Hybrid Retrieval

```python
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.orchestrator import HybridRetrievalOrchestrator
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.wiki_retriever import WikiRetriever
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.wiki_orchestrator import WikiEnhancedOrchestrator

# Initialize
hybrid = HybridRetrievalOrchestrator(...)
wiki = WikiRetriever()
enhanced = WikiEnhancedOrchestrator(hybrid, wiki)

# Use
result = enhanced.retrieve_enhanced("What companies are investing in Georgia?")
print(result.combined_context)  # Has wiki + hybrid results
```

## Commands Reference

| Command | What it does |
|---|---|
| `ingest` | Process documents, build wiki pages |
| `search` | Find pages by keyword |
| `show` | Display a full page |
| `list` | List all pages (optionally by type) |
| `export` | Export entire wiki to markdown |
| `stats` | Show wiki statistics |

## Examples

### Search for companies
```bash
python -m georgia_ev_intelligence.kb_builder.wiki_cli search "battery" --top-k 10
```

### List all companies
```bash
python -m georgia_ev_intelligence.kb_builder.wiki_cli list --type company
```

### Show a company page
```bash
python -m georgia_ev_intelligence.kb_builder.wiki_cli show "Duckyang"
```

### Export and read the full wiki
```bash
python -m georgia_ev_intelligence.kb_builder.wiki_cli export \
  --output outputs/gnem_wiki.md

# Then open outputs/gnem_wiki.md
```

## File Locations

After ingestion, you'll have:

```
kb/wiki/
  _index.json              ← Master index
  sk_innovation.md         ← Entity pages
  duckyang.md
  georgia.md
  (... one per entity)
```

## Troubleshooting

**Q: Import error: `No module named 'anthropic'`**
```bash
pip install -r requirements.txt
```

**Q: No such file or directory: `kb/raw_docs/ddg_search.jsonl`**
Check that the file exists and you're running from the repo root.

**Q: API Key error**
```bash
# Check .env
cat .env | grep ANTHROPIC_API_KEY

# If missing, add it
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
```

**Q: Wiki is empty after ingest**
Try resetting:
```bash
rm -rf kb/wiki/
python -m georgia_ev_intelligence.kb_builder.wiki_cli ingest --source kb/raw_docs/ddg_search.jsonl --limit 5
```

## Next Steps

1. ✅ Test with `--limit 20`
2. ✅ Explore with `search`, `show`, `list`
3. ✅ Full ingestion
4. ✅ Integrate with answer generation pipeline
5. ✅ See `docs/LLM_WIKI.md` for advanced usage

---

**Full Documentation**: See `docs/LLM_WIKI.md`
