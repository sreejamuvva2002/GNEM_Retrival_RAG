# Georgia EV Supply Chain Intelligence 🔋

> A ChatGPT-like chatbot exclusively for Georgia's Electric Vehicle supply chain.
> Answers any question about 205 EV companies, their relationships, investments, and facilities.

## Architecture

```
205 Companies (GNEM Excel)
        +
Internet Documents (Kb_Enrichment pipeline)
        ↓
Phase 1: Extract → Store in B2 → Parse key facts → PostgreSQL
Phase 2: Chunk (hierarchical) → Embed → Store in Qdrant
Phase 3: Build Neo4j Knowledge Graph
Phase 4: LLM Agent (Qwen3:14b) + Hybrid Retriever + Tavily fallback
Phase 5: ChatGPT-like Chat UI
```

## Quick Start

### 1. Fill in your .env
```
cp .env .env.backup  # keep a backup
# Edit .env with your actual API keys
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 3. Run Phase 1
```bash
python -m phase1_extraction.pipeline
```

### 4. Validate Phase 1
```bash
python -m phase1_extraction.validator
```

### 5. Run Phase 2
```bash
python -m phase2_embedding.pipeline
```

### 6. Run Phase 3
```bash
python -m phase3_graph.pipeline
```

### 7. Run Phase 4 + Chat UI
```bash
python -m phase5_ui.main
# Open: http://localhost:8000
```

## Project Structure

```
georgia_ev_intelligence/
├── .env                      ← Your API keys (never commit)
├── requirements.txt
├── config/
│   └── settings.yaml         ← All non-secret config
├── shared/                   ← Shared across all phases
│   ├── db.py                 ← PostgreSQL connection + models
│   ├── storage.py            ← Backblaze B2 access
│   └── config.py             ← Settings loader
├── phase1_extraction/        ← Data collection + extraction
│   ├── extractor.py          ← PDF/HTML/DOCX/CSV text extraction
│   ├── entity_extractor.py   ← Pull facts → PostgreSQL tables
│   ├── validator.py          ← Confirm everything worked
│   └── pipeline.py           ← Run Phase 1 end-to-end
├── phase2_embedding/         ← Chunking + vectors
│   ├── chunker.py            ← Hierarchical parent-child
│   ├── embedder.py           ← nomic-embed-text via Ollama
│   ├── vector_store.py       ← Qdrant upload + search
│   └── pipeline.py
├── phase3_graph/             ← Neo4j knowledge graph
│   ├── graph_schema.py       ← Node/relationship definitions
│   ├── graph_loader.py       ← GNEM Excel → Neo4j
│   ├── graph_enricher.py     ← Extracted facts → Neo4j
│   └── pipeline.py
├── phase4_agent/             ← LLM intelligence
│   ├── retriever.py          ← Hybrid: Neo4j + Qdrant
│   ├── tavily_fallback.py    ← Live search + auto-store
│   ├── router.py             ← Route: structured vs semantic vs live
│   └── agent.py              ← LangGraph orchestration
├── phase5_ui/                ← Chat interface
│   ├── main.py               ← FastAPI app
│   ├── routers/
│   │   └── chat.py           ← /chat WebSocket endpoint
│   └── static/
│       ├── index.html        ← Chat UI
│       ├── style.css         ← Georgia EV branding
│       └── app.js            ← Streaming message handler
└── evaluate/
    ├── run_eval.py           ← Run 50 Q&A through chatbot
    └── score.py              ← Accuracy + hallucination metrics
```

## Keys Needed (fill in .env)

| Key | Where to Get | Free? |
|-----|-------------|-------|
| `TAVILY_API_KEY` | https://app.tavily.com | ✅ 1000/month free |
| `QDRANT_URL` + `QDRANT_API_KEY` | https://cloud.qdrant.io | ✅ 1 cluster free |
| `NEO4J_URI` + `NEO4J_PASSWORD` | https://console.neo4j.io | ✅ AuraDB free |
| All others | Already in Kb_Enrichment/.env | ✅ Already set |
