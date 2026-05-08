"""
Phase 6 — FastAPI Streaming API
─────────────────────────────────────────────────────────────────────────────
REST API with Server-Sent Events (SSE) streaming for the EV Intelligence chatbot.

ENDPOINTS:
  POST /ask              — non-streaming (for simple clients / RAGAS eval)
  GET  /stream?q=...     — SSE streaming (for browser UI)
  GET  /health           — health check
  GET  /entities?q=...   — debug: show extracted entities for a question

USAGE:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

  Then open: http://localhost:8000
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core_agent.agent_pipeline import EVAgent
from filters_and_validation.query_entity_extractor import extract
from core_agent.streaming import stream_answer, build_answer_prompt, _budget_tokens
from shared.config import Config
from shared.logger import get_logger

logger = get_logger("api.main")

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Georgia EV Intelligence API",
    description="AI-powered Georgia EV Supply Chain Intelligence System",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded agent (initialized on first request)
_agent: EVAgent | None = None


def get_agent() -> EVAgent:
    global _agent
    if _agent is None:
        _agent = EVAgent()
    return _agent


# ── Request / Response Models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question:          str
    answer:            str
    retrieved_context: str
    entities:          dict
    retrieved_count:   int
    elapsed_s:         float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — verifies DB/LLM connectivity."""
    return {
        "status":    "ok",
        "model":     Config.get().ollama_llm_model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@app.get("/entities")
def debug_entities(q: str = Query(..., description="Question to parse")):
    """Debug endpoint: show extracted entities for a question."""
    e = extract(q)
    return {
        "question":      q,
        "tier":          e.tier,
        "county":        e.county,
        "oem":           e.oem,
        "industry_group":e.industry_group,
        "ev_role":       e.ev_role,
        "ev_role_list":  e.ev_role_list,
        "facility_type": e.facility_type,
        "product_kw":    e.product_keywords,
        "is_aggregate":  e.is_aggregate,
        "is_risk_query": e.is_risk_query,
        "is_oem_dep":    e.is_oem_dependency,
        "is_capacity":   e.is_capacity_risk,
        "is_misalign":   e.is_misalignment,
        "is_top_n":      e.is_top_n,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Non-streaming question answering.
    Returns complete answer once ready (~10-45s depending on complexity).
    Better for programmatic use (RAGAS eval, batch processing).
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = get_agent().ask(req.question)
    return AskResponse(**result)


@app.get("/stream")
async def stream_question(q: str = Query(..., description="Question to answer")):
    """
    SSE streaming endpoint.
    Streams answer tokens as they are generated.
    Client receives: data: <token>\\n\\n
    Ends with:       data: [DONE]\\n\\n

    Frontend usage:
        const es = new EventSource('/stream?q=...');
        es.onmessage = (e) => { output += e.data === '[DONE]' ? '' : e.data; };
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    async def event_generator() -> AsyncGenerator[str, None]:
        agent = get_agent()

        # Step 1: Retrieve (sync, but fast)
        from filters_and_validation.query_entity_extractor import extract as _extract
        entities = _extract(q)
        context, _ = agent._retrieve(q, entities)

        # Step 2: If direct answer (SPOF), stream it word by word
        if context.startswith("__DIRECT_ANSWER__:"):
            answer = context[len("__DIRECT_ANSWER__:"):]
            for word in answer.split():
                yield f"data: {word} \n\n"
                await asyncio.sleep(0.01)
            yield "data: [DONE]\n\n"
            return

        # Step 3: Stream LLM synthesis tokens
        # Run the blocking stream_answer() in a thread to avoid blocking event loop
        loop = asyncio.get_event_loop()
        token_queue: asyncio.Queue = asyncio.Queue()

        def _stream_worker():
            try:
                for token in stream_answer(q, context):
                    loop.call_soon_threadsafe(token_queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(token_queue.put_nowait, None)  # sentinel

        import threading
        thread = threading.Thread(target=_stream_worker, daemon=True)
        thread.start()

        while True:
            token = await token_queue.get()
            if token is None:
                break
            escaped = token.replace("\n", "\\n")
            yield f"data: {escaped}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":     "keep-alive",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    """Serve the chat UI."""
    try:
        from pathlib import Path
        html = Path("api/static/index.html").read_text(encoding="utf-8")
        return HTMLResponse(content=html)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>UI not built yet. Use /ask or /stream endpoints.</h1>")
