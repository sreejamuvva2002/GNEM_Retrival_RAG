"""
FastAPI REST API for the Georgia EV Intelligence pipeline.

Endpoints:
  POST /ask          — structured pipeline result as JSON
  GET  /stream       — SSE streaming (yields answer tokens)
  GET  /health       — health check
"""
from __future__ import annotations
import json
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import pipeline, synthesizer, config
from .semantic_retriever import retriever_backend_label

app = FastAPI(title="Georgia EV Intelligence", version="2.0")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_backend": "anthropic" if config.USE_ANTHROPIC else "ollama",
        "retriever_backend": retriever_backend_label(),
    }


@app.post("/ask")
def ask(req: AskRequest):
    result = pipeline.run(req.question)
    return {
        "question": result.question,
        "answer": result.answer,
        "support_level": result.support_level,
        "hallucination_risk": result.hallucination_risk,
        "retrieval_method": result.retrieval_method,
        "evidence_count": result.evidence_count,
        "intent": result.intent,
        "filters_applied": result.filters_applied,
        "key_terms_matched": result.key_terms_matched,
        "evidence_rows": result.evidence_rows,
    }


@app.get("/stream")
def stream(q: str = Query(..., description="Question to answer")):
    def generate():
        result = pipeline.run(q)
        # Yield metadata first
        meta = {
            "support_level": result.support_level,
            "hallucination_risk": result.hallucination_risk,
            "evidence_count": result.evidence_count,
        }
        yield f"data: {json.dumps({'type': 'meta', 'data': meta})}\n\n"
        # Stream answer word-by-word
        for word in result.answer.split():
            yield f"data: {json.dumps({'type': 'token', 'data': word + ' '})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
