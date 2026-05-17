"""
FastAPI REST API for the Georgia EV Intelligence pipeline.

Endpoints:
  POST /ask          — structured pipeline result as JSON
  GET  /health       — health check
"""
from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel

from ...shared import config
from .. import pipeline

app = FastAPI(title="Georgia EV Intelligence", version="3.0")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_backend": "ollama",
        "retrieval_method": "hybrid_rrf",
    }


@app.post("/ask")
def ask(req: AskRequest):
    result = pipeline.run(req.question)
    return {
        "question": result.question,
        "answer": result.answer,
        "parent_contexts_used": result.parent_contexts_used,
        "retrieval_method": result.retrieval_method,
        "used_citations": [
            asdict(c) for c in result.citations.used_citations
        ],
        "all_source_records": [
            asdict(c) for c in result.citations.all_source_records
        ],
        "latency": result.trace.latency,
        "errors": result.trace.errors,
    }
