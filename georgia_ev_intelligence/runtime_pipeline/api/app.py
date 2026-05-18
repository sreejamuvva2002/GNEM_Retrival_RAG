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

from .. import pipeline

RETRIEVAL_METHOD = "hybrid_rrf"

app = FastAPI(title="Georgia EV Intelligence", version="3.1")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_backend": "ollama",
        "retrieval_method": RETRIEVAL_METHOD,
    }


@app.post("/ask")
def ask(req: AskRequest):
    result = pipeline.run(req.question)
    trace = result.trace
    total_latency = sum(trace.latency.values()) if trace.latency else 0.0

    return {
        "question": result.question,
        "answer": result.answer,
        "retrieval_method": RETRIEVAL_METHOD,
        "used_citations": [asdict(c) for c in result.citations.used_citations],
        "all_source_records": [asdict(c) for c in result.citations.all_source_records],
        "retrieval": {
            "backend": RETRIEVAL_METHOD,
            "dense_result_count": trace.dense_result_count,
            "bm25_result_count": trace.bm25_result_count,
            "hybrid_result_count": trace.hybrid_result_count,
            "fetched_parent_count": trace.fetched_parent_count,
            "context_parent_count": trace.llm_context_parent_count,
        },
        "latency": trace.latency,
        "total_latency_seconds": round(total_latency, 4),
        "errors": trace.errors,
    }
