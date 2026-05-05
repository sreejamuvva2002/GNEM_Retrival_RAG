"""
evaluate/format_runner.py
─────────────────────────────────────────────────────────────────────────────
Runs the 4 evaluation formats with strict bias isolation.

FORMAT 1 — Only RAG:
  - Entity extraction → SQL/Neo4j retrieval → LLM synthesis (context only)
  - Few-shot injection: DISABLED
  - LLM prompt: "Answer ONLY from retrieved rows"

FORMAT 2 — No RAG:
  - LLM only, no retrieval, no context
  - Few-shot injection: DISABLED
  - LLM prompt: "Answer from your pre-trained knowledge"

FORMAT 3 — RAG + Pre-trained (current production):
  - Full Phase 4+5 pipeline
  - Few-shot injection: ENABLED
  - LLM prompt: allows knowledge supplement

FORMAT 4 — RAG + Pre-trained + Web:
  - Full Phase 4+5 + Tavily web search
  - Few-shot injection: ENABLED
  - LLM prompt: DB context + web context combined
"""
from __future__ import annotations

import json
import time
from typing import Any

import httpx

from shared.config import Config
from shared.logger import get_logger

logger = get_logger("evaluate.format_runner")

_cfg = Config.get()


# ── Format 1: Only RAG — strict context-only synthesis ───────────────────────

_F1_PROMPT = """You are an expert analyst for the Georgia EV Supply Chain Intelligence System.

QUESTION: {question}

RETRIEVED DATA (from PostgreSQL/Neo4j — pre-filtered to match the question):
{context}

STRICT INSTRUCTION: Answer ONLY using the data above.
- Do NOT use any knowledge from your training data.
- Do NOT guess or infer beyond what the rows show.
- Copy company names exactly as shown.
- If the data does not contain enough information, say "Insufficient data in retrieved context."
- This table has exactly {row_count} {row_noun}. List all {row_count}.

Answer:"""


def run_format1(question: str, pipeline, model: str | None = None) -> dict[str, Any]:
    """
    Format 1: Pure RAG — retrieved context + strict synthesis.
    Few-shot: DISABLED (measures pure retrieval quality).
    """
    from phase4_agent.entity_extractor import extract
    import dataclasses

    t0 = time.monotonic()
    entities = extract(question)

    # Get context using pipeline's retriever (bypasses few-shot)
    context, cypher_used = pipeline._retrieve(question, entities)

    if context.startswith("__DIRECT_ANSWER__:"):
        answer = context[len("__DIRECT_ANSWER__:"):]
        retrieved_count = 0
    else:
        # Count rows
        row_count = sum(
            1 for line in context.splitlines()
            if " | " in line and not line.strip().startswith("Company")
        )
        row_noun = "company" if row_count == 1 else "companies"

        prompt = _F1_PROMPT.format(
            question=question,
            context=context[:3500],
            row_count=row_count if row_count > 0 else "unknown",
            row_noun=row_noun,
        )
        answer = _call_llm(prompt, max_tokens=600, model=model)
        retrieved_count = row_count

    elapsed = time.monotonic() - t0
    return {
        "format": "F1_ONLY_RAG",
        "question": question,
        "answer": answer,
        "retrieved_context": context,
        "retrieved_count": retrieved_count,
        "entities": dataclasses.asdict(entities),
        "elapsed_s": round(elapsed, 1),
        "few_shot_used": False,
    }


# ── Format 2: No RAG — pure pre-trained knowledge ────────────────────────────

_F2_PROMPT = """You are an expert on the Georgia EV supply chain ecosystem.

QUESTION: {question}

Answer this question using your pre-trained knowledge about Georgia's EV supply chain,
automotive manufacturers, battery suppliers, and related industries.
Be specific about company names, locations, and tiers where you know them.
If you are uncertain, say so clearly rather than guessing.

Answer:"""


def run_format2(question: str, model: str | None = None) -> dict[str, Any]:
    """
    Format 2: No RAG — pure LLM pre-trained knowledge.
    No retrieval. No context. No few-shot.
    Baseline: shows what model knows without your pipeline.
    """
    t0 = time.monotonic()
    prompt = _F2_PROMPT.format(question=question)
    answer = _call_llm(prompt, max_tokens=500, model=model)
    elapsed = time.monotonic() - t0

    return {
        "format": "F2_NO_RAG",
        "question": question,
        "answer": answer,
        "retrieved_context": "",   # no retrieval
        "retrieved_count": 0,
        "entities": {},
        "elapsed_s": round(elapsed, 1),
        "few_shot_used": False,
    }


# ── Format 3: RAG + Pre-trained — Phase 4+5 full system ─────────────────────

_F3_SYSTEM_HINT = (
    "\n\nNote: You may supplement the retrieved data with relevant general "
    "EV industry knowledge (e.g. known OEM partnerships, general supply chain patterns), "
    "but clearly mark any supplemented information as '[General knowledge]'."
)


def run_format3(question: str, pipeline) -> dict[str, Any]:
    """
    Format 3: Full Phase 4+5 pipeline — RAG + Pre-trained knowledge.
    Few-shot: ENABLED (via text_to_sql injection).
    This is the current production configuration.
    """
    t0 = time.monotonic()
    result = pipeline.ask(question)
    result["format"] = "F3_RAG_PRETRAINED"
    result["few_shot_used"] = True
    result["elapsed_s"] = round(time.monotonic() - t0, 1)
    return result


# ── Format 4: RAG + Pre-trained + Web ────────────────────────────────────────

_F4_PROMPT = """You are an expert analyst for the Georgia EV Supply Chain Intelligence System.

QUESTION: {question}

RETRIEVED DATA (from internal PostgreSQL/Neo4j database):
{db_context}

WEB SEARCH RESULTS (real-time, from Tavily):
{web_context}

INSTRUCTIONS:
1. Prioritize the RETRIEVED DATA (internal DB) for company-specific facts.
2. Use WEB SEARCH RESULTS to supplement with recent news or context not in the DB.
3. Copy company names exactly. Mark web-sourced information as '[Web]'.
4. The table has {row_count} {row_noun}. List all {row_count} from the DB.
5. Be factual and cite which source each fact comes from.

Answer:"""


def run_format4(question: str, pipeline, model: str | None = None) -> dict[str, Any]:
    """
    Format 4: RAG + Pre-trained + Web (Tavily enrichment).
    Few-shot: ENABLED.
    Adds real-time web search results alongside DB context.
    """
    import dataclasses
    from phase4_agent.entity_extractor import extract

    t0 = time.monotonic()

    # Step 1: Get DB context (same as F3)
    entities = extract(question)
    db_context, cypher_used = pipeline._retrieve(question, entities)

    # Step 2: Web search via Tavily
    web_context = _tavily_search(question)

    if db_context.startswith("__DIRECT_ANSWER__:"):
        # For direct answers (SPOF etc.), just append web context as note
        answer = db_context[len("__DIRECT_ANSWER__:"):]
        if web_context:
            answer += f"\n\n[Web context]: {web_context[:500]}"
        retrieved_count = 0
    else:
        row_count = sum(
            1 for line in db_context.splitlines()
            if " | " in line and not line.strip().startswith("Company")
        )
        row_noun = "company" if row_count == 1 else "companies"
        prompt = _F4_PROMPT.format(
            question=question,
            db_context=db_context[:2500],
            web_context=web_context[:800] if web_context else "No web results available.",
            row_count=row_count if row_count > 0 else "unknown",
            row_noun=row_noun,
        )
        answer = _call_llm(prompt, max_tokens=700, model=model)
        retrieved_count = row_count

    elapsed = time.monotonic() - t0
    return {
        "format": "F4_RAG_PRETRAINED_WEB",
        "question": question,
        "answer": answer,
        "retrieved_context": db_context,
        "web_context": web_context,
        "retrieved_count": retrieved_count,
        "entities": dataclasses.asdict(entities),
        "elapsed_s": round(elapsed, 1),
        "few_shot_used": True,
    }


# ── Shared utilities ─────────────────────────────────────────────────────────

def _call_llm(prompt: str, max_tokens: int = 500, model: str | None = None) -> str:
    """Call Ollama LLM with the given prompt. Returns response text."""
    cfg = Config.get()
    llm_model = model or cfg.ollama_llm_model
    payload = {
        "model": llm_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.05,
            "num_predict": max_tokens,
            "num_ctx": 6144,
        },
    }
    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(f"{cfg.ollama_base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        return f"[LLM error: {exc}]"


def _tavily_search(question: str) -> str:
    """
    Run Tavily web search for the question.
    Returns formatted string of top results.
    """
    try:
        from tavily import TavilyClient
        cfg = Config.get()
        if not getattr(cfg, "tavily_api_key", None):
            return ""
        client = TavilyClient(api_key=cfg.tavily_api_key)
        results = client.search(
            query=f"Georgia EV supply chain {question}",
            max_results=3,
            search_depth="basic",
        )
        snippets = []
        for r in results.get("results", []):
            title = r.get("title", "")
            content = r.get("content", "")[:300]
            url = r.get("url", "")
            snippets.append(f"[{title}] {content} (source: {url})")
        return "\n\n".join(snippets)
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return ""


def check_few_shot_contamination(
    eval_questions: list[str],
    threshold: float = 0.85,
) -> dict[str, Any]:
    """
    Check if any eval question is too similar to few-shot training examples.
    Returns contamination report.
    Similarity > threshold means the eval question could be 'hinted' by few-shot.
    """
    try:
        from phase5_fewshot.embedder import embed_text
        from phase5_fewshot.qdrant_store import search_similar
    except ImportError:
        return {"error": "Phase 5 not available", "contaminated": []}

    contaminated = []
    for q in eval_questions:
        vec = embed_text(q)
        hits = search_similar(vec, top_k=1)
        if hits and hits[0]["score"] >= threshold:
            contaminated.append({
                "eval_question": q[:80],
                "similar_fewshot": hits[0].get("question", "")[:80],
                "similarity": hits[0]["score"],
            })

    return {
        "total_eval_questions": len(eval_questions),
        "contaminated_count": len(contaminated),
        "contamination_rate": round(len(contaminated) / len(eval_questions), 3),
        "contaminated": contaminated,
        "threshold_used": threshold,
        "verdict": "CLEAN" if len(contaminated) == 0 else
                   "ACCEPTABLE" if len(contaminated) / len(eval_questions) < 0.05 else
                   "WARNING — re-check few-shot/eval question overlap",
    }
