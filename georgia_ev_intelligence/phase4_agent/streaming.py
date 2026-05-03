"""
phase4_agent/streaming.py
==============================================================
Streaming synthesis — send tokens to the caller as they arrive.

WHY STREAMING:
  Without streaming: user waits 15-30s staring at blank screen
  With streaming:    user sees first word in <1s, reads as it types

HOW IT WORKS:
  Ollama supports stream=True which returns NDJSON lines:
    {"response": "Troup", "done": false}
    {"response": " County", "done": false}
    {"response": " has", "done": false}
    ...
    {"response": ".", "done": true, "total_duration": 12000000000}

  We yield each token as a string. Callers can:
    - Print to terminal (smoke test)
    - Push via Server-Sent Events (FastAPI chatbot)
    - Collect into a string (backward compat)

USAGE:
    # Collect into string (backward compat):
    answer = collect_stream(stream_answer(question, context, cfg))

    # Stream to terminal:
    for token in stream_answer(question, context, cfg):
        print(token, end="", flush=True)

    # Stream via FastAPI SSE:
    async def chat_endpoint(question: str):
        return StreamingResponse(
            sse_stream(question, context, cfg),
            media_type="text/event-stream"
        )
"""
from __future__ import annotations

import json
from typing import Generator

import httpx

from shared.config import Config
from shared.logger import get_logger

logger = get_logger("phase4.streaming")

ANSWER_PROMPT = """You are an expert analyst for the Georgia EV Supply Chain Intelligence System.

QUESTION: {question}

RETRIEVED DATA (pre-filtered by the pipeline to match the question):
{context}

CRITICAL INSTRUCTIONS — read carefully before answering:
1. The RETRIEVED DATA above has ALREADY been filtered by SQL/graph queries to match the question.
   You MUST treat every row in the table as a valid answer. Do NOT re-apply question criteria.
2. Read EVERY single row in the table from top to bottom. Do not stop reading early.
3. List ALL companies shown in the table — do not skip, omit, or summarize any.
4. Include exact values: company name, tier, role, county, and employment number.
5. If the table shows employment by county, identify the county with the highest total.
6. Do NOT say "database does not contain" or "not found" if ANY rows appear in the table above.
7. Only say "No companies found" if the retrieved data says "No matching companies found." literally.
8. Analyze risks, relationships, or trends based only on companies listed — no external knowledge.
9. Be factual and concise. Use exact numbers.

Answer:"""




def _budget_tokens(context: str) -> int:
    """
    Dynamic token budget for synthesis output.
    WHY: A fixed 400-token cap works for 3-4 companies but cuts off at 10+.
    Each company row needs ~50 output tokens to describe clearly.
    We count pipe-separated rows (company table) + a base overhead.

    Budget = max(300, min(700, rows * 50 + 200))
    - 300 minimum: enough for a 2-sentence answer or empty result
    - 700 maximum: ~25s synthesis time cap (qwen2.5:7b generates ~28 tok/s)
    """
    rows = sum(1 for line in context.splitlines()
               if " | " in line and not line.strip().startswith("Company"))
    budget = max(300, min(700, rows * 50 + 200))
    logger.debug("Token budget: %d (rows=%d)", budget, rows)
    return budget


def stream_answer(
    question: str,
    context: str,
    timeout: float = 120.0,
) -> Generator[str, None, None]:
    """
    Stream synthesis tokens from the LLM.
    Yields one token string at a time as the model generates.

    Args:
        question: The user's question
        context:  Retrieved data context (DB results)
        timeout:  HTTP timeout in seconds

    Yields:
        str — each token as it arrives from Ollama
    """
    cfg = Config.get()
    prompt = ANSWER_PROMPT.format(
        question=question,
        context=context[:3500],   # same cap as non-streaming path
    )
    payload = {
        "model":  cfg.ollama_llm_model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.05,
            "num_predict": _budget_tokens(context),  # dynamic: scales with row count
            "num_ctx":     6144,
        },

    }

    url = f"{cfg.ollama_base_url}/api/generate"
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break
    except Exception as exc:
        logger.error("Streaming failed: %s", exc)
        yield f"[Error: {exc}]"


def collect_stream(gen: Generator[str, None, None]) -> str:
    """
    Collect a streaming generator into a single string.
    Use this for backward compatibility with non-streaming callers.
    """
    return "".join(gen)


def stream_answer_collected(question: str, context: str) -> str:
    """
    Non-streaming interface that internally uses streaming.
    Exactly equivalent to the old _generate() method but uses
    streaming under the hood (better timeout handling per-token
    instead of waiting for full response).
    """
    return collect_stream(stream_answer(question, context))


# ── FastAPI SSE helper (for chatbot API) ──────────────────────────────────────

def sse_format(token: str) -> str:
    """Format a token as a Server-Sent Event line."""
    # SSE format: "data: <content>\n\n"
    # Frontend reads: eventSource.onmessage = (e) => { output += e.data }
    escaped = token.replace("\n", "\\n")
    return f"data: {escaped}\n\n"


def stream_sse(question: str, context: str) -> Generator[str, None, None]:
    """
    Yield SSE-formatted tokens for FastAPI StreamingResponse.

    Usage in FastAPI:
        from fastapi.responses import StreamingResponse
        from phase4_agent.streaming import stream_sse

        @app.post("/chat/stream")
        def chat_stream(question: str, context: str):
            return StreamingResponse(
                stream_sse(question, context),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
    """
    for token in stream_answer(question, context):
        yield sse_format(token)
    # Send a final done event so the client knows to stop
    yield "data: [DONE]\n\n"
