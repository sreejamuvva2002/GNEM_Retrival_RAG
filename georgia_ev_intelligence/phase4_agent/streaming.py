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

_MAX_PROMPT_CONTEXT_CHARS = 12000

_ANSWER_PROMPT_TEMPLATE = """You are an expert analyst for the Georgia EV Supply Chain Intelligence System.

QUESTION: {question}

RETRIEVED DATA (retrieved from the vector knowledge base):
{context}

CRITICAL INSTRUCTIONS:
1. The data above comes from the Georgia EV vector knowledge base. Use it as your source of truth.
2. List them one by one in order. Do not skip, omit, or combine any rows.
3. Copy company names EXACTLY character-by-character from the table — do NOT paraphrase or shorten.
5. For each entry include the fields that matter for the question, especially exact name, tier, role, county, employment, OEMs, facility type, and products when they are present, if they are not present then ignore then (leave it blank).
6. {format_instruction}
7. If the table shows employment by county, identify the highest-total county.
9. say "No companies found" if the context has no companies in it "No matching companies found."

Answer:"""

_FORMAT_COMPACT  = "Use ONLY a numbered list: '1. Name | Tier | Role | County | N employees'. No extra sentences. if (Name or Tire or Role or County etc etc ) information is not available then leave it blank"
_FORMAT_DETAILED = "if the data has tier, role, county, and employment for each entry then include them or include only what you have and leave other as blank."


def prompt_context_for_model(context: str) -> str:
    """Return the exact retrieved-context slice injected into the synthesis prompt."""
    return context[:_MAX_PROMPT_CONTEXT_CHARS]


def count_context_rows(context: str) -> int:
    """Count visible rows in the exact context block sent to the model."""
    row_count = sum(
        1 for line in context.splitlines()
        if " | " in line and not line.strip().startswith("Company")
    )
    if row_count == 0:
        row_count = sum(
            1 for line in context.splitlines()
            if line.strip() and not line.startswith("[") and ":" in line
            and not line.startswith("Total")
        )
    return row_count


def build_answer_prompt(question: str, context: str) -> str:
    """
    Build the synthesis prompt with:
    - Explicit row count in instruction #2 (prevents LLM stopping at row 1)
    - Company name faithfulness instruction (prevents Racemark → Mark hallucination)
    - Compact format for >10 rows (reduces T5-style 56s → ~22s by cutting prose tokens)

    The row count is computed from the context (pipe-separated rows, excluding header).
    For county aggregate data (no pipe rows) the count is computed from newlines.
    """
    prompt_context = prompt_context_for_model(context)
    row_count = count_context_rows(prompt_context)

    row_noun = "company" if row_count == 1 else "companies"
    # For large lists: force compact numbered format to reduce token generation time
    format_instruction = _FORMAT_COMPACT if row_count > 10 else _FORMAT_DETAILED

    return _ANSWER_PROMPT_TEMPLATE.format(
        question=question,
        context=prompt_context,
        row_count=row_count if row_count > 0 else "unknown number of",
        row_noun=row_noun,
        format_instruction=format_instruction,
    )


def _budget_tokens(context: str) -> int:
    """
    Dynamic token budget for synthesis output.

    Budget = max(300, min(900, rows * 40 + 200))
    - 300 minimum: enough for a 2-sentence answer or empty result
    - 900 maximum: covers 15+ companies in compact list format (~35 tok/company)
    - Rows >10 use compact format (40 tok/company instead of 50)
    """
    rows = sum(1 for line in context.splitlines()
               if " | " in line and not line.strip().startswith("Company"))
    # Compact format for large lists uses fewer tokens per row
    tok_per_row = 35 if rows > 10 else 50
    budget = max(300, min(900, rows * tok_per_row + 200))
    logger.debug("Token budget: %d (rows=%d, tok/row=%d)", budget, rows, tok_per_row)
    return budget


def stream_answer(
    question: str,
    context: str,
    timeout: float = 120.0,
    model: str | None = None,    # overrides cfg.ollama_llm_model when set
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
    prompt = build_answer_prompt(question, context)

    payload = {
        "model":  model or cfg.ollama_llm_model,  # model_override takes priority
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.05,
            "num_predict": _budget_tokens(context),
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


def stream_answer_collected(question: str, context: str, model: str | None = None) -> str:
    """
    Non-streaming interface — collects all tokens into a string.
    Pass model= to override the default LLM (e.g. 'gemma2:9b' for eval).
    """
    return collect_stream(stream_answer(question, context, model=model))


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
