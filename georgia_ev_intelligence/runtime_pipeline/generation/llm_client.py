"""LLM client for answer generation using local Ollama."""
from __future__ import annotations

import re
import time

import requests

from ...shared import config
from ..debug_trace import record_step


def generate_answer(prompt: str, timeout: int = 1800) -> str:
    """Generate an answer using the local Ollama model.

    Args:
        prompt: The full prompt including system instruction, context, and question.
        timeout: Request timeout in seconds.

    Returns:
        The generated answer text, cleaned of common artifacts.
    """
    model = config.OLLAMA_LLM_MODEL
    options = {
        "temperature": config.OLLAMA_TEMPERATURE,
        "top_p": config.OLLAMA_TOP_P,
        "num_predict": config.OLLAMA_NUM_PREDICT,
    }
    started = time.perf_counter()

    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                # Disable chain-of-thought for reasoning models (e.g. qwen3): we want
                # a direct, fast answer for the JSON-output RAG prompt, not reasoning
                # that consumes the whole num_predict budget. No-op for non-thinking
                # models (llama3, qwen2.5), so it's safe regardless of OLLAMA_LLM_MODEL.
                "think": False,
                "options": options,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        cleaned = _clean_answer(raw)
    except Exception as exc:
        record_step(
            "llm_call",
            status="error",
            summary=f"Ollama call failed: {exc}",
            error=str(exc),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            details={
                "model": model,
                "options": options,
                "timeout": timeout,
                "prompt_chars": len(prompt or ""),
                "prompt": prompt,
            },
        )
        raise

    record_step(
        "llm_call",
        status="ok",
        summary=f"{model}: prompt {len(prompt or '')} chars → response {len(cleaned)} chars",
        duration_ms=(time.perf_counter() - started) * 1000.0,
        details={
            "model": model,
            "options": options,
            "timeout": timeout,
            "http_status": resp.status_code,
            "prompt_chars": len(prompt or ""),
            "response_chars": len(cleaned),
            "prompt": prompt,
            "raw_response": raw,
            "cleaned_response": cleaned,
        },
    )
    return cleaned


def _clean_answer(answer: str) -> str:
    """Remove common local-model artifacts without changing factual content."""
    if not answer:
        return ""

    cleaned = answer.strip()

    # Remove thinking traces
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()

    # Remove accidental markdown code fences around normal prose
    cleaned = re.sub(r"^```(?:text|markdown)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    return cleaned
