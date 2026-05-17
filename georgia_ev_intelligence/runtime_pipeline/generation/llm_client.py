"""LLM client for answer generation using local Ollama."""
from __future__ import annotations

import re

import requests

from ...shared import config


def generate_answer(prompt: str, timeout: int = 180) -> str:
    """Generate an answer using the local Ollama model.

    Args:
        prompt: The full prompt including system instruction, context, and question.
        timeout: Request timeout in seconds.

    Returns:
        The generated answer text, cleaned of common artifacts.
    """
    model = config.OLLAMA_LLM_MODEL

    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 4096,
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()

    answer = resp.json().get("response", "").strip()
    return _clean_answer(answer)


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
