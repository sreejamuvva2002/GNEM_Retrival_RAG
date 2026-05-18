"""LLM client for the phrase classifier."""
from __future__ import annotations

import logging
import os
import re

import requests

from ...shared import config

logger = logging.getLogger(__name__)

# Settings with safe defaults — read at call time so tests can patch env.
_DEFAULT_TIMEOUT = 960


def _is_enabled() -> bool:
    return os.environ.get("PHRASE_CLASSIFIER_ENABLED", "true").lower() == "true"


def _model() -> str:
    return os.environ.get("PHRASE_CLASSIFIER_MODEL", config.OLLAMA_LLM_MODEL)


def _timeout() -> int:
    return int(os.environ.get("PHRASE_CLASSIFIER_TIMEOUT", str(_DEFAULT_TIMEOUT)))


def call_classifier_llm(prompt: str, timeout: int | None = None) -> str:
    """Call the LLM for phrase classification.

    Returns the raw text response, or empty string on failure/disabled.
    Never raises — callers should treat empty string as "use fallback".
    """
    if not _is_enabled():
        logger.info("Phrase classifier is disabled (PHRASE_CLASSIFIER_ENABLED=false)")
        return ""

    effective_timeout = timeout or _timeout()
    model = _model()

    try:
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 2048,
                },
            },
            timeout=effective_timeout,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "").strip()
        return _clean_response(answer)
    except requests.Timeout:
        logger.warning("Phrase classifier LLM call timed out after %ds", effective_timeout)
        return ""
    except Exception:
        logger.warning("Phrase classifier LLM call failed", exc_info=True)
        return ""


def _clean_response(text: str) -> str:
    """Remove thinking traces and markdown fences."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:json|text)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned
