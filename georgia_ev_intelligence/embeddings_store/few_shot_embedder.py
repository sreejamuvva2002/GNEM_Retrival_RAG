"""
phase5_fewshot/embedder.py
─────────────────────────────────────────────────────────────────────────────
Local embedding using Ollama's nomic-embed-text model.

WHY nomic-embed-text:
  - Runs locally via Ollama (zero cloud cost, zero latency variance)
  - 768-dim vectors, good semantic quality for domain text
  - ~30ms per embedding on CPU — negligible overhead
  - Already available if Ollama is installed (pull with: ollama pull nomic-embed-text)

FALLBACK:
  If nomic-embed-text is not available, falls back to a simple TF-IDF style
  bag-of-words vector so the system still runs (lower quality but functional).
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

import httpx

from shared.config import Config
from shared.logger import get_logger

logger = get_logger("embeddings_store.embedder")

_EMBED_MODEL = "nomic-embed-text"
_EMBED_DIM   = 768


def embed_text(text: str) -> list[float]:
    """
    Embed a single text string using nomic-embed-text via Ollama.
    Returns a 768-dim float vector.
    Falls back to keyword hash vector if Ollama embed API fails.
    """
    cfg = Config.get()
    url = f"{cfg.ollama_base_url}/api/embeddings"
    try:
        resp = httpx.post(
            url,
            json={"model": _EMBED_MODEL, "prompt": text},
            timeout=15.0,
        )
        resp.raise_for_status()
        embedding = resp.json().get("embedding", [])
        if len(embedding) == _EMBED_DIM:
            logger.debug("Embedded %d chars → %d-dim vector", len(text), len(embedding))
            return embedding
        logger.warning("Embed returned wrong dim %d — using fallback", len(embedding))
    except Exception as exc:
        logger.warning("Ollama embed failed (%s) — using keyword fallback", exc)

    return _keyword_fallback_embedding(text)


def _keyword_fallback_embedding(text: str) -> list[float]:
    """
    Deterministic keyword-hash fallback embedding (768-dim).
    Quality is much lower than nomic-embed-text but guarantees the system runs.
    Each of 768 positions is set by hashing word n-grams from the text.
    """
    words = text.lower().split()
    vec = [0.0] * _EMBED_DIM
    for i, word in enumerate(words):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = h % _EMBED_DIM
        vec[idx] += 1.0 / (i + 1)   # position-weighted
    # L2 normalize
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts. Sequential (Ollama doesn't support batched embedding)."""
    return [embed_text(t) for t in texts]


def check_embed_model_available() -> bool:
    """Return True if nomic-embed-text is available in Ollama."""
    cfg = Config.get()
    try:
        resp = httpx.get(f"{cfg.ollama_base_url}/api/tags", timeout=5.0)
        models = [m["name"] for m in resp.json().get("models", [])]
        available = any(_EMBED_MODEL in m for m in models)
        if not available:
            logger.warning(
                "nomic-embed-text not found in Ollama. "
                "Run: ollama pull nomic-embed-text"
            )
        return available
    except Exception:
        return False
