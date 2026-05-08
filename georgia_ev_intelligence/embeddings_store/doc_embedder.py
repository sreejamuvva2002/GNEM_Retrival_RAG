"""
Phase 2 — Embedder

Converts text chunks → configured-dimension vectors using the active Ollama embed model.

WHY THIS MODULE:
  - Uses whichever embedding model is configured in `.env`
  - Runs 100% locally via Ollama — zero cost, zero latency to external API
  - Keeps query and document embeddings aligned by using the same model

BATCHING STRATEGY:
  - Batch size 32 (optimal for Ollama local inference on CPU+GPU)
  - Retry with exponential backoff on Ollama failures
  - Progress tracking for large batches
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from chunking.chunker import Chunk
from shared.config import Config
from shared.logger import get_logger

logger = get_logger("phase2.embedder")

# Batch size for Ollama embedding calls
# 32 is a conservative default for local Ollama embedding
EMBED_BATCH_SIZE = 32

# Ollama embedding endpoint
OLLAMA_EMBED_PATH = "/api/embed"

# Retry settings for Ollama (which can occasionally time out)
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds


def _get_ollama_url() -> str:
    cfg = Config.get()
    return cfg.ollama_base_url.rstrip("/")


def _get_embed_model() -> str:
    cfg = Config.get()
    return cfg.ollama_embed_model  # nomic-embed-text


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using the active Ollama embedding model.

    Args:
        texts: List of text strings to embed

    Returns:
        List of float vectors, one per input text

    Raises:
        RuntimeError: If Ollama is unreachable or returns invalid response
    """
    if not texts:
        return []

    base_url = _get_ollama_url()
    model = _get_embed_model()
    url = f"{base_url}{OLLAMA_EMBED_PATH}"

    all_embeddings: list[list[float]] = []

    # Process in batches
    for batch_start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[batch_start: batch_start + EMBED_BATCH_SIZE]

        for attempt in range(MAX_RETRIES):
            try:
                response = httpx.post(
                    url,
                    json={"model": model, "input": batch},
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()

                # Ollama /api/embed returns {"embeddings": [[...], [...]]}
                embeddings = data.get("embeddings", [])
                if len(embeddings) != len(batch):
                    raise ValueError(
                        f"Expected {len(batch)} embeddings, got {len(embeddings)}"
                    )

                all_embeddings.extend(embeddings)
                logger.debug(
                    "Embedded batch [%d:%d] → %d vectors (dim=%d)",
                    batch_start,
                    batch_start + len(batch),
                    len(embeddings),
                    len(embeddings[0]) if embeddings else 0,
                )
                break

            except httpx.TimeoutException:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        "Ollama embed timeout (attempt %d/%d), retrying in %.1fs",
                        attempt + 1, MAX_RETRIES, wait
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Ollama embedding timed out after {MAX_RETRIES} attempts. "
                        f"Is Ollama running? Check: ollama serve"
                    )

            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"Ollama embed HTTP error {exc.response.status_code}: {exc.response.text[:200]}"
                ) from exc

            except Exception as exc:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAY * (2 ** attempt)
                    logger.warning("Embed error (attempt %d): %s — retrying", attempt + 1, exc)
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Embedding failed after {MAX_RETRIES} attempts: {exc}") from exc

    return all_embeddings


def embed_chunks(chunks: list[Chunk]) -> dict[str, list[float]]:
    """
    Embed a list of Chunk objects.

    Returns:
        Dict mapping chunk_id → embedding vector
    """
    if not chunks:
        return {}

    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]

    logger.info("Embedding %d chunks with %s...", len(chunks), _get_embed_model())
    start = time.monotonic()

    vectors = embed_texts(texts)

    elapsed = time.monotonic() - start
    logger.info(
        "Embedded %d chunks in %.1fs (%.0f chunks/sec)",
        len(vectors), elapsed, len(vectors) / max(elapsed, 0.001)
    )

    return dict(zip(chunk_ids, vectors))


def embed_single(text: str) -> list[float]:
    """
    Embed a single text string. Convenience wrapper for query embedding.

    Args:
        text: Query or document text

    Returns:
        Embedding vector for the configured model
    """
    result = embed_texts([text])
    if not result:
        raise RuntimeError("No embedding returned for input text")
    return result[0]


def verify_ollama_embed() -> dict[str, Any]:
    """
    Verify that Ollama is running and the configured embed model is available.

    Returns:
        Dict with: ok (bool), model (str), dimensions (int), error (str or None)
    """
    try:
        test_text = "Georgia EV supply chain test"
        vector = embed_single(test_text)
        dims = len(vector)
        expected_dims = Config.get().qdrant_dimensions
        ok = dims == expected_dims
        if not ok:
            logger.warning("Unexpected embedding dimensions: %d (expected %d)", dims, expected_dims)
        else:
            logger.info("Ollama embed verified: model=%s, dims=%d", _get_embed_model(), dims)
        return {
            "ok": ok,
            "model": _get_embed_model(),
            "dimensions": dims,
            "error": None if ok else f"Wrong dimensions: {dims} (expected {expected_dims})",
        }
    except Exception as exc:
        logger.error("Ollama embed verification FAILED: %s", exc)
        return {
            "ok": False,
            "model": _get_embed_model(),
            "dimensions": 0,
            "error": str(exc),
        }
