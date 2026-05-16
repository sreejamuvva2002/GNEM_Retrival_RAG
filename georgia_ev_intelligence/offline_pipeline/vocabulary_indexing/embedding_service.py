"""Generate embeddings for vocabulary terms using the shared embedding utilities."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from georgia_ev_intelligence.shared import config
from georgia_ev_intelligence.shared.embeddings import (
    as_document_text,
    load_sentence_transformer,
)

from .models import VocabularyTerm

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def generate_embeddings(
    terms: list[VocabularyTerm],
    model_name: str | None = None,
    batch_size: int | None = None,
) -> tuple[list[VocabularyTerm], int]:
    """Populate term_vector on each VocabularyTerm in-place.

    Parameters
    ----------
    terms : list[VocabularyTerm]
        Terms to embed. Modified in-place.
    model_name : str, optional
        Override embedding model (default from config.EMBEDDING_MODEL).
    batch_size : int, optional
        Override batch size (default from config.PGVECTOR_BATCH_SIZE).

    Returns
    -------
    tuple[list[VocabularyTerm], int]
        The same list of terms (with vectors populated) and vector dimension.
    """
    model_id = model_name or config.EMBEDDING_MODEL
    size = batch_size or config.PGVECTOR_BATCH_SIZE

    logger.info("Loading embedding model: %s", model_id)
    model = load_sentence_transformer(model_id)
    vector_dim = _get_dimension(model)
    logger.info("Vector dimension: %d", vector_dim)

    total = len(terms)
    embedded_count = 0

    for batch_start in range(0, total, size):
        batch = terms[batch_start : batch_start + size]
        texts = [as_document_text(t.normalized_value) for t in batch]

        vectors = model.encode(
            texts,
            batch_size=size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        for i, term in enumerate(batch):
            term.term_vector = vectors[i]

        embedded_count += len(batch)
        logger.info(
            "Embedded %d/%d terms (batch %d-%d).",
            embedded_count,
            total,
            batch_start,
            batch_start + len(batch),
        )

    return terms, vector_dim


def get_vector_dimension(model_name: str | None = None) -> int:
    """Load the embedding model and return its vector dimension."""
    model_id = model_name or config.EMBEDDING_MODEL
    model = load_sentence_transformer(model_id)
    return _get_dimension(model)


def _get_dimension(model: "SentenceTransformer") -> int:
    """Extract vector dimension from a SentenceTransformer model."""
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())
