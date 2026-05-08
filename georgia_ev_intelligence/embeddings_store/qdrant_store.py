"""
phase5_fewshot/qdrant_store.py
─────────────────────────────────────────────────────────────────────────────
Qdrant vector store for few-shot examples.
Uses local on-disk mode (no extra service needed).

API NOTE: Uses query_points() (qdrant-client >= 1.9.0 stable API).
          The older client.search() was deprecated in 1.9+ and removed in 2.x.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from shared.logger import get_logger

logger = get_logger("embeddings_store.qdrant_store")

_COLLECTION_NAME = "gev_fewshot"
_STORE_PATH      = Path("outputs/fewshot_store")
_EMBED_DIM       = 768


def _get_client():
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")
    _STORE_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(_STORE_PATH))


def _ensure_collection(client) -> None:
    """Create the collection if it doesn't exist."""
    from qdrant_client.models import Distance, VectorParams
    existing = [c.name for c in client.get_collections().collections]
    if _COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=_COLLECTION_NAME,
            vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection: %s", _COLLECTION_NAME)


def _make_uuid(point_id: str | None) -> str:
    """
    Return a valid UUID string for Qdrant.
    Qdrant local mode requires proper UUID format (not arbitrary strings).
    If point_id is already a valid UUID string, return it as-is.
    Otherwise generate a new UUID4.
    """
    if point_id:
        try:
            return str(uuid.UUID(point_id))
        except ValueError:
            pass
    return str(uuid.uuid4())


def upsert_example(
    question: str,
    vector: list[float],
    query_type: str,
    answer: str,
    sql: str | None = None,
    cypher: str | None = None,
    category: str = "GENERAL",
    source: str = "manual",
    point_id: str | None = None,
) -> str:
    """Insert or update a few-shot example. Returns the point ID string."""
    from qdrant_client.models import PointStruct

    client = _get_client()
    _ensure_collection(client)

    pid = _make_uuid(point_id)
    client.upsert(
        collection_name=_COLLECTION_NAME,
        points=[
            PointStruct(
                id=pid,
                vector=vector,
                payload={
                    "question":   question,
                    "query_type": query_type,
                    "sql":        sql,
                    "cypher":     cypher,
                    "answer":     answer,
                    "category":   category,
                    "source":     source,
                },
            )
        ],
    )
    logger.debug("Upserted example %s (%s)", pid, query_type)
    return pid


def search_similar(
    vector: list[float],
    top_k: int = 3,
    query_type_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Find top-k most similar examples using cosine similarity.
    Uses query_points() API (stable in qdrant-client >= 1.9.0).
    Falls back to legacy search() for older clients.

    Returns list of payload dicts with added 'score' field.
    """
    client = _get_client()
    _ensure_collection(client)

    # Build filter if needed
    search_filter = None
    if query_type_filter:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        search_filter = Filter(
            must=[FieldCondition(
                key="query_type",
                match=MatchValue(value=query_type_filter),
            )]
        )

    # Try query_points (>= 1.9.0) first, fall back to search (< 1.9.0)
    raw_results = []
    try:
        result = client.query_points(
            collection_name=_COLLECTION_NAME,
            query=vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        )
        raw_results = result.points
    except AttributeError:
        # Older qdrant-client — use legacy search API
        raw_results = client.search(
            collection_name=_COLLECTION_NAME,
            query_vector=vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        )

    hits = []
    for r in raw_results:
        payload = dict(r.payload or {})
        payload["score"] = round(float(r.score), 4)
        hits.append(payload)

    if hits:
        logger.debug("Few-shot search: %d hits, top=%.3f", len(hits), hits[0]["score"])
    return hits


def count_examples() -> int:
    """Return total number of stored few-shot examples."""
    client = _get_client()
    _ensure_collection(client)
    return client.count(collection_name=_COLLECTION_NAME).count


def delete_collection() -> None:
    """Wipe the entire collection (for re-seeding)."""
    client = _get_client()
    client.delete_collection(_COLLECTION_NAME)
    logger.info("Deleted collection %s", _COLLECTION_NAME)
