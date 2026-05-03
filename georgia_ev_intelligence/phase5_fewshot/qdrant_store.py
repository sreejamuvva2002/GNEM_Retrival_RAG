"""
phase5_fewshot/qdrant_store.py
─────────────────────────────────────────────────────────────────────────────
Qdrant vector store for few-shot examples.

WHY LOCAL QDRANT (not Qdrant Cloud):
  - No additional service to run — uses qdrant-client in local mode (SQLite backend)
  - Zero cost, zero network latency
  - Persistence: data survives restarts (stored in outputs/fewshot_store/)
  - Easy to upgrade to cloud later (just swap QdrantClient constructor)

COLLECTION SCHEMA:
  Each point represents one verified (question → query → answer) triplet:
  {
    "id":           UUID,
    "vector":       [768 floats],   ← nomic-embed-text of the question
    "payload": {
      "question":   str,            ← original question text
      "query_type": "sql" | "cypher" | "direct",
      "sql":        str | None,     ← verified SQL query (if applicable)
      "cypher":     str | None,     ← verified Cypher query (if applicable)
      "answer":     str,            ← human-validated answer
      "category":   str,            ← AGGREGATE / COUNTY / OEM / RISK / GENERAL
      "source":     str,            ← "manual" | "ragas_eval" | "auto"
    }
  }
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from shared.logger import get_logger

logger = get_logger("phase5.qdrant_store")

_COLLECTION_NAME = "gev_fewshot"
_STORE_PATH      = Path("outputs/fewshot_store")
_EMBED_DIM       = 768


def _get_client():
    """
    Return a Qdrant client in local (on-disk) mode.
    Import is deferred so the module can be imported even if qdrant-client
    is not installed (will fail only when client is first accessed).
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        raise ImportError(
            "qdrant-client is not installed. "
            "Run: pip install qdrant-client"
        )
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


def upsert_example(
    question: str,
    vector: list[float],
    query_type: str,           # "sql" | "cypher" | "direct"
    answer: str,
    sql: str | None = None,
    cypher: str | None = None,
    category: str = "GENERAL",
    source: str = "manual",
    point_id: str | None = None,
) -> str:
    """
    Insert or update a single few-shot example.
    Returns the point ID (UUID string).
    """
    from qdrant_client.models import PointStruct

    client = _get_client()
    _ensure_collection(client)

    pid = point_id or str(uuid.uuid4())
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
    Find the top-k most similar few-shot examples.
    Optionally filter by query_type ("sql", "cypher", "direct").

    Returns list of payload dicts with an added "score" field.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = _get_client()
    _ensure_collection(client)

    search_filter = None
    if query_type_filter:
        search_filter = Filter(
            must=[FieldCondition(
                key="query_type",
                match=MatchValue(value=query_type_filter),
            )]
        )

    results = client.search(
        collection_name=_COLLECTION_NAME,
        query_vector=vector,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    )

    hits = []
    for r in results:
        payload = dict(r.payload or {})
        payload["score"] = round(r.score, 4)
        hits.append(payload)

    logger.debug("Few-shot search returned %d hits (top score=%.3f)",
                 len(hits), hits[0]["score"] if hits else 0)
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
