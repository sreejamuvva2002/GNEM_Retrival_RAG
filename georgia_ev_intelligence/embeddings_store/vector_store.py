"""
Phase 2 — Qdrant Vector Store

Uploads embedded chunks to Qdrant Cloud and provides hybrid search.

WHY QDRANT (from Evidence_Based_Decisions.md):
  - Native hybrid search (dense + sparse BM25 vectors simultaneously)
  - Pre-filters metadata DURING search (not after) — critical for our multi-field
    filters: county + tier + OEM + ev_relevant simultaneously
  - pgvector applies filters AFTER vector search → wrong results for our use case
  - Free cloud tier sufficient for our ~100K chunks
  - Reference: https://qdrant.tech/documentation/concepts/filtering/

COLLECTION CONFIG (matches what you set in Qdrant dashboard):
  - Collection: model-specific collection derived from georgia_ev_chunks
  - Dense vector: "dense" (configured dims, Cosine)
  - Sparse vector: "sparse" (BM25 with IDF enabled)
  - Mode: Hybrid Search + Global Search

PARENT-CHILD PATTERN:
  - We upload ONLY child chunks (256 tokens) to Qdrant for vector search
  - Parent text is stored in the payload under "parent_text"
  - When retrieved, agent uses parent_text (800 tokens) for LLM context
  - This gives precise matching + rich context
"""
from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    SparseVector,
)

from chunking.chunker import Chunk
from shared.config import Config
from shared.logger import get_logger

logger = get_logger("embeddings_store.vector_store")

# How many points to upload in one Qdrant batch
UPLOAD_BATCH_SIZE = 100

_qdrant_client: QdrantClient | None = None
_payload_indexes_ensured = False


def get_qdrant_client() -> QdrantClient:
    """Singleton Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        cfg = Config.get()
        _qdrant_client = QdrantClient(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key,
            timeout=60,
        )
        logger.info("Qdrant client initialized: %s", cfg.qdrant_url)
    return _qdrant_client


def get_collection_name() -> str:
    return Config.get().qdrant_collection


def ensure_payload_indexes() -> None:
    """Create the payload indexes needed for GNEM-only filtering."""
    global _payload_indexes_ensured
    if _payload_indexes_ensured:
        return

    client = get_qdrant_client()
    collection_name = get_collection_name()
    index_fields = {
        # Legacy fields
        "source_type": models.PayloadSchemaType.KEYWORD,
        "chunk_type": models.PayloadSchemaType.KEYWORD,
        "chunk_view": models.PayloadSchemaType.KEYWORD,
        "company_row_id": models.PayloadSchemaType.KEYWORD,
        "kb_schema_version": models.PayloadSchemaType.KEYWORD,
        "source_row_hash": models.PayloadSchemaType.KEYWORD,
        "embed_model": models.PayloadSchemaType.KEYWORD,
        "company_name": models.PayloadSchemaType.KEYWORD,
        "tier": models.PayloadSchemaType.KEYWORD,
        "location_county": models.PayloadSchemaType.KEYWORD,
        "location_city": models.PayloadSchemaType.KEYWORD,
        "industry_group": models.PayloadSchemaType.KEYWORD,
        "facility_type": models.PayloadSchemaType.KEYWORD,
        "classification_method": models.PayloadSchemaType.KEYWORD,
        "supplier_affiliation_type": models.PayloadSchemaType.KEYWORD,
        "primary_oems": models.PayloadSchemaType.KEYWORD,
        "document_type": models.PayloadSchemaType.KEYWORD,
        "ev_supply_chain_role": models.PayloadSchemaType.KEYWORD,
        "ev_battery_relevant": models.PayloadSchemaType.KEYWORD,
        "employment": models.PayloadSchemaType.FLOAT,
        # Schema-aligned fields (data dictionary)
        "Record_ID":       models.PayloadSchemaType.KEYWORD,
        "Company":         models.PayloadSchemaType.KEYWORD,
        "Company_Clean":   models.PayloadSchemaType.KEYWORD,
        "County":          models.PayloadSchemaType.KEYWORD,
        "Tier_Level":      models.PayloadSchemaType.KEYWORD,
        "Tier_Confidence": models.PayloadSchemaType.KEYWORD,
        "OEM_GA":          models.PayloadSchemaType.KEYWORD,
        "Industry_Group":  models.PayloadSchemaType.KEYWORD,
        "Industry_Code":   models.PayloadSchemaType.INTEGER,
        "Industry_Name":   models.PayloadSchemaType.KEYWORD,
        "Is_Announcement": models.PayloadSchemaType.KEYWORD,
        "Chunk_ID":        models.PayloadSchemaType.KEYWORD,
        "Employment":      models.PayloadSchemaType.INTEGER,
    }

    for field_name, schema in index_fields.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )
        except Exception as exc:
            logger.warning("Payload index setup skipped for %s: %s", field_name, exc)

    _payload_indexes_ensured = True
    logger.info("Verified payload indexes for Qdrant collection '%s'", collection_name)


# ─────────────────────────────────────────────────────────────────────────────
# COLLECTION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def ensure_collection_exists() -> bool:
    """
    Verify the Qdrant collection exists.
    We do NOT create it here — it was created manually in the Qdrant dashboard
    with Hybrid Search + Global Search + IDF enabled settings.

    Returns True if collection exists, False otherwise.
    """
    client = get_qdrant_client()
    collection_name = get_collection_name()

    try:
        info = client.get_collection(collection_name)
        logger.info(
            "Qdrant collection '%s' verified: %d points",
            collection_name,
            info.points_count or 0,
        )
        return True
    except Exception as exc:
        logger.error(
            "Qdrant collection '%s' NOT found: %s\n"
            "Please create it manually in Qdrant dashboard with:\n"
            "  - Mode: Global Search\n"
            "  - Search: Simple Hybrid Search\n"
            "  - Dense vector: 'dense', %d dims, Cosine\n"
            "  - Sparse vector: 'sparse', IDF enabled",
            collection_name, exc, Config.get().qdrant_dimensions
        )
        return False


def get_collection_stats() -> dict[str, Any]:
    """Return current collection statistics."""
    client = get_qdrant_client()
    collection_name = get_collection_name()
    try:
        info = client.get_collection(collection_name)
        return {
            "points_count": info.points_count or 0,
            "indexed_vectors_count": info.indexed_vectors_count or 0,
            "status": str(info.status),
            "collection": collection_name,
        }
    except Exception as exc:
        return {"error": str(exc), "collection": collection_name}


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE VECTOR GENERATION (BM25-style TF-IDF)
# ─────────────────────────────────────────────────────────────────────────────

def _build_sparse_vector(text: str) -> dict[str, Any]:
    """
    Build a sparse BM25-style vector for a text.

    Since Qdrant handles IDF server-side (we enabled IDF in the dashboard),
    we just need to provide term frequencies as the sparse vector.

    Format: {"indices": [token_ids], "values": [term_frequencies]}

    We use a simple character n-gram hash approach to generate stable token IDs
    without needing a separate vocabulary. This matches what the Qdrant sparse
    index expects when IDF is server-managed.
    """
    import re
    # Tokenize: lowercase, split on non-alphanumeric
    tokens = re.findall(r"[a-z0-9]+", text.lower())

    # Remove very common stopwords (they hurt BM25 precision)
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall",
        "this", "that", "these", "those", "it", "its", "as", "up",
    }
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 2]

    if not tokens:
        # Return minimal sparse vector
        return {"indices": [0], "values": [0.0]}

    # Count term frequencies
    tf: dict[str, int] = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1

    # Convert tokens to integer indices using hash
    # Use modulo 2^20 (~1M vocab) to stay within Qdrant limits
    VOCAB_SIZE = 1_048_576  # 2^20

    indices = []
    values = []
    seen_indices: set[int] = set()

    for token, count in tf.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, "big") % VOCAB_SIZE
        if idx not in seen_indices:
            seen_indices.add(idx)
            indices.append(idx)
            values.append(float(count))

    return {"indices": indices, "values": values}


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

def upload_chunks(
    chunks: list[Chunk],
    vectors: dict[str, list[float]],
    parent_chunks: dict[str, Chunk] | None = None,
) -> int:
    """
    Upload chunks with their embeddings to Qdrant.

    WHAT WE UPLOAD:
      - Child chunks + company chunks → these get embedded and stored as points
      - Parent text is stored IN the payload of child chunks (no separate upload)
      - This implements the parent-child retrieval pattern without two searches

    Args:
        chunks       : List of Chunk objects (child + company, NOT parents)
        vectors      : Dict of chunk_id → 768-dim dense vector
        parent_chunks: Dict of parent_id → parent Chunk (for storing parent text)

    Returns:
        Number of points successfully uploaded
    """
    if not chunks:
        return 0

    client = get_qdrant_client()
    collection_name = get_collection_name()
    parent_chunks = parent_chunks or {}

    points = []
    skipped = 0

    for chunk in chunks:
        if chunk.chunk_id not in vectors:
            logger.warning("No vector for chunk %s — skipping", chunk.chunk_id[:8])
            skipped += 1
            continue

        dense_vector = vectors[chunk.chunk_id]
        sv = _build_sparse_vector(chunk.text)
        # SparseVector model required — plain dict causes upload errors
        sparse_vector = SparseVector(indices=sv["indices"], values=sv["values"])

        # Build payload — everything Qdrant will store and filter on
        payload = {**chunk.metadata}
        payload["text"] = chunk.text
        payload["chunk_id"] = chunk.chunk_id
        payload["token_estimate"] = chunk.token_estimate
        payload["char_count"] = chunk.char_count
        payload.setdefault("embed_model", Config.get().ollama_embed_model)

        # Store parent text inside child payload
        # KEY PATTERN: search with small (256 tokens), return large (800 tokens)
        if chunk.chunk_type == "child" and chunk.parent_id:
            parent = parent_chunks.get(chunk.parent_id)
            if parent:
                payload["parent_text"] = parent.text
                payload["parent_token_estimate"] = parent.token_estimate

        points.append(
            PointStruct(
                id=chunk.chunk_id,
                vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector,
                },
                payload=payload,
            )
        )

    if skipped:
        logger.warning("Skipped %d chunks (no vector)", skipped)

    if not points:
        return 0

    # Upload in batches
    uploaded = 0
    for batch_start in range(0, len(points), UPLOAD_BATCH_SIZE):
        batch = points[batch_start: batch_start + UPLOAD_BATCH_SIZE]
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True,
            )
            uploaded += len(batch)
            logger.debug(
                "Uploaded batch [%d:%d] to Qdrant",
                batch_start, batch_start + len(batch)
            )
        except Exception as exc:
            logger.error(
                "Qdrant upload error for batch [%d:%d]: %s",
                batch_start, batch_start + len(batch), exc
            )

    logger.info("Uploaded %d/%d points to Qdrant '%s'", uploaded, len(points), collection_name)
    return uploaded


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def _build_metadata_filter(filters: dict[str, Any]) -> Filter | None:
    """
    Build a Qdrant Filter from a dict of filter conditions.

    Supported filter keys:
        company_name     : str — exact match
        tier             : str — exact match (e.g. "Tier 1", "OEM")
        location_county  : str — exact match
        location_city    : str — exact match
        ev_battery_relevant : str — "Yes", "No", "Indirect"
        primary_oems     : str — substring match (not supported directly, use must_not)
        min_employment   : float — employment >= value
        max_employment   : float — employment <= value
        source_type      : str — "gnem_excel" | "web_document"
        chunk_type       : str — "company" | "child"
    """
    if not filters:
        return None

    must_conditions = []

    str_fields = [
        # Legacy fields
        "company_name", "tier", "location_county", "location_city",
        "ev_battery_relevant", "source_type", "chunk_type",
        "document_type", "ev_supply_chain_role", "facility_type",
        "industry_group", "chunk_view", "company_row_id",
        "kb_schema_version", "source_row_hash", "embed_model",
        "classification_method", "supplier_affiliation_type", "primary_oems",
        # Schema-aligned fields
        "Record_ID", "Company", "Company_Clean", "County",
        "Tier_Level", "Tier_Confidence", "Industry_Group", "Industry_Name",
        "Chunk_ID",
    ]
    for field in str_fields:
        if field in filters:
            must_conditions.append(
                FieldCondition(key=field, match=MatchValue(value=filters[field]))
            )

    # Boolean fields (OEM_GA, Is_Announcement) — stored as Python bool, match directly
    for bool_field in ("OEM_GA", "Is_Announcement"):
        if bool_field in filters:
            must_conditions.append(
                FieldCondition(key=bool_field, match=MatchValue(value=filters[bool_field]))
            )

    # Industry_Code — integer exact match
    if "Industry_Code" in filters:
        must_conditions.append(
            FieldCondition(key="Industry_Code", match=MatchValue(value=int(filters["Industry_Code"])))
        )

    # Employment range — support both legacy and schema-aligned key prefixes
    min_emp = filters.get("min_employment") or filters.get("min_Employment")
    max_emp = filters.get("max_employment") or filters.get("max_Employment")
    if min_emp is not None or max_emp is not None:
        must_conditions.append(
            FieldCondition(
                key="employment",
                range=Range(
                    gte=float(min_emp) if min_emp is not None else None,
                    lte=float(max_emp) if max_emp is not None else None,
                ),
            )
        )

    if not must_conditions:
        return None

    return Filter(must=must_conditions)


def search_hybrid(
    query_text: str,
    query_vector: list[float],
    top_k: int = 6,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Hybrid search: dense + sparse simultaneously.

    Returns top_k results with their payloads.
    Results include 'parent_text' if available (for LLM context).

    Args:
        query_text   : Raw query string (used for sparse vector)
        query_vector : Dense embedding of query (768-dim)
        top_k        : Number of results to return
        filters      : Optional metadata filters

    Returns:
        List of result dicts with keys: score, text, parent_text, metadata
    """
    client = get_qdrant_client()
    collection_name = get_collection_name()
    cfg = Config.get()
    ensure_payload_indexes()

    qdrant_filter = _build_metadata_filter(filters or {})
    sv = _build_sparse_vector(query_text)

    try:
        # Build typed Prefetch objects — plain dicts cause 'Unsupported query type' error
        prefetch_dense = models.Prefetch(
            query=query_vector,
            using=cfg.qdrant_dense_name,
            limit=top_k * 3,
            filter=qdrant_filter,
        )
        prefetch_sparse = models.Prefetch(
            query=models.SparseVector(indices=sv["indices"], values=sv["values"]),
            using=cfg.qdrant_sparse_name,
            limit=top_k * 3,
            filter=qdrant_filter,
        )

        results = client.query_points(
            collection_name=collection_name,
            prefetch=[prefetch_dense, prefetch_sparse],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        search_results = []
        for point in results.points:
            payload = point.payload or {}
            # "parent_text" is the full 800-token context returned to LLM
            # "text" is the 256-token child used for matching
            search_results.append({
                "score": point.score,
                "chunk_id": payload.get("chunk_id", str(point.id)),
                "text": payload.get("text", ""),
                "parent_text": payload.get("parent_text", payload.get("text", "")),
                "company_name": payload.get("company_name", ""),
                "source_url": payload.get("source_url", ""),
                "document_type": payload.get("document_type", ""),
                "chunk_type": payload.get("chunk_type", ""),
                "tier": payload.get("tier", ""),
                "location_county": payload.get("location_county", ""),
                "ev_battery_relevant": payload.get("ev_battery_relevant", ""),
                "metadata": payload,
            })

        logger.info(
            "Hybrid search: '%s...' → %d results (filter=%s)",
            query_text[:50], len(search_results), bool(filters)
        )
        return search_results

    except Exception as exc:
        logger.error("Qdrant hybrid search failed: %s", exc)
        # Fallback to dense-only search
        try:
            logger.info("Falling back to dense-only search...")
            results = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=cfg.qdrant_dense_name,
                limit=top_k,
                with_payload=True,
                **({"query_filter": qdrant_filter} if qdrant_filter else {}),
            )
            search_results = []
            for point in results.points:
                payload = point.payload or {}
                search_results.append({
                    "score": point.score,
                    "chunk_id": payload.get("chunk_id", str(point.id)),
                    "text": payload.get("text", ""),
                    "parent_text": payload.get("parent_text", payload.get("text", "")),
                    "company_name": payload.get("company_name", ""),
                    "source_url": payload.get("source_url", ""),
                    "chunk_type": payload.get("chunk_type", ""),
                    "metadata": payload,
                })
            return search_results
        except Exception as exc2:
            logger.error("Dense fallback also failed: %s", exc2)
            return []


def search_dense(
    query_vector: list[float],
    top_k: int = 12,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Dense-only search over the Qdrant collection."""
    client = get_qdrant_client()
    collection_name = get_collection_name()
    cfg = Config.get()
    ensure_payload_indexes()
    qdrant_filter = _build_metadata_filter(filters or {})

    try:
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=cfg.qdrant_dense_name,
            limit=top_k,
            with_payload=True,
            **({"query_filter": qdrant_filter} if qdrant_filter else {}),
        )
    except Exception as exc:
        logger.error("Qdrant dense search failed: %s", exc)
        return []

    search_results = []
    for point in results.points:
        payload = point.payload or {}
        search_results.append({
            "score": point.score,
            "chunk_id": payload.get("chunk_id", str(point.id)),
            "text": payload.get("text", ""),
            "parent_text": payload.get("parent_text", payload.get("text", "")),
            "company_name": payload.get("company_name", ""),
            "source_url": payload.get("source_url", ""),
            "document_type": payload.get("document_type", ""),
            "chunk_type": payload.get("chunk_type", ""),
            "tier": payload.get("tier", ""),
            "location_county": payload.get("location_county", ""),
            "ev_battery_relevant": payload.get("ev_battery_relevant", ""),
            "metadata": payload,
        })

    logger.info("Dense search returned %d results", len(search_results))
    return search_results


def scroll_points(
    filters: dict[str, Any] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Scroll payload-bearing records from Qdrant without vector ranking.

    Useful when we want exhaustive retrieval from the vector DB itself,
    then do filtering/aggregation in Python.
    """
    client = get_qdrant_client()
    collection_name = get_collection_name()
    ensure_payload_indexes()
    qdrant_filter = _build_metadata_filter(filters or {})

    records: list[dict[str, Any]] = []
    next_offset = None
    remaining = limit

    while True:
        batch_limit = 128
        if remaining is not None:
            batch_limit = max(1, min(batch_limit, remaining))

        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_filter,
            limit=batch_limit,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            records.append({
                "id": str(point.id),
                "payload": point.payload or {},
            })

        if remaining is not None:
            remaining -= len(points)
            if remaining <= 0:
                break
        if not next_offset or not points:
            break

    logger.info(
        "Scrolled %d points from Qdrant (filter=%s)",
        len(records), bool(filters)
    )
    return records


def delete_company_chunks(company_name: str) -> int:
    """Delete all chunks for a specific company (for re-embedding)."""
    return delete_company_index_chunks(company_name=company_name)


def delete_company_index_chunks(company_name: str | None = None) -> int:
    """
    Delete GNEM company chunks only.

    This intentionally leaves web-document chunks untouched, even if they share
    a company name with the structured KB row.
    """
    client = get_qdrant_client()
    collection_name = get_collection_name()
    must = [
        FieldCondition(key="source_type", match=MatchValue(value="gnem_excel")),
        FieldCondition(key="chunk_type", match=MatchValue(value="company")),
    ]
    if company_name:
        must.append(FieldCondition(key="company_name", match=MatchValue(value=company_name)))

    try:
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(must=must),
            wait=True,
        )
        logger.info("Deleted GNEM company chunks from Qdrant (company=%s)", company_name or "all")
        return 1
    except Exception as exc:
        logger.error("Failed to delete GNEM company chunks (company=%s): %s", company_name, exc)
        return 0


def prune_stale_company_index_chunks(
    keep_point_ids: set[str],
    company_name: str | None = None,
) -> int:
    """
    Delete GNEM company chunks not present in keep_point_ids.

    Rebuild flow uses this after uploading fresh deterministic IDs, so old
    random-ID chunks from earlier indexing runs do not linger in retrieval.
    """
    existing = scroll_points(
        filters={
            "source_type": "gnem_excel",
            "chunk_type": "company",
            **({"company_name": company_name} if company_name else {}),
        },
        limit=None,
    )
    stale_ids = [record["id"] for record in existing if record["id"] not in keep_point_ids]
    if not stale_ids:
        logger.info("No stale GNEM company chunks to prune (company=%s)", company_name or "all")
        return 0

    client = get_qdrant_client()
    collection_name = get_collection_name()
    deleted = 0
    for batch_start in range(0, len(stale_ids), UPLOAD_BATCH_SIZE):
        batch = stale_ids[batch_start: batch_start + UPLOAD_BATCH_SIZE]
        try:
            client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=batch),
                wait=True,
            )
            deleted += len(batch)
        except Exception as exc:
            logger.error(
                "Failed to prune stale GNEM company chunks [%d:%d]: %s",
                batch_start,
                batch_start + len(batch),
                exc,
            )

    logger.info("Pruned %d stale GNEM company chunks (company=%s)", deleted, company_name or "all")
    return deleted


def verify_qdrant_connection() -> dict[str, Any]:
    """Verify Qdrant is reachable and collection exists."""
    try:
        stats = get_collection_stats()
        if "error" in stats:
            return {"ok": False, "error": stats["error"]}
        ensure_payload_indexes()
        return {
            "ok": True,
            "collection": stats["collection"],
            "points": stats["points_count"],
            "status": stats["status"],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
