"""
Phase 2 — Embedding Pipeline Orchestrator

End-to-end: PostgreSQL documents → chunk → embed → Qdrant

TWO EMBEDDING PASSES:
  Pass 1: GNEM Companies (205 structured records from gev_companies)
    - One chunk per company (structured "Company: X | Tier: Y ..." text)
    - Fast: ~1 min for all 205 companies

  Pass 2: Web Documents (from phase1 extraction, stored in gev_documents)
    - Hierarchical parent-child chunks for each document
    - Only processes documents with extraction_status = 'extracted'
    - Tracks which documents are already embedded (gev_document_chunks table)

Usage:
  # Embed GNEM companies only (fast, good starting point):
  venv\\Scripts\\python -m phase2_embedding.pipeline --companies-only

  # Embed everything:
  venv\\Scripts\\python -m phase2_embedding.pipeline

  # Embed for a specific company:
  venv\\Scripts\\python -m phase2_embedding.pipeline --company "Hanwha Q Cells"

  # Re-embed (overwrite existing):
  venv\\Scripts\\python -m phase2_embedding.pipeline --reembed
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from phase2_embedding.chunker import (
    Chunk,
    chunk_company_record,
    chunk_document,
    get_child_chunks,
)
from phase2_embedding.embedder import embed_chunks, verify_ollama_embed
from phase2_embedding.vector_store import (
    ensure_collection_exists,
    get_collection_stats,
    upload_chunks,
    verify_qdrant_connection,
)
from phase1_extraction.kb_loader import (
    build_document_text,
    get_all_companies_from_db,
    load_companies_from_excel,
    sync_companies_to_db,
)
from shared.db import (
    Company, Document, DocumentChunk,
    create_tables, get_session,
)
from shared.logger import get_logger
from sqlalchemy import text

logger = get_logger("phase2.pipeline")

SEPARATOR = "=" * 60


# ─────────────────────────────────────────────────────────────────────────────
# PASS 1: Embed GNEM Company Records
# ─────────────────────────────────────────────────────────────────────────────

def embed_companies(company_filter: str | None = None, reembed: bool = False) -> dict[str, int]:
    """
    Embed all 205 GNEM company records into Qdrant.

    Each company → 1 "company" chunk with all metadata fields as payload.
    This gives the agent immediate access to all 205 companies without
    needing to wait for web documents.

    Args:
        company_filter : Optional partial company name to filter
        reembed        : If True, re-embed even if already in Qdrant

    Returns:
        Dict with embedded/skipped/failed counts
    """
    logger.info("=== Pass 1: Embedding GNEM Company Records ===")

    companies = get_all_companies_from_db()
    if company_filter:
        companies = [c for c in companies if company_filter.lower() in c["company_name"].lower()]

    if not companies:
        logger.warning("No companies found in DB. Run Phase 1 first.")
        return {"embedded": 0, "skipped": 0, "failed": 0}

    logger.info("Processing %d companies...", len(companies))

    all_chunks: list[Chunk] = []
    parent_map: dict[str, Chunk] = {}  # parent_id → parent chunk

    for company in companies:
        doc_text = build_document_text(company)
        company_chunks = chunk_company_record(company, doc_text)
        all_chunks.extend(company_chunks)

    # Embed all company chunks (no children here — company chunks are complete)
    logger.info("Embedding %d company chunks...", len(all_chunks))
    vectors = embed_chunks(all_chunks)

    # Upload to Qdrant
    uploaded = upload_chunks(all_chunks, vectors, parent_chunks=parent_map)

    logger.info("✅ Pass 1 done: %d company chunks uploaded to Qdrant", uploaded)
    return {"embedded": uploaded, "skipped": 0, "failed": len(all_chunks) - uploaded}


# ─────────────────────────────────────────────────────────────────────────────
# PASS 2: Embed Web Documents
# ─────────────────────────────────────────────────────────────────────────────

def _is_already_embedded(document_id: int, session) -> bool:
    """Check if this document already has chunks tracked in PostgreSQL."""
    count = session.execute(
        text("SELECT COUNT(*) FROM gev_document_chunks WHERE document_id = :did"),
        {"did": document_id}
    ).scalar()
    return (count or 0) > 0


def _save_chunk_tracking(
    document_id: int,
    chunks: list[Chunk],
    session,
) -> None:
    """
    Track uploaded chunks in gev_document_chunks table.
    This lets us know which documents are embedded and skip them on re-runs.
    """
    for chunk in chunks:
        if chunk.chunk_type == "parent":
            continue  # Only track child + company chunks (what's in Qdrant)

        from shared.db import DocumentChunk
        dc = DocumentChunk(
            document_id=document_id,
            qdrant_id=chunk.chunk_id,
            chunk_type=chunk.chunk_type,
            parent_qdrant_id=chunk.parent_id,
            chunk_index=chunk.metadata.get("chunk_index"),
            token_count=chunk.token_estimate,
            char_count=chunk.char_count,
            text_preview=chunk.text[:500],
        )
        session.add(dc)
    try:
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.warning("Failed to save chunk tracking: %s", exc)


def embed_documents(
    company_filter: str | None = None,
    reembed: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    """
    Embed all extracted web documents into Qdrant.

    Reads from gev_documents (extraction_status='extracted'),
    chunks each document, embeds, uploads to Qdrant,
    tracks in gev_document_chunks.

    Args:
        company_filter : Only process documents for this company (partial match)
        reembed        : Re-embed even if already tracked
        limit          : Process at most N documents (for testing)

    Returns:
        Dict with embedded/skipped/failed/total counts
    """
    logger.info("=== Pass 2: Embedding Web Documents ===")

    session = get_session()
    stats = {"embedded": 0, "skipped": 0, "failed": 0, "total": 0}

    try:
        # Build query
        query = """
            SELECT
                d.id, d.company_id, d.company_name,
                d.source_url, d.document_type,
                d.word_count, d.b2_key
            FROM gev_documents d
            WHERE d.extraction_status = 'extracted'
              AND d.word_count >= 50
        """
        params = {}

        if company_filter:
            query += " AND LOWER(d.company_name) LIKE :name_filter"
            params["name_filter"] = f"%{company_filter.lower()}%"

        query += " ORDER BY d.id"

        if limit:
            query += f" LIMIT {limit}"

        rows = session.execute(text(query), params).fetchall()
        stats["total"] = len(rows)
        logger.info("Found %d extracted documents to process", len(rows))

        if not rows:
            logger.warning("No extracted documents found. Run Phase 1 first.")
            return stats

        # Load company metadata for enriching chunk payloads
        companies_by_name = {
            c["company_name"]: c for c in get_all_companies_from_db()
        }

        for row in rows:
            doc_id = row[0]
            company_id = row[1]
            company_name = row[2]
            source_url = row[3]
            document_type = row[4] or "web"
            word_count = row[5] or 0
            b2_key = row[6]

            # Skip if already embedded (unless --reembed)
            if not reembed and _is_already_embedded(doc_id, session):
                logger.debug("Skipping doc #%d for %s (already embedded)", doc_id, company_name)
                stats["skipped"] += 1
                continue

            # Fetch document text from Qdrant B2 or fallback
            # For now: we need to re-extract text from the document
            # Strategy: get text from the extraction already stored
            # We pull the word_count to confirm, then re-extract from B2
            doc_text = _fetch_document_text(doc_id, b2_key, session)

            if not doc_text:
                logger.warning("No text for doc #%d (%s) — skipping", doc_id, company_name)
                stats["failed"] += 1
                continue

            # Get company metadata for enriching payloads
            company_meta = companies_by_name.get(company_name, {})

            # Chunk the document
            chunks = chunk_document(
                text=doc_text,
                company_name=company_name,
                document_id=doc_id,
                source_url=source_url,
                document_type=document_type,
                company_metadata=company_meta,
            )

            if not chunks:
                logger.warning("No chunks for doc #%d — skipping", doc_id)
                stats["failed"] += 1
                continue

            # Get only child chunks for embedding (parents stored in payload)
            embeddable = get_child_chunks(chunks)
            parent_map = {c.chunk_id: c for c in chunks if c.chunk_type == "parent"}

            # Embed
            vectors = embed_chunks(embeddable)

            # Upload to Qdrant
            uploaded = upload_chunks(embeddable, vectors, parent_chunks=parent_map)

            if uploaded > 0:
                # Track in PostgreSQL
                _save_chunk_tracking(doc_id, embeddable, session)
                stats["embedded"] += 1
                logger.info(
                    "✅ Doc #%d [%s] → %d chunks uploaded",
                    doc_id, company_name[:30], uploaded
                )
            else:
                stats["failed"] += 1
                logger.warning("❌ Doc #%d upload failed", doc_id)

    finally:
        session.close()

    logger.info(
        "=== Pass 2 done: %d embedded, %d skipped, %d failed / %d total ===",
        stats["embedded"], stats["skipped"], stats["failed"], stats["total"]
    )
    return stats


def _fetch_document_text(doc_id: int, b2_key: str | None, session) -> str | None:
    """
    Fetch document text for embedding.

    Strategy: re-download from B2 (PDF → PyMuPDF, HTML → decode text).
    Falls back to None if B2 key missing or download fails.
    """
    if not b2_key:
        return None

    try:
        import shared.storage as storage_mod

        raw_bytes = storage_mod.download_bytes(b2_key)

        if b2_key.endswith(".pdf"):
            # Re-extract text from PDF
            import fitz  # PyMuPDF
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            pages = [page.get_text() for page in doc]
            return "\n\n".join(pages).strip() or None
        else:
            # HTML/text — decode directly
            try:
                return raw_bytes.decode("utf-8", errors="replace").strip() or None
            except Exception:
                return None

    except FileNotFoundError:
        logger.warning("B2 key not found: %s", b2_key)
        return None
    except Exception as exc:
        logger.error("Failed to fetch doc #%d from B2: %s", doc_id, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Georgia EV Intelligence — Phase 2 Embedding Pipeline"
    )
    parser.add_argument(
        "--companies-only",
        action="store_true",
        help="Only embed GNEM company records (fast, ~2 min)",
    )
    parser.add_argument(
        "--documents-only",
        action="store_true",
        help="Only embed web documents (skip companies)",
    )
    parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Process only matching company (partial name match)",
    )
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="Re-embed even if already in Qdrant",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N documents (for testing)",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("GEORGIA EV INTELLIGENCE — PHASE 2 EMBEDDING PIPELINE")
    print(f"{'=' * 60}\n")

    # Pre-checks
    logger.info("Verifying connections...")

    embed_check = verify_ollama_embed()
    if not embed_check["ok"]:
        logger.error("Ollama embed FAILED: %s", embed_check["error"])
        logger.error("Make sure Ollama is running: ollama serve")
        sys.exit(1)
    logger.info("✅ Ollama: %s (%d dims)", embed_check["model"], embed_check["dimensions"])

    qdrant_check = verify_qdrant_connection()
    if not qdrant_check["ok"]:
        logger.error("Qdrant FAILED: %s", qdrant_check["error"])
        sys.exit(1)
    logger.info(
        "✅ Qdrant: collection '%s' has %d points",
        qdrant_check["collection"],
        qdrant_check.get("points", 0)
    )

    create_tables()

    # Run embedding passes
    start = time.monotonic()
    total_embedded = 0

    if not args.documents_only:
        result = embed_companies(company_filter=args.company, reembed=args.reembed)
        total_embedded += result["embedded"]

    if not args.companies_only:
        result = embed_documents(
            company_filter=args.company,
            reembed=args.reembed,
            limit=args.limit,
        )
        total_embedded += result["embedded"]

    elapsed = time.monotonic() - start

    # Final stats
    stats = get_collection_stats()
    print(f"\n{SEPARATOR}")
    print("PHASE 2 COMPLETE")
    print(SEPARATOR)
    print(f"  Total embedded   : {total_embedded}")
    print(f"  Qdrant points    : {stats.get('points_count', 'N/A')}")
    print(f"  Time elapsed     : {elapsed:.1f}s")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
