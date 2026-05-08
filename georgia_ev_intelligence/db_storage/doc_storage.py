"""
Phase 1 — Document Storage
Saves extracted documents to:
  - Backblaze B2 (raw content / PDF bytes)
  - PostgreSQL gev_documents (metadata + status tracking)
Deduplicates using SHA-256 content hash.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from shared.db import Document, get_session
from shared.logger import get_logger
from shared.storage import key_exists, make_document_key, upload_bytes
from web_extraction.extractor import ExtractedDocument

logger = get_logger("db_storage.doc_storage")


def save_document(
    extracted: ExtractedDocument,
    company_id: int | None,
    search_query: str = "",
    relevance_score: float = 0.0,
) -> int | None:
    """
    Save an extracted document to B2 + PostgreSQL.

    Deduplication:
        - If content_hash already exists in gev_documents → skip B2 upload, return existing ID
        - If same URL already indexed → update metadata, return existing ID

    Returns:
        document_id (int) if saved/exists, None if save failed
    """
    if not extracted.text or extracted.error:
        logger.debug("Skipping empty/failed extraction for %s", extracted.url)
        return None

    session = get_session()
    try:
        # Check for duplicate content (same hash = same document from different URL)
        if extracted.content_hash:
            existing_by_hash = (
                session.query(Document)
                .filter_by(content_hash_sha256=extracted.content_hash)
                .first()
            )
            if existing_by_hash:
                logger.debug(
                    "Duplicate content for %s (hash matches doc %d) — skipping",
                    extracted.url, existing_by_hash.id
                )
                return existing_by_hash.id

        # Check for duplicate URL
        existing_by_url = (
            session.query(Document)
            .filter_by(source_url=extracted.url)
            .first()
        )

        # Upload to B2
        b2_key = None
        if extracted.content_hash:
            ext = "pdf" if extracted.content_type == "pdf" else "html"
            b2_key = make_document_key(
                extracted.company_name,
                extracted.content_hash,
                ext,
            )
            if not key_exists(b2_key):
                content_to_upload = (
                    extracted.raw_bytes if extracted.content_type == "pdf" and extracted.raw_bytes
                    else extracted.text.encode("utf-8")
                )
                mime = (
                    "application/pdf" if extracted.content_type == "pdf"
                    else "text/plain; charset=utf-8"
                )
                upload_bytes(content_to_upload, b2_key, mime)
                logger.info(
                    "B2 upload OK [%s] %.1fKB -> %s",
                    extracted.content_type.upper(), len(content_to_upload) / 1024, b2_key
                )
            else:
                logger.info("B2 key already exists (dedup skipped): %s", b2_key)

        if existing_by_url:
            # Update existing record
            existing_by_url.b2_key = b2_key or existing_by_url.b2_key
            existing_by_url.content_hash_sha256 = extracted.content_hash
            existing_by_url.word_count = extracted.word_count
            existing_by_url.char_count = extracted.char_count
            existing_by_url.extraction_status = "extracted"
            existing_by_url.extracted_at = datetime.utcnow()
            session.commit()
            return existing_by_url.id

        # New document
        doc = Document(
            company_id=company_id,
            company_name=extracted.company_name,
            source_url=extracted.url,
            b2_key=b2_key,
            content_type=extracted.content_type,
            content_hash_sha256=extracted.content_hash,
            file_size_bytes=extracted.raw_bytes_size or len(extracted.text.encode("utf-8")),
            document_type=_infer_document_type(extracted.url, extracted.title if hasattr(extracted, "title") else ""),
            relevance_score=relevance_score,
            extraction_status="extracted",
            word_count=extracted.word_count,
            search_query=search_query[:500] if search_query else None,
            extracted_at=datetime.utcnow(),
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)
        logger.info(
            "DB saved document #%d [%s] %d words -> %s",
            doc.id, extracted.content_type, extracted.word_count,
            extracted.url[:80]
        )
        return doc.id

    except Exception as exc:
        session.rollback()
        logger.error("Failed to save document for %s: %s", extracted.url, exc)
        return None
    finally:
        session.close()


def _infer_document_type(url: str, title: str) -> str:
    """Infer document type from URL and title."""
    url_lower = url.lower()
    title_lower = title.lower()
    combined = url_lower + " " + title_lower
    if "sec.gov" in url_lower or "10-k" in combined or "10-q" in combined:
        return "SEC Filing"
    if "press release" in combined or "pressrelease" in url_lower or "pr-" in url_lower:
        return "Press Release"
    if "annual report" in combined or "annual-report" in url_lower:
        return "Annual Report"
    if "energy.gov" in url_lower or "doe.gov" in url_lower:
        return "Government Report"
    if "georgia.org" in url_lower or "selectgeorgia" in url_lower or "gdced" in url_lower:
        return "Economic Development"
    if url_lower.endswith(".pdf"):
        return "PDF Document"
    if any(w in combined for w in ["news", "article", "reuters", "bloomberg", "autonews"]):
        return "News Article"
    return "Web Page"


def mark_document_failed(url: str, error: str, company_id: int | None = None, company_name: str = "") -> None:
    """Record a failed extraction attempt in PostgreSQL."""
    session = get_session()
    try:
        existing = session.query(Document).filter_by(source_url=url).first()
        if existing:
            existing.extraction_status = "failed"
            existing.extraction_error = error[:500]
        else:
            doc = Document(
                company_id=company_id,
                company_name=company_name,
                source_url=url,
                extraction_status="failed",
                extraction_error=error[:500],
            )
            session.add(doc)
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("Could not record failed document %s: %s", url, exc)
    finally:
        session.close()


def get_document_count_for_company(company_name: str) -> int:
    """How many successfully extracted documents does this company have?"""
    session = get_session()
    try:
        return (
            session.query(Document)
            .filter_by(company_name=company_name, extraction_status="extracted")
            .count()
        )
    finally:
        session.close()
