"""Build a ParentRecord from a raw_documents row fetched from PostgreSQL.

This adapter bridges the web KB pipeline into the existing chunking
infrastructure without touching the Excel-KB path.
"""
from __future__ import annotations

import hashlib
from typing import Any

from georgia_ev_intelligence.offline_pipeline.chunking.parent_chunk import ParentRecord


def build_parent_record_from_raw_doc(doc: dict) -> ParentRecord:
    """Convert a raw_documents DB row dict into a ParentRecord.

    The record_id is deterministic (sha256-based doc_id prefix) so repeated
    indexing of the same document is idempotent via the existing upsert.
    """
    doc_id: str = doc["doc_id"]          # sha256:<hash>
    short_hash = doc_id.replace("sha256:", "")[:12]
    source_type = doc.get("source_type", "web")

    # Synthesise a record_id compatible with the KB_ROW_ format convention
    record_id = f"WEB_{source_type.upper()[:10]}_{short_hash}"

    parent_chunk_text = _build_web_parent_text(doc)

    return ParentRecord(
        record_id=record_id,
        source_row_id=0,               # no row ID for web docs
        source_type=source_type,
        company=doc.get("linked_company_id") or "",
        category="",
        industry_group="",
        updated_location="",
        address="",
        latitude=None,
        longitude=None,
        primary_facility_type="",
        ev_supply_chain_role="",
        primary_oems="",
        supplier_or_affiliation_type="",
        employment=None,
        product_service="",
        ev_battery_relevant="",
        classification_method="web_crawl",
        row_id=0,
        raw_row=doc,
        parent_chunk_text=parent_chunk_text,
    )


def _build_web_parent_text(doc: dict) -> str:
    """Build the parent_chunk_text that the LLM will see at answer-time."""
    lines = [
        f"Source: {doc.get('source_type', 'web')}",
        f"URL: {doc.get('url', '')}",
        f"Title: {doc.get('title', '')}",
        f"Domain: {doc.get('domain', '')}",
        f"Crawled: {doc.get('crawled_at', '')}",
        "",
        doc.get("body_text", ""),
    ]
    return "\n".join(lines).strip()
