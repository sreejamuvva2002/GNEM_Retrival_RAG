"""Writer: append RawDocument to JSONL, upsert into PostgreSQL, and upload to Backblaze B2."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from georgia_ev_intelligence.kb_builder.models import RawDocument

logger = logging.getLogger(__name__)


def _jsonl_path(raw_docs_dir: Path, source_type: str) -> Path:
    """Map source_type to a dedicated JSONL shard file."""
    mapping = {
        "company_site": "company_sites.jsonl",
        "news":         "news.jsonl",
        "gov_doc":      "gov_docs.jsonl",
        "ddg_search":   "ddg_search.jsonl",
    }
    filename = mapping.get(source_type, "other.jsonl")
    return raw_docs_dir / filename


def _b2_available() -> bool:
    """Return True if B2 credentials are configured."""
    from georgia_ev_intelligence.shared import config
    return bool(config.B2_KEY_ID and config.B2_APPLICATION_KEY and config.B2_BUCKET_NAME)


def write_document(
    doc: RawDocument,
    raw_docs_dir: Path,
    *,
    db: bool = True,
    b2: bool = True,
) -> None:
    """Persist a RawDocument to disk (JSONL), optionally to PostgreSQL and Backblaze B2.

    Parameters
    ----------
    doc:
        The document to persist.
    raw_docs_dir:
        Path to the ``kb/raw_docs/`` directory.
    db:
        If True (default), also upsert into the PostgreSQL raw_documents table.
    b2:
        If True (default), upload raw bytes + JSONL shard to Backblaze B2.
        Silently skipped when B2 credentials are not configured.
    """
    raw_docs_dir.mkdir(parents=True, exist_ok=True)
    path = _jsonl_path(raw_docs_dir, doc.source_type)

    # 1. Append to JSONL — crash-safe write-ahead log
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

    logger.debug("JSONL  ← %s  (%s chars)", doc.url, len(doc.body_text))

    # 2. PostgreSQL upsert
    if db:
        try:
            from georgia_ev_intelligence.offline_pipeline.postgres_store import (
                upsert_raw_document,
            )
            upsert_raw_document(doc.to_dict())
            logger.debug("DB     ← %s", doc.doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DB upsert failed for %s: %s", doc.url, exc)

    # 3. Backblaze B2 upload
    if b2 and _b2_available():
        _upload_to_b2(doc)


def _upload_to_b2(doc: RawDocument) -> None:
    """Upload raw bytes for a single document to B2. Logs warnings on failure."""
    from georgia_ev_intelligence.shared import config
    from georgia_ev_intelligence.kb_builder.b2_uploader import upload_raw_bytes

    raw = doc.raw_binary
    if not raw:
        logger.debug("B2: no raw_binary for %s — skipping byte upload", doc.doc_id)
        return

    try:
        key = upload_raw_bytes(
            doc_id=doc.doc_id,
            file_type=doc.file_type,
            raw_bytes=raw,
            bucket_name=config.B2_BUCKET_NAME,
            metadata={
                "url":               doc.url,
                "source_type":       doc.source_type,
                "crawled_at":        doc.crawled_at.isoformat(),
                "linked_company_id": doc.linked_company_id or "",
            },
        )
        logger.info("B2 ← %s  (%d bytes)", key, len(raw))
    except Exception as exc:  # noqa: BLE001
        logger.warning("B2 upload failed for %s: %s", doc.url, exc)
