"""Document registry (README §7, §8).

Builds one record per raw B2 object by listing the flat ``raw-*`` prefixes the crawler
writes. The stable ``document_id`` is derived from the source key; the raw filename's
stem is the crawler's content hash, which links each record back to its ``raw_documents``
row for best-effort enrichment (source_url, crawl_timestamp, original title).
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

from georgia_ev_intelligence.shared import config

from . import b2_client, detector
from .hashing import sha256_bytes
from .manifest import ManifestStore

logger = logging.getLogger(__name__)

# Statuses considered "done" — re-runs preserve their fields (README §7).
TERMINAL_FIELDS = ("markdown_key", "error_key", "last_processed_at", "processing_status")


def document_id_for(source_key: str) -> str:
    """Stable document_id = ``doc_`` + first 16 hex chars of sha256(source_key)."""
    return "doc_" + sha256_bytes(source_key.encode("utf-8"))[:16]


def _raw_content_hash(source_key: str) -> str:
    """The crawler names raw files ``<content_hash>.<ext>`` — return that stem."""
    base = source_key.rsplit("/", 1)[-1]
    stem, _ = os.path.splitext(base)
    return stem


def _new_record(item: dict, source_bucket: str) -> dict:
    source_key = item["Key"]
    name = source_key.rsplit("/", 1)[-1]
    _, ext = os.path.splitext(name)
    ext = ext.lower()
    raw_hash = _raw_content_hash(source_key)
    return {
        "document_id": document_id_for(source_key),
        "source_bucket": source_bucket,
        "source_key": source_key,
        "original_file_name": name,
        "file_extension": ext,
        "mime_type": detector.TYPE_TO_MIME.get(detector.EXTENSION_TO_TYPE.get(ext, ""), None),
        "file_size_bytes": item.get("Size"),
        "raw_sha256": None,                       # filled at processing time (hash of bytes)
        "raw_content_hash": raw_hash,             # crawler's body-text hash (from filename)
        "crawler_doc_id": f"sha256:{raw_hash}",   # link to raw_documents.doc_id
        "source_url": None,
        "crawl_timestamp": None,
        "discovered_at": None,
        "processing_status": "pending",
        "markdown_key": None,
        "error_key": None,
        "extraction_version": config.EXTRACTION_VERSION,
        "last_processed_at": None,
    }


def _enrich_from_db(records: dict[str, dict]) -> None:
    """Best-effort: fill source_url / crawl_timestamp / title from raw_documents.

    Silently skips if the DB is unreachable or has no matching rows.
    """
    crawler_ids = [r["crawler_doc_id"] for r in records.values() if r.get("crawler_doc_id")]
    if not crawler_ids:
        return
    try:
        from georgia_ev_intelligence.offline_pipeline import postgres_store
        conn = postgres_store._get_connection()
    except Exception as exc:
        logger.info("Registry DB enrichment skipped: %s", exc)
        return

    by_crawler_id: dict[str, dict] = {r["crawler_doc_id"]: r for r in records.values()}
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, url, crawled_at, title FROM raw_documents "
                "WHERE doc_id = ANY(%(ids)s::text[])",
                {"ids": crawler_ids},
            )
            for doc_id, url, crawled_at, title in cur.fetchall():
                rec = by_crawler_id.get(doc_id)
                if not rec:
                    continue
                rec["source_url"] = url
                rec["crawl_timestamp"] = crawled_at.isoformat() if crawled_at else None
                if title and not rec.get("original_title"):
                    rec["original_title"] = title
    except Exception as exc:
        logger.info("Registry DB enrichment query failed: %s", exc)
    finally:
        conn.close()


def build_registry(
    prefixes: Iterable[str],
    *,
    store: ManifestStore,
    enrich_db: bool = True,
    source_bucket: str | None = None,
) -> dict[str, dict]:
    """List all raw objects under ``prefixes``, merge into the registry, persist it.

    Existing records keep their terminal fields (status/markdown_key/...) so re-running
    is idempotent. Returns the full {document_id: record} map.
    """
    bucket = source_bucket or config.B2_BUCKET_NAME
    existing = store.load_registry()

    for prefix in prefixes:
        for item in b2_client.list_objects(prefix):
            key = item.get("Key", "")
            if not key or key.endswith("/"):
                continue
            rec = _new_record(item, bucket)
            doc_id = rec["document_id"]
            if doc_id in existing:
                # Preserve prior processing state; refresh size/metadata only.
                prior = existing[doc_id]
                for field in TERMINAL_FIELDS:
                    rec[field] = prior.get(field, rec[field])
                rec["source_url"] = prior.get("source_url") or rec["source_url"]
                rec["crawl_timestamp"] = prior.get("crawl_timestamp") or rec["crawl_timestamp"]
            existing[doc_id] = rec

    if enrich_db:
        # Only enrich records still missing provenance.
        missing = {k: v for k, v in existing.items() if not v.get("source_url")}
        _enrich_from_db(missing)

    store.save_registry(existing)
    return existing
