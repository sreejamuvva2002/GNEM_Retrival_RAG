"""Writer: append RawDocument to JSONL and upsert into PostgreSQL raw_documents."""
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
    }
    filename = mapping.get(source_type, "other.jsonl")
    return raw_docs_dir / filename


def write_document(doc: RawDocument, raw_docs_dir: Path, *, db: bool = True) -> None:
    """Persist a RawDocument to disk (JSONL) and optionally to PostgreSQL.

    Parameters
    ----------
    doc:
        The document to persist.
    raw_docs_dir:
        Path to the ``kb/raw_docs/`` directory.
    db:
        If True (default), also upsert into the PostgreSQL raw_documents table.
        Set False in dry-run / test mode.
    """
    raw_docs_dir.mkdir(parents=True, exist_ok=True)
    path = _jsonl_path(raw_docs_dir, doc.source_type)

    # Append to JSONL — atomic enough for sequential writes
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

    logger.debug("JSONL  ← %s  (%s chars)", doc.url, len(doc.body_text))

    if db:
        try:
            from georgia_ev_intelligence.offline_pipeline.postgres_store import (
                upsert_raw_document,
            )
            upsert_raw_document(doc.to_dict())
            logger.debug("DB     ← %s", doc.doc_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DB upsert failed for %s: %s", doc.url, exc)
