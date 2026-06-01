"""Manifest / registry JSONL files (README §27).

Four append-only logs live under ``LOCAL_MANIFEST_DIR`` (crash-safe local writes) and are
synced to ``MANIFEST_B2_PREFIX`` after each batch (full-replace, mirroring
``b2_uploader.upload_jsonl_shard``):

    document_registry.jsonl            (one row per raw file; merged by document_id)
    markdown_conversion_manifest.jsonl (one row per successful conversion)
    failed_conversions.jsonl           (one row per failure)
    extraction_quality_report.jsonl    (one row per processed file)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from georgia_ev_intelligence.shared import config

logger = logging.getLogger(__name__)

REGISTRY = "document_registry.jsonl"
CONVERSIONS = "markdown_conversion_manifest.jsonl"
FAILURES = "failed_conversions.jsonl"
QUALITY = "extraction_quality_report.jsonl"

_ALL_FILES = (REGISTRY, CONVERSIONS, FAILURES, QUALITY)


class ManifestStore:
    """Local JSONL manifest writer with optional B2 sync."""

    def __init__(self, local_dir: Path | None = None, *, use_b2: bool = True):
        self.local_dir = Path(local_dir or config.LOCAL_MANIFEST_DIR)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.use_b2 = use_b2

    # ---- generic append-only logs ---------------------------------------- #

    def append(self, filename: str, record: dict) -> None:
        path = self.local_dir / filename
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def append_conversion(self, record: dict) -> None:
        self.append(CONVERSIONS, record)

    def append_failure(self, record: dict) -> None:
        self.append(FAILURES, record)

    def append_quality(self, record: dict) -> None:
        self.append(QUALITY, record)

    # ---- registry (merged by document_id) -------------------------------- #

    def load_registry(self) -> dict[str, dict]:
        """Load the registry into a {document_id: record} map (last write wins)."""
        path = self.local_dir / REGISTRY
        records: dict[str, dict] = {}
        if not path.exists():
            return records
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = rec.get("document_id")
                if doc_id:
                    records[doc_id] = rec
        return records

    def save_registry(self, records: dict[str, dict]) -> None:
        """Rewrite the registry file from a {document_id: record} map."""
        path = self.local_dir / REGISTRY
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in records.values():
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        tmp.replace(path)

    # ---- B2 sync --------------------------------------------------------- #

    def sync_to_b2(self) -> list[str]:
        """Upload every local manifest file to MANIFEST_B2_PREFIX. Best-effort."""
        if not self.use_b2 or not config.B2_BUCKET_NAME:
            return []
        from . import b2_client

        uploaded: list[str] = []
        prefix = config.MANIFEST_B2_PREFIX.rstrip("/") + "/"
        for name in _ALL_FILES:
            path = self.local_dir / name
            if not path.exists():
                continue
            try:
                key = b2_client.upload_text(
                    prefix + name,
                    path.read_text(encoding="utf-8"),
                    content_type="application/x-ndjson",
                )
                uploaded.append(key)
            except Exception as exc:
                logger.warning("Manifest B2 sync failed for %s: %s", name, exc)
        return uploaded
