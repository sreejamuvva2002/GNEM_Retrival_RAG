"""Orchestration: per-document conversion + batch runner (README §30, §32, §43).

Each document is downloaded, detected, converted, wrapped with YAML front matter,
quality-checked, and uploaded to the processed Markdown prefix; manifests track every
outcome. A single document's failure never aborts the batch.
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from georgia_ev_intelligence.shared import config

from . import b2_client, detector, metadata
from .converters import get_converter
from .hashing import sha256_bytes, sha256_text
from .manifest import ManifestStore
from .quality import compute_quality

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PipelineOptions:
    output_prefix: str
    extraction_version: str
    use_b2: bool = True            # upload markdown + manifests to B2
    write_local: bool = True       # also write markdown to LOCAL_MARKDOWN_DIR
    force: bool = False            # ignore incremental skip
    max_file_size_mb: int = 100


def markdown_key_for(document_id: str, output_prefix: str) -> str:
    prefix = output_prefix.rstrip("/") + "/"
    return f"{prefix}source_documents/{document_id}.md"


def _should_skip(record: dict, raw_hash: str, opts: PipelineOptions) -> bool:
    """Incremental skip (README §30)."""
    if opts.force:
        return False
    if record.get("processing_status") != "success":
        return False
    if record.get("extraction_version") != opts.extraction_version:
        return False
    if record.get("raw_sha256") != raw_hash:
        return False
    return bool(record.get("markdown_key"))


def _local_markdown_path(document_id: str) -> Path:
    base = Path(config.LOCAL_MARKDOWN_DIR) / "source_documents"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{document_id}.md"


def process_document(
    record: dict,
    *,
    store: ManifestStore,
    opts: PipelineOptions,
) -> str:
    """Process one registry record end-to-end. Returns the final status string."""
    document_id = record["document_id"]
    source_key = record["source_key"]

    # 1. Download raw bytes.
    try:
        raw_bytes = b2_client.download_object(source_key)
    except Exception as exc:
        return _fail(record, store, "download", "download_failed", exc, retryable=True)

    if not raw_bytes:
        return _fail(record, store, "download", "empty_file", ValueError("0 bytes"), retryable=False)

    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > opts.max_file_size_mb:
        return _skip(record, store, "skipped_low_value",
                     f"File {size_mb:.1f}MB exceeds MAX_FILE_SIZE_MB={opts.max_file_size_mb}")

    raw_hash = sha256_bytes(raw_bytes)
    record["raw_sha256"] = raw_hash

    # 2. Incremental skip — keep the existing 'success' registry status, but report
    #    a distinct 'skipped' outcome so re-runs are visible in the summary.
    if _should_skip(record, raw_hash, opts):
        logger.info("skip[unchanged] %s — already converted", document_id)
        return "skipped"

    # 3. Detect type.
    file_type = detector.detect_file_type(source_key, raw_bytes)
    converter = get_converter(file_type)
    if converter is None:
        return _skip(record, store, "skipped_unsupported", f"unsupported type for {source_key}")

    # 4. Convert.
    try:
        result = converter.convert(raw_bytes, source_name=source_key)
    except Exception as exc:
        return _fail(record, store, f"{file_type}_conversion", "parser_error", exc, retryable=False)

    # 5. Front matter + Markdown + quality.
    md_key = markdown_key_for(document_id, opts.output_prefix)
    quality = compute_quality(result.markdown_body, result.metadata, result.quality_status)
    record["text_sha256"] = sha256_text(result.markdown_body)

    front = metadata.build_frontmatter(
        registry_record=record,
        raw_hash=raw_hash,
        result=result,
        quality=quality,
        markdown_key=md_key,
        extraction_version=opts.extraction_version,
    )
    markdown_text = metadata.build_markdown(front, result.markdown_body)

    # Quality report (always logged).
    quality_row = {"document_id": document_id, **quality, "file_type": file_type}
    store.append_quality(quality_row)

    if quality["quality_status"] == "failed":
        return _fail(
            record, store, "quality_check", "quality_check_failed",
            ValueError(f"quality failed: text_length={quality['text_length']}"),
            retryable=False,
        )

    # 6. Persist Markdown (local + B2).
    if opts.write_local:
        try:
            _local_markdown_path(document_id).write_text(markdown_text, encoding="utf-8")
        except Exception as exc:
            logger.warning("Local markdown write failed for %s: %s", document_id, exc)

    if opts.use_b2:
        try:
            b2_client.upload_markdown(md_key, markdown_text)
        except Exception as exc:
            return _fail(record, store, "upload", "upload_failed", exc, retryable=True)

    # 7. Manifest + registry update.
    md_hash = sha256_bytes(markdown_text.encode("utf-8"))
    store.append_conversion({
        "document_id": document_id,
        "source_key": source_key,
        "markdown_key": md_key,
        "file_type": file_type,
        "raw_sha256": raw_hash,
        "markdown_sha256": md_hash,
        "text_sha256": record["text_sha256"],
        "extraction_tool": front.get("extraction_tool"),
        "extraction_version": opts.extraction_version,
        "status": quality["quality_status"],
        "quality_score": quality["quality_score"],
        "extracted_at": _now(),
    })

    status = "needs_review" if quality["quality_status"] in ("needs_review", "low_value") else "success"
    record["processing_status"] = status
    record["markdown_key"] = md_key
    record["error_key"] = None
    record["last_processed_at"] = _now()
    record["extraction_version"] = opts.extraction_version
    return status


# --------------------------------------------------------------------------- #
# outcome helpers
# --------------------------------------------------------------------------- #

def _skip(record, store, status, reason, *, touch=True):
    record["processing_status"] = status
    if touch:
        record["last_processed_at"] = _now()
    logger.info("skip[%s] %s — %s", status, record["document_id"], reason)
    return status


def _fail(record, store, stage, error_type, exc, *, retryable):
    document_id = record["document_id"]
    error_record = {
        "document_id": document_id,
        "source_key": record.get("source_key"),
        "file_type": record.get("file_extension"),
        "stage": stage,
        "error_type": error_type,
        "error_message": str(exc),
        "traceback_short": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
        "retryable": retryable,
        "failed_at": _now(),
    }
    store.append_failure(error_record)
    record["processing_status"] = "failed"
    record["last_processed_at"] = _now()
    logger.warning("fail[%s] %s — %s: %s", stage, document_id, error_type, exc)
    return "failed"


# --------------------------------------------------------------------------- #
# batch runner
# --------------------------------------------------------------------------- #

def run_batch(
    records: list[dict],
    *,
    store: ManifestStore,
    opts: PipelineOptions,
    registry: dict[str, dict],
    batch_size: int = 100,
) -> dict[str, int]:
    """Process ``records`` in chunks, syncing manifests + registry after each chunk."""
    counts: dict[str, int] = {}
    total = len(records)
    for start in range(0, total, batch_size):
        chunk = records[start:start + batch_size]
        for i, record in enumerate(chunk, start=start + 1):
            logger.info("Processing %d/%d: %s", i, total, record["source_key"])
            try:
                status = process_document(record, store=store, opts=opts)
            except Exception as exc:  # safety net — never abort the batch
                status = _fail(record, store, "pipeline", "parser_error", exc, retryable=False)
            counts[status] = counts.get(status, 0) + 1
            registry[record["document_id"]] = record
        # Persist progress after each batch (resumable).
        store.save_registry(registry)
        store.sync_to_b2()
    return counts
