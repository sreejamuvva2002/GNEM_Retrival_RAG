"""Backblaze B2 uploader using the S3-compatible API (boto3).

B2 bucket layout
----------------
raw-html/
    <doc_id>.html          ← full raw HTML bytes of every crawled page
    <doc_id>.pdf           ← raw PDF bytes
    <doc_id>.docx          ← raw DOCX bytes
jsonl/
    company_sites.jsonl    ← synced at end-of-crawl (full shard upload)
    ddg_search.jsonl
    gov_docs.jsonl
    news.jsonl
    other.jsonl

Required environment variables (set in .env):
    B2_KEY_ID              ← Backblaze Application Key ID
    B2_APPLICATION_KEY     ← Backblaze Application Key
    B2_BUCKET_NAME         ← Bucket name
    B2_ENDPOINT_URL        ← e.g. https://s3.us-west-004.backblazeb2.com
"""
from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Extension → S3 prefix mapping
_PREFIX = {
    "html":  "raw-html",
    "pdf":   "raw-pdf",
    "docx":  "raw-docx",
    "image": "raw-image",
    "excel": "raw-excel",
    "csv":   "raw-csv",
    "json":  "raw-json",
    "xml":   "raw-xml",
    "text":  "raw-text",
}

_EXT = {
    "html":  ".html",
    "pdf":   ".pdf",
    "docx":  ".docx",
    "image": ".png",
    "excel": ".xlsx",
    "csv":   ".csv",
    "json":  ".json",
    "xml":   ".xml",
    "text":  ".txt",
}


def _get_client():
    """Build a boto3 S3 client pointing at Backblaze B2."""
    try:
        import boto3  # type: ignore
    except ImportError as exc:
        raise ImportError("boto3 is required: pip install boto3>=1.34") from exc

    from georgia_ev_intelligence.shared import config

    if not config.B2_KEY_ID or not config.B2_APPLICATION_KEY or not config.B2_ENDPOINT_URL:
        raise RuntimeError(
            "B2_KEY_ID, B2_APPLICATION_KEY, and B2_ENDPOINT_URL must be set in .env "
            "to use Backblaze B2 upload."
        )

    return boto3.client(
        "s3",
        endpoint_url=config.B2_ENDPOINT_URL,
        aws_access_key_id=config.B2_KEY_ID,
        aws_secret_access_key=config.B2_APPLICATION_KEY,
    )


def _object_key(doc_id: str, file_type: str) -> str:
    """Build the S3 object key for a raw document."""
    prefix = _PREFIX.get(file_type, "raw-other")
    ext    = _EXT.get(file_type, ".bin")
    # doc_id is "sha256:<hash>" — strip the prefix for the filename
    safe_id = doc_id.replace("sha256:", "")
    return f"{prefix}/{safe_id}{ext}"


def upload_raw_bytes(
    doc_id: str,
    file_type: str,
    raw_bytes: bytes,
    bucket_name: str,
    *,
    metadata: Optional[dict] = None,
) -> str:
    """Upload raw bytes (HTML/PDF/DOCX) to B2.

    Returns the S3 object key on success, raises on failure.
    """
    client = _get_client()
    key = _object_key(doc_id, file_type)
    ext = _EXT.get(file_type, ".bin")
    content_type = mimetypes.types_map.get(ext, "application/octet-stream")

    extra: dict = {"ContentType": content_type}
    if metadata:
        # B2 / S3 metadata values must be strings
        extra["Metadata"] = {k: str(v) for k, v in metadata.items()}

    client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=raw_bytes,
        **extra,
    )
    logger.debug("B2 ← %s  (%d bytes)", key, len(raw_bytes))
    return key


def upload_jsonl_shard(shard_path: Path, bucket_name: str) -> str:
    """Upload (overwrite) a JSONL shard file to B2 under jsonl/<filename>.

    This is a full replace — B2 doesn't support append.  Call it at the
    end of a crawl run to sync the latest shard state.

    Returns the S3 object key.
    """
    client = _get_client()
    key = f"jsonl/{shard_path.name}"
    with shard_path.open("rb") as fh:
        client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=fh,
            ContentType="application/x-ndjson",
        )
    logger.info("B2 ← jsonl/%s  (%d bytes)", shard_path.name, shard_path.stat().st_size)
    return key


def sync_all_jsonl_shards(raw_docs_dir: Path, bucket_name: str) -> list[str]:
    """Upload every *.jsonl file in raw_docs_dir to B2.

    Call once at the end of a crawl run to sync all shards.
    Returns a list of uploaded object keys.
    """
    uploaded: list[str] = []
    for shard in sorted(raw_docs_dir.glob("*.jsonl")):
        try:
            key = upload_jsonl_shard(shard, bucket_name)
            uploaded.append(key)
        except Exception as exc:
            logger.warning("B2 JSONL sync failed for %s: %s", shard.name, exc)
    return uploaded
