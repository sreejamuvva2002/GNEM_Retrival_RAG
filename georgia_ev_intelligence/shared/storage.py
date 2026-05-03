"""
Backblaze B2 storage client.
S3-compatible — uses boto3. Handles upload, download, and existence checks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from shared.config import Config
from shared.logger import get_logger

logger = get_logger("shared.storage")

_s3_client = None


def _get_client():
    global _s3_client
    if _s3_client is None:
        cfg = Config.get()
        _s3_client = boto3.client(
            "s3",
            endpoint_url=cfg.b2_endpoint,
            aws_access_key_id=cfg.b2_access_key,
            aws_secret_access_key=cfg.b2_secret_key,
            region_name=cfg.b2_region,
        )
        logger.info("Backblaze B2 client initialized (bucket=%s)", cfg.b2_bucket)
    return _s3_client


def get_b2_client():
    """Public accessor for the B2 boto3 client. Use for list/verify operations."""
    return _get_client()


def upload_bytes(content: bytes, key: str, content_type: str = "application/octet-stream") -> str:
    """
    Upload raw bytes to B2. Returns the full B2 key.

    Args:
        content: Raw bytes to upload
        key: B2 object key (path within bucket), e.g. "companies/hanwha/doc_001.pdf"
        content_type: MIME type

    Returns:
        The B2 key of the uploaded object
    """
    cfg = Config.get()
    client = _get_client()
    client.put_object(
        Bucket=cfg.b2_bucket,
        Key=key,
        Body=content,
        ContentType=content_type,
    )
    logger.debug("Uploaded %d bytes to B2: %s", len(content), key)
    return key


def upload_file(local_path: Path, key: str) -> str:
    """Upload a local file to B2. Returns the B2 key."""
    cfg = Config.get()
    client = _get_client()
    client.upload_file(str(local_path), cfg.b2_bucket, key)
    logger.debug("Uploaded file to B2: %s → %s", local_path, key)
    return key


def download_bytes(key: str) -> bytes:
    """
    Download an object from B2 as bytes.

    Raises:
        FileNotFoundError: if the key does not exist
    """
    cfg = Config.get()
    client = _get_client()
    try:
        response = client.get_object(Bucket=cfg.b2_bucket, Key=key)
        return response["Body"].read()
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code in ("NoSuchKey", "404"):
            raise FileNotFoundError(f"B2 key not found: {key}") from exc
        raise


def key_exists(key: str) -> bool:
    """Check if a B2 key exists without downloading it."""
    cfg = Config.get()
    client = _get_client()
    try:
        client.head_object(Bucket=cfg.b2_bucket, Key=key)
        return True
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code in ("NoSuchKey", "404", "403"):
            return False
        raise


def list_keys(prefix: str) -> list[str]:
    """List all B2 keys with the given prefix. Returns key names."""
    cfg = Config.get()
    client = _get_client()
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=cfg.b2_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def make_document_key(company_name: str, content_hash: str, extension: str) -> str:
    """
    Build a standardized B2 key for a document.

    Format: documents/{company_slug}/{hash}{extension}
    Example: documents/hanwha_q_cells/abc123def456.pdf

    Using content hash ensures deduplication — same content never stored twice.
    """
    slug = _slugify(company_name)
    ext = extension.lstrip(".")
    return f"documents/{slug}/{content_hash}.{ext}"


def _slugify(text: str) -> str:
    """Convert company name to a safe filesystem-like string."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:80]  # Cap at 80 chars


def verify_connection() -> bool:
    """Test that B2 is reachable. Returns True if OK."""
    try:
        cfg = Config.get()
        client = _get_client()
        client.head_bucket(Bucket=cfg.b2_bucket)
        logger.info("Backblaze B2 connection verified (bucket=%s)", cfg.b2_bucket)
        return True
    except Exception as exc:
        logger.error("Backblaze B2 connection FAILED: %s", exc)
        return False
