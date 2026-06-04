"""Backblaze B2 read/write client for the Markdown pipeline.

Reuses ``kb_builder.b2_uploader._get_client`` for credentials and the S3-compatible
boto3 client, then adds the listing/downloading/existence operations the upload-only
uploader lacks (README §24).

All functions raise a clear RuntimeError (via ``_get_client``) when B2 credentials are
not configured; callers should run with ``--local-only`` to skip B2 entirely.
"""
from __future__ import annotations

import logging
from typing import Iterator, Optional

from georgia_ev_intelligence.kb_builder.b2_uploader import _get_client
from georgia_ev_intelligence.shared import config

logger = logging.getLogger(__name__)


def _bucket() -> str:
    if not config.B2_BUCKET_NAME:
        raise RuntimeError("B2_BUCKET_NAME must be set in .env to use Backblaze B2.")
    return config.B2_BUCKET_NAME


def list_objects(prefix: str) -> Iterator[dict]:
    """Yield object summaries under ``prefix`` (paginated).

    Each yielded dict has at least: ``Key``, ``Size``, ``ETag``, ``LastModified``.
    """
    client = _get_client()
    bucket = _bucket()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            yield item


def download_object(key: str) -> bytes:
    """Download an object's full bytes."""
    client = _get_client()
    response = client.get_object(Bucket=_bucket(), Key=key)
    return response["Body"].read()


def object_exists(key: str) -> bool:
    """Return True if the object exists (HEAD, 404-tolerant)."""
    client = _get_client()
    try:
        client.head_object(Bucket=_bucket(), Key=key)
        return True
    except Exception as exc:  # botocore ClientError 404 or any access miss
        code = getattr(getattr(exc, "response", {}), "get", lambda *_: {})("Error", {})
        if isinstance(code, dict) and code.get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        # head_object raises ClientError with response dict for 404
        resp = getattr(exc, "response", None)
        if isinstance(resp, dict):
            status = resp.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 404:
                return False
        logger.debug("object_exists(%s) ambiguous error: %s", key, exc)
        return False


def get_object_metadata(key: str) -> dict:
    """Return object metadata (HEAD response: ContentType, ContentLength, Metadata, ...)."""
    client = _get_client()
    return client.head_object(Bucket=_bucket(), Key=key)


def upload_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload arbitrary bytes to ``key``. Returns the key."""
    client = _get_client()
    client.put_object(Bucket=_bucket(), Key=key, Body=data, ContentType=content_type)
    logger.debug("B2 ← %s (%d bytes)", key, len(data))
    return key


def upload_markdown(key: str, markdown_text: str) -> str:
    """Upload a Markdown document as UTF-8."""
    return upload_bytes(
        key,
        markdown_text.encode("utf-8"),
        content_type="text/markdown; charset=utf-8",
    )


def upload_text(key: str, text: str, content_type: str) -> str:
    """Upload arbitrary text (e.g. JSONL manifests) as UTF-8."""
    return upload_bytes(key, text.encode("utf-8"), content_type=content_type)
