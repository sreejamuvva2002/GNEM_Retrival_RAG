"""File-type detection (README §25).

Detect the internal file type from multiple signals, in priority order:
  1. file extension (from the B2 key),
  2. magic bytes (content sniffing fallback).

The returned type string matches the converter registry keys in ``converters``.
Unknown types return ``None`` so the pipeline can mark them ``skipped_unsupported``
instead of crashing.
"""
from __future__ import annotations

import os

# Extension → internal file type.
EXTENSION_TO_TYPE: dict[str, str] = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".xml": "xml",
    ".docx": "docx",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".tiff": "image",
    ".tif": "image",
    ".bmp": "image",
    ".gif": "image",
    ".csv": "csv",
    ".tsv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
    ".txt": "text",
    ".md": "text",
}

# Internal type → canonical extension (for naming / MIME lookup).
TYPE_TO_EXTENSION: dict[str, str] = {
    "pdf": ".pdf",
    "html": ".html",
    "xml": ".xml",
    "docx": ".docx",
    "image": ".png",
    "csv": ".csv",
    "excel": ".xlsx",
    "json": ".json",
    "text": ".txt",
}

# Internal type → MIME type.
TYPE_TO_MIME: dict[str, str] = {
    "pdf": "application/pdf",
    "html": "text/html",
    "xml": "application/xml",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image": "image/png",
    "csv": "text/csv",
    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "json": "application/json",
    "text": "text/plain",
}


def detect_from_extension(key_or_name: str) -> str | None:
    """Detect internal type from the file extension of a key or filename."""
    _, ext = os.path.splitext(key_or_name.lower())
    return EXTENSION_TO_TYPE.get(ext)


def detect_from_bytes(raw: bytes) -> str | None:
    """Best-effort magic-byte sniffing fallback."""
    if not raw:
        return None
    head = raw[:512]
    # PDF
    if head.startswith(b"%PDF"):
        return "pdf"
    # ZIP-based OOXML (docx/xlsx) — disambiguate by inner part names.
    if head.startswith(b"PK\x03\x04"):
        window = raw[:4096]
        if b"word/" in window:
            return "docx"
        if b"xl/" in window:
            return "excel"
        return "docx"  # default OOXML guess
    # Image signatures
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image"
    if head.startswith(b"\xff\xd8\xff"):  # JPEG
        return "image"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image"
    if head.startswith(b"BM"):  # BMP
        return "image"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image"
    if head.startswith((b"II*\x00", b"MM\x00*")):  # TIFF
        return "image"
    # Text-ish formats
    stripped = head.lstrip()
    lowered = stripped[:64].lower()
    if lowered.startswith(b"<?xml"):
        return "xml"
    if lowered.startswith((b"<!doctype html", b"<html")):
        return "html"
    if stripped[:1] in (b"{", b"["):
        return "json"
    return None


def detect_file_type(key: str, raw: bytes | None = None) -> str | None:
    """Detect internal file type from key extension first, then magic bytes.

    Returns ``None`` for unsupported/unknown types.
    """
    by_ext = detect_from_extension(key)
    if by_ext:
        return by_ext
    if raw is not None:
        return detect_from_bytes(raw)
    return None
