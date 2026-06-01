"""Hashing utilities (README §41).

Two hash kinds:
  * raw_sha256  — exact duplicate raw files (hash of raw bytes)
  * text_sha256 — duplicate *content* across formats/URLs (hash of normalized text)
"""
from __future__ import annotations

import hashlib


def sha256_bytes(data: bytes) -> str:
    """SHA256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """SHA256 of normalized text: trimmed, blank-line-free, lowercased."""
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return hashlib.sha256(normalized.lower().encode("utf-8")).hexdigest()
