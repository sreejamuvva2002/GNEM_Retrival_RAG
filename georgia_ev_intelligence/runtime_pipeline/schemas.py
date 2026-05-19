"""Shared runtime data models for the retained hybrid retrieval path."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievedChildChunk:
    """A single child chunk returned by dense or BM25 retrieval."""

    chunk_id: str
    parent_record_id: str
    chunk_type: str
    metadata: dict[str, Any]


@dataclass
class ParentContext:
    """A parent chunk fetched for answer generation."""

    record_id: str
    source_row_id: int
    parent_chunk_text: str
