"""Parent-child chunk relationship helpers."""
from __future__ import annotations

from ..child_chunk import ChildChunk, build_embedding_text
from ..parent_chunk import ParentRecord


def build_child_chunk(index: int, parent: ParentRecord) -> ChildChunk:
    return ChildChunk(
        chunk_id=f"GA_AUTO_{index:04d}",
        parent_record_id=parent.record_id,
        embedding_text=build_embedding_text(parent),
        parent=parent,
    )
