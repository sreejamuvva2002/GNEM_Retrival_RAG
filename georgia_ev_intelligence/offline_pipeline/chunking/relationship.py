"""Parent-child chunk relationship building and validation."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from .child_chunk import (
    CHILD_CHUNK_FIELDS,
    ChildChunk,
    ChildChunkType,
    build_child_metadata,
    build_embedding_text,
)
from .parent_chunk import ParentRecord


def build_child_chunk(
    parent: ParentRecord,
    row: pd.Series,
    chunk_type: ChildChunkType,
) -> ChildChunk:
    fields = CHILD_CHUNK_FIELDS[chunk_type]
    chunk_id = f"{parent.record_id}_{chunk_type.value.upper()}"
    return ChildChunk(
        chunk_id=chunk_id,
        parent_record_id=parent.record_id,
        chunk_type=chunk_type,
        source_type="excel",
        embedding_text=build_embedding_text(row, chunk_type, fields),
        metadata=build_child_metadata(row, fields),
    )


def build_child_chunks(parent: ParentRecord, row: pd.Series) -> list[ChildChunk]:
    return [build_child_chunk(parent, row, chunk_type) for chunk_type in ChildChunkType]


def validate_relationships(
    parents: list[ParentRecord],
    children: list[ChildChunk],
) -> None:
    """Validate parent-child integrity.

    Checks:
    - Total children == len(parents) * 5
    - Each parent has exactly 5 children
    - Every child.parent_record_id matches a known parent.record_id
    - No duplicate child_id values
    """
    parent_ids = {p.record_id for p in parents}
    expected_total = len(parents) * 5

    if len(children) != expected_total:
        raise ValueError(
            f"Expected {expected_total} child chunks ({len(parents)} parents × 5), "
            f"got {len(children)}."
        )

    chunk_ids = [c.chunk_id for c in children]
    duplicate_ids = [cid for cid, count in Counter(chunk_ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(
            f"Duplicate child_id values found: {duplicate_ids}"
        )

    orphaned = [c.chunk_id for c in children if c.parent_record_id not in parent_ids]
    if orphaned:
        raise ValueError(
            f"{len(orphaned)} child chunk(s) have unknown parent_record_id. "
            f"First offender: {orphaned[0]}"
        )

    children_per_parent = Counter(c.parent_record_id for c in children)
    wrong_count = [
        (pid, count)
        for pid, count in children_per_parent.items()
        if count != 5
    ]
    if wrong_count:
        raise ValueError(
            f"Expected exactly 5 children per parent. Violations: {wrong_count[:5]}"
        )
