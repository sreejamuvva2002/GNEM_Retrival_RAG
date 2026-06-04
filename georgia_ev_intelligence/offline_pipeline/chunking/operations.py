"""High-level offline chunking pipeline: DataFrame → parents + children.

WHY THIS FILE EXISTS:
  Thin orchestration layer that ties together parent_chunk.py and
  relationship.py.  Exposes a single clean function (build_parent_child_chunks)
  that the main index script calls, plus XLSX export helpers for debugging.

MAIN FUNCTION:
  build_parent_child_chunks(df) → ChunkingArtifacts
    1. One ParentRecord per DataFrame row (one per KB company record)
    2. Five ChildChunks per parent (identity / product_role / oem_relationship /
       location_employment / classification)
    3. Returns ChunkingArtifacts(parents, children)

EXPORT HELPERS:
  export_parent_chunks_to_xlsx — writes record_id + source_row_id + parent_chunk_text
  export_child_chunks_to_xlsx  — writes chunk_id + parent_record_id + chunk_type + embedding_text
  Both files are written to outputs/ every indexing run for inspection.

RELATIONSHIPS:
  Calls: parent_chunk.py, relationship.py
  Called by: offline_pipeline/index_pgvector.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .child_chunk import ChildChunk
from .parent_chunk import ParentRecord, build_parent_record
from .relationship import build_child_chunks


@dataclass
class ChunkingArtifacts:
    parents: list[ParentRecord]
    children: list[ChildChunk]


def build_parent_chunks(df: pd.DataFrame) -> list[ParentRecord]:
    """Build one ParentRecord per normalized DataFrame row."""
    df = df.reset_index(drop=True)
    return [build_parent_record(row) for _, row in df.iterrows()]


def build_child_chunks_for_parents(
    parents: list[ParentRecord],
    df: pd.DataFrame,
) -> list[ChildChunk]:
    """Build 5 child chunks for each parent using the same-index DataFrame rows."""
    df = df.reset_index(drop=True)
    children: list[ChildChunk] = []
    for parent, (_, row) in zip(parents, df.iterrows()):
        children.extend(build_child_chunks(parent, row))
    return children


def build_parent_child_chunks(df: pd.DataFrame) -> ChunkingArtifacts:
    """Build all parent records and their 5 child chunks from a normalized DataFrame."""
    df = df.reset_index(drop=True)
    parents = build_parent_chunks(df)
    children = build_child_chunks_for_parents(parents, df)
    return ChunkingArtifacts(parents=parents, children=children)


def export_parent_chunks_to_xlsx(parents: list[ParentRecord], path: Path | str) -> None:
    """Export parent chunks to Excel with record_id, source_row_id, parent_chunk_text."""
    rows = [
        {
            "record_id": p.record_id,
            "source_row_id": p.source_row_id,
            "parent_chunk_text": p.parent_chunk_text,
        }
        for p in parents
    ]
    pd.DataFrame(rows).to_excel(path, index=False)


def export_child_chunks_to_xlsx(children: list[ChildChunk], path: Path | str) -> None:
    """Export child chunks to Excel with chunk_id, parent_record_id, chunk_type, embedding_text."""
    rows = [
        {
            "chunk_id": c.chunk_id,
            "parent_record_id": c.parent_record_id,
            "chunk_type": c.chunk_type.value,
            "embedding_text": c.embedding_text,
        }
        for c in children
    ]
    pd.DataFrame(rows).to_excel(path, index=False)
