"""Chunking operations built from parent, child, and relationship adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .child_chunks import (
    ChildChunk,
    ChildChunkAdapter,
    MultiFieldGroupChildChunkAdapter,
)
from .parent_chunk import ParentChunkAdapter, ParentRecord
from .relationships import ParentChildRelationshipAdapter


def build_parent_chunks(
    df: pd.DataFrame,
    *,
    parent_adapter: ParentChunkAdapter | None = None,
) -> list[ParentRecord]:
    """
    Build parent chunks from a normalized/raw KB DataFrame.

    Each parent chunk preserves the complete Excel row plus parent metadata.
    """
    parent_builder = parent_adapter or ParentChunkAdapter()
    parents: list[ParentRecord] = []

    for idx, row in df.reset_index(drop=True).iterrows():
        parents.append(parent_builder.build(row=row, row_index=idx))

    return parents


def build_parent_child_chunks(
    df: pd.DataFrame,
    *,
    parent_adapter: ParentChunkAdapter | None = None,
    child_adapter: Any | None = None,
    relationship_adapter: ParentChildRelationshipAdapter | None = None,
) -> list[ChildChunk]:
    """
    Build parent-child chunks from a normalized/raw KB DataFrame.

    Each parent contains the complete Excel row.
    Each parent produces multiple field-group child chunks:
    - identity
    - product_role
    - oem_relationship
    - location_employment
    - classification

    Child chunks are used for embedding/retrieval.
    Parent rows are used for final answer generation.
    """
    parent_builder = parent_adapter or ParentChunkAdapter()
    child_builder = child_adapter or MultiFieldGroupChildChunkAdapter()
    relationship_builder = relationship_adapter or ParentChildRelationshipAdapter()

    chunks: list[ChildChunk] = []

    for idx, row in df.reset_index(drop=True).iterrows():
        parent = parent_builder.build(row=row, row_index=idx)
        children = _build_children(
            parent=parent,
            child_builder=child_builder,
        )

        relations = relationship_builder.relate(
            parent=parent,
            children=children,
        )

        chunks.extend(relation.child for relation in relations)

    return chunks


def parent_chunks_to_dataframe(parents: list[ParentRecord]) -> pd.DataFrame:
    """
    Convert parent chunks to a DataFrame for inspection/export.

    Output includes:
    - record_id
    - source_row_id
    - parent_chunk_text
    """
    return pd.DataFrame(
        [
            {
                "record_id": parent.record_id,
                "source_row_id": parent.source_row_id,
                "parent_chunk_text": build_parent_chunk_text(parent),
            }
            for parent in parents
        ]
    )


def export_parent_chunks_to_xlsx(
    parents: list[ParentRecord],
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Store parent chunks with complete parent details in a separate XLSX file.

    Returns the exported DataFrame so callers can inspect or test it.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    parents_df = parent_chunks_to_dataframe(parents)
    parents_df.to_excel(path, sheet_name="parent_chunks", index=False)
    return parents_df


def _build_children(
    parent: ParentRecord,
    child_builder: Any,
) -> list[ChildChunk]:
    """
    Build child chunks from a parent record.

    Supports:
    - MultiFieldGroupChildChunkAdapter with build_all()
    - Any single-child adapter with build()
    """
    if hasattr(child_builder, "build_all"):
        children = child_builder.build_all(parent)

        if not isinstance(children, list):
            raise TypeError(
                "child_builder.build_all(parent) must return list[ChildChunk]."
            )

        return children

    if hasattr(child_builder, "build"):
        child = child_builder.build(parent=parent, child_index=0)

        if not isinstance(child, ChildChunk):
            raise TypeError(
                "child_builder.build(parent, child_index=0) must return ChildChunk."
            )

        return [child]

    raise TypeError(
        "child_builder must provide either build_all(parent) or "
        "build(parent, child_index=0)."
    )


def chunks_to_dataframe(chunks: list[ChildChunk]) -> pd.DataFrame:
    """
    Convert chunks to a DataFrame for inspection/export.

    Output includes:
    - parent record fields
    - child chunk metadata
    - chunk type
    - selected child fields
    - embedding text
    """
    rows: list[dict[str, Any]] = []

    for chunk in chunks:
        rows.append(chunk.payload())

    return pd.DataFrame(rows)


def build_parent_chunk_text(parent: ParentRecord) -> str:
    """
    Build the exact formatted full-row parent chunk text.

    This is a display/retrieval representation only. It does not normalize or
    modify the underlying parent row values.
    """
    row_data = parent.row_data

    lines = [
        "Parent Chunk",
        f"Record ID: {parent.record_id}",
        f"Source Row ID: {_to_text(parent.source_row_id)}",
        "",
        f"Company: {_to_text(row_data.get('company'))}",
        f"Category: {_to_text(row_data.get('category'))}",
        f"Industry Group: {_to_text(row_data.get('industry_group'))}",
        f"Updated Location: {_to_text(row_data.get('updated_location'))}",
        f"Address: {_to_text(row_data.get('address'))}",
        f"Latitude: {_to_text(row_data.get('latitude'))}",
        f"Longitude: {_to_text(row_data.get('longitude'))}",
        f"Primary Facility Type: {_to_text(row_data.get('primary_facility_type'))}",
        f"EV Supply Chain Role: {_to_text(row_data.get('ev_supply_chain_role'))}",
        f"Primary OEMs: {_to_text(row_data.get('primary_oems'))}",
        "Supplier or Affiliation Type: "
        f"{_to_text(row_data.get('supplier_or_affiliation_type'))}",
        f"Employment: {_to_text(row_data.get('employment'))}",
        f"Product / Service: {_to_text(row_data.get('product_service'))}",
        f"EV / Battery Relevant: {_to_text(row_data.get('ev_battery_relevant'))}",
        f"Classification Method: {_to_text(row_data.get('classification_method'))}",
        f"Original Row ID: {_to_text(row_data.get('_row_id'))}",
    ]

    return "\n".join(lines)


def build_embedding_text(parent: ParentRecord) -> str:
    """
    Build full-row embedding-style text from the complete parent row.

    This function is kept for compatibility/debugging.
    The main indexing pipeline should use field-group child chunks instead.
    """
    lines: list[str] = []

    for column_name, value in parent.row_data.items():
        lines.append(f"{column_name}: {_to_text(value)}")

    return "\n".join(lines)


def _to_text(value: Any) -> str:
    """
    Convert value to text for debug/export display only.

    This does not normalize KB meaning.
    """
    return str(value)
