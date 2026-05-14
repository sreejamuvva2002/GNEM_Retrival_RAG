"""Child chunk types and adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import pandas as pd

from .parent_chunk import ParentRecord


class ChildChunkType(str, Enum):
    """Supported child chunk types for Excel row parent records."""

    IDENTITY = "identity"
    PRODUCT_ROLE = "product_role"
    OEM_RELATIONSHIP = "oem_relationship"
    LOCATION_EMPLOYMENT = "location_employment"
    CLASSIFICATION = "classification"


CHILD_CHUNK_FIELDS: dict[ChildChunkType, tuple[str, ...]] = {
    ChildChunkType.IDENTITY: (
        "company",
        "category",
        "industry_group",
        "updated_location",
    ),
    ChildChunkType.PRODUCT_ROLE: (
        "company",
        "ev_supply_chain_role",
        "product_service",
        "ev_battery_relevant",
    ),
    ChildChunkType.OEM_RELATIONSHIP: (
        "company",
        "primary_oems",
        "supplier_or_affiliation_type",
        "category",
    ),
    ChildChunkType.LOCATION_EMPLOYMENT: (
        "company",
        "updated_location",
        "address",
        "latitude",
        "longitude",
        "employment",
    ),
    ChildChunkType.CLASSIFICATION: (
        "company",
        "primary_facility_type",
        "classification_method",
        "category",
        "ev_battery_relevant",
    ),
}


CHILD_CHUNK_DESCRIPTIONS: dict[ChildChunkType, str] = {
    ChildChunkType.IDENTITY: "Company identity, tier/category, industry group, and location.",
    ChildChunkType.PRODUCT_ROLE: "EV supply chain role, product/service, and EV/battery relevance.",
    ChildChunkType.OEM_RELATIONSHIP: "OEM relationship and supplier/affiliation type.",
    ChildChunkType.LOCATION_EMPLOYMENT: "Location, address, coordinates, and employment information.",
    ChildChunkType.CLASSIFICATION: "Facility type, classification method, category, and EV relevance.",
}


FIELD_LABELS: dict[str, str] = {
    "_row_id": "Source Row ID",
    "company": "Company",
    "category": "Category",
    "industry_group": "Industry Group",
    "updated_location": "Updated Location",
    "address": "Address",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "primary_facility_type": "Primary Facility Type",
    "ev_supply_chain_role": "EV Supply Chain Role",
    "primary_oems": "Primary OEMs",
    "supplier_or_affiliation_type": "Supplier or Affiliation Type",
    "employment": "Employment",
    "product_service": "Product / Service",
    "ev_battery_relevant": "EV / Battery Relevant",
    "classification_method": "Classification Method",
}


COMMON_METADATA_FIELDS: tuple[str, ...] = (
    "_row_id",
    "company",
    "category",
    "industry_group",
    "updated_location",
    "primary_facility_type",
    "ev_supply_chain_role",
    "primary_oems",
    "supplier_or_affiliation_type",
    "employment",
    "ev_battery_relevant",
    "classification_method",
)


@dataclass(frozen=True)
class ChildChunk:
    chunk_id: str
    parent_record_id: str
    embedding_text: str
    parent: ParentRecord
    chunk_type: ChildChunkType
    fields: tuple[str, ...]
    metadata: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        """
        Build payload for vector DB / retrieval storage.

        The parent payload keeps the complete Excel row.
        Child-specific fields identify this chunk type and its selected fields.
        """
        char_count = len(self.embedding_text)

        payload = self.parent.payload()
        payload.update(
            {
                "chunk_id": self.chunk_id,
                "parent_record_id": self.parent_record_id,
                "chunk_type": self.chunk_type.value,
                "chunk_fields": list(self.fields),
                "chunk_description": CHILD_CHUNK_DESCRIPTIONS[self.chunk_type],
                "char_count": char_count,
                "token_estimate": round(char_count / 4),
            }
        )

        # Attach exact metadata values for filtering.
        payload.update(self.metadata)

        # Keep embedding text in payload only if your current pipeline needs it.
        # If you store text separately in PostgreSQL, you can remove this field.
        payload["embedding_text"] = self.embedding_text

        return _json_safe(payload)


class ChildChunkAdapter(Protocol):
    """Adapter contract for creating child chunks from a parent record."""

    def build(self, parent: ParentRecord, child_index: int = 0) -> ChildChunk:
        ...


class FieldGroupChildChunkAdapter:
    """Create one field-group child chunk from a parent record."""

    def __init__(self, chunk_type: ChildChunkType):
        self.chunk_type = chunk_type
        self.fields = CHILD_CHUNK_FIELDS[chunk_type]

    def build(self, parent: ParentRecord, child_index: int = 0) -> ChildChunk:
        chunk_id = f"{parent.record_id}_{self.chunk_type.value.upper()}_{child_index:03d}"

        return ChildChunk(
            chunk_id=chunk_id,
            parent_record_id=parent.record_id,
            embedding_text=build_field_group_embedding_text(
                parent=parent,
                chunk_type=self.chunk_type,
                fields=self.fields,
            ),
            parent=parent,
            chunk_type=self.chunk_type,
            fields=self.fields,
            metadata=build_child_metadata(parent=parent),
        )


class MultiFieldGroupChildChunkAdapter:
    """
    Create all recommended child chunks for one parent record.

    One parent row produces:
    - identity child
    - product_role child
    - oem_relationship child
    - location_employment child
    - classification child
    """

    def __init__(
        self,
        chunk_types: tuple[ChildChunkType, ...] | None = None,
    ):
        self.chunk_types = chunk_types or tuple(ChildChunkType)

    def build_all(self, parent: ParentRecord) -> list[ChildChunk]:
        chunks: list[ChildChunk] = []

        for child_index, chunk_type in enumerate(self.chunk_types):
            adapter = FieldGroupChildChunkAdapter(chunk_type=chunk_type)
            chunks.append(adapter.build(parent=parent, child_index=child_index))

        return chunks

    def build(self, parent: ParentRecord, child_index: int = 0) -> ChildChunk:
        """
        Compatibility method.

        Returns one child chunk by index.
        Prefer build_all() when indexing the full KB.
        """
        chunk_type = self.chunk_types[child_index % len(self.chunk_types)]
        adapter = FieldGroupChildChunkAdapter(chunk_type=chunk_type)
        return adapter.build(parent=parent, child_index=child_index)


def build_all_child_chunks(parent: ParentRecord) -> list[ChildChunk]:
    """Convenience function to build all child chunks for a parent row."""
    return MultiFieldGroupChildChunkAdapter().build_all(parent)


def build_field_group_embedding_text(
    parent: ParentRecord,
    chunk_type: ChildChunkType,
    fields: tuple[str, ...],
) -> str:
    """
    Build embedding text for a selected field group.

    This does not normalize KB values.
    It only formats selected raw values as labeled text.
    """
    lines: list[str] = [
        f"Chunk Type: {chunk_type.value}",
        f"Purpose: {CHILD_CHUNK_DESCRIPTIONS[chunk_type]}",
    ]

    for field_name in fields:
        value = parent.row_data.get(field_name)

        if _is_missing(value):
            continue

        label = FIELD_LABELS.get(field_name, field_name)
        lines.append(f"{label}: {_to_text(value)}")

    return "\n".join(lines)


def build_child_metadata(parent: ParentRecord) -> dict[str, Any]:
    """
    Build exact metadata values attached to every child chunk.

    Metadata is used for filtering, deduplication, and parent lookup.
    This does not normalize KB values.
    """
    metadata: dict[str, Any] = {
        "source_type": "excel",
        "parent_record_id": parent.record_id,
        "source_row_id": parent.source_row_id,
    }

    for field_name in COMMON_METADATA_FIELDS:
        if field_name in parent.row_data:
            metadata[field_name] = parent.row_data[field_name]

    return _json_safe(metadata)


def _to_text(value: Any) -> str:
    """
    Convert value to text only for embedding display.

    This does not normalize the KB meaning.
    """
    if _is_missing(value):
        return ""

    return str(value)


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Convert payload values into JSON-safe values.

    This is serialization safety, not KB normalization.
    """
    return {key: _to_json_safe_value(value) for key, value in payload.items()}


def _to_json_safe_value(value: Any) -> Any:
    """Convert one value into a JSON-safe Python value."""

    if _is_missing(value):
        return None

    # Convert pandas/numpy scalar types to native Python values.
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, (list, tuple)):
        return [_to_json_safe_value(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _to_json_safe_value(val)
            for key, val in value.items()
        }

    return str(value)


def _is_missing(value: Any) -> bool:
    """
    Safely detect None, NaN, pd.NA, and pd.NaT.

    Avoids errors when pd.isna() returns arrays for list-like values.
    """
    if value is None:
        return True

    try:
        result = pd.isna(value)
    except Exception:
        return False

    if isinstance(result, bool):
        return result

    return False