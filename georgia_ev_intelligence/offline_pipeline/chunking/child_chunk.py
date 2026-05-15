"""Child chunk construction for offline embedding."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from georgia_ev_intelligence.shared.data.loader import KBColumns


class ChildChunkType(str, Enum):
    IDENTITY = "identity"
    PRODUCT_ROLE = "product_role"
    OEM_RELATIONSHIP = "oem_relationship"
    LOCATION_EMPLOYMENT = "location_employment"
    CLASSIFICATION = "classification"


CHILD_CHUNK_FIELDS: dict[ChildChunkType, tuple[str, ...]] = {
    ChildChunkType.IDENTITY: (
        KBColumns.COMPANY,
        KBColumns.CATEGORY,
        KBColumns.INDUSTRY_GROUP,
        KBColumns.UPDATED_LOCATION,
    ),
    ChildChunkType.PRODUCT_ROLE: (
        KBColumns.COMPANY,
        KBColumns.EV_SUPPLY_CHAIN_ROLE,
        KBColumns.PRODUCT_SERVICE,
        KBColumns.EV_BATTERY_RELEVANT,
    ),
    ChildChunkType.OEM_RELATIONSHIP: (
        KBColumns.COMPANY,
        KBColumns.PRIMARY_OEMS,
        KBColumns.SUPPLIER_OR_AFFILIATION_TYPE,
        KBColumns.CATEGORY,
    ),
    ChildChunkType.LOCATION_EMPLOYMENT: (
        KBColumns.COMPANY,
        KBColumns.UPDATED_LOCATION,
        KBColumns.ADDRESS,
        KBColumns.LATITUDE,
        KBColumns.LONGITUDE,
        KBColumns.EMPLOYMENT,
    ),
    ChildChunkType.CLASSIFICATION: (
        KBColumns.COMPANY,
        KBColumns.PRIMARY_FACILITY_TYPE,
        KBColumns.CLASSIFICATION_METHOD,
        KBColumns.CATEGORY,
        KBColumns.EV_BATTERY_RELEVANT,
    ),
}


@dataclass(frozen=True)
class ChildChunk:
    chunk_id: str
    parent_record_id: str
    chunk_type: ChildChunkType
    source_type: str
    embedding_text: str
    metadata: dict

    def payload(self) -> dict[str, Any]:
        """Return lightweight Qdrant payload — no full parent data."""
        return {
            "chunk_id": self.chunk_id,
            "parent_record_id": self.parent_record_id,
            "chunk_type": self.chunk_type.value,
            "source_type": self.source_type,
            **self.metadata,
        }


def build_embedding_text(
    row: pd.Series,
    chunk_type: ChildChunkType,
    fields: tuple[str, ...],
) -> str:
    """Build embedding text from normalized KB fields for one chunk type."""
    lines = [f"chunk_type: {chunk_type.value}"]
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        lines.append(f"{field}: {value}")
    return "\n".join(lines)


def build_child_metadata(row: pd.Series, fields: tuple[str, ...]) -> dict[str, Any]:
    """Build lightweight metadata dict containing only the fields for this chunk type."""
    return {field: row.get(field, "Unknown") for field in fields}
