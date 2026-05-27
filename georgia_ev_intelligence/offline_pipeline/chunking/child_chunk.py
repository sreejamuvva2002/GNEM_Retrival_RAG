"""Child chunk construction for offline embedding.

WHY THIS FILE EXISTS:
  Each KB row is split into 5 thematically-focused child chunks so that
  embedding search can match on specific aspects of a company (identity,
  products, OEM relationships, location/employment, classification).
  Embedding a single dense "everything about Acme Corp" text would bury
  specific signals; 5 focused chunks give both BM25 and dense retrieval
  more precise targets.

THE 5 CHILD CHUNK TYPES (ChildChunkType enum):
  IDENTITY            — Company + Category + Industry Group + Location
                        Answers: "which companies are Tier 1?" or "where is X?"
  PRODUCT_ROLE        — Company + EV Supply Chain Role + Product/Service + EV Relevant
                        Answers: "who makes battery packs?" or "EV component suppliers"
  OEM_RELATIONSHIP    — Company + Primary OEMs + Affiliation Type + Category
                        Answers: "who supplies Rivian?" or "Hyundai Kia suppliers"
  LOCATION_EMPLOYMENT — Company + Location + Address + Lat/Long + Employment
                        Answers: "large employers in Savannah" or "Chatham County companies"
  CLASSIFICATION      — Company + Facility Type + Classification Method + EV Relevant
                        Answers: "manufacturing plants" or "directly relevant EV companies"

CHILD CHUNK ID FORMAT:
  "{parent_record_id}_{CHUNK_TYPE_UPPER}"
  e.g. "KB_ROW_0042_abc123_IDENTITY"

EMBEDDING TEXT FORMAT (build_embedding_text):
  "chunk_type: identity\\ncompany: acme corp\\ncategory: Tier 1\\n..."
  Field names are preserved so BM25 can match on "ev_supply_chain_role: Battery Pack"
  rather than just loose tokens.

RELATIONSHIPS:
  Called by: offline_pipeline/chunking/relationship.py
  Stored in: PostgreSQL child_chunks table with pgvector embeddings
  Searched at runtime by: retrieval/bm25_retriever.py, retrieval/dense_pgvector_retriever.py
"""
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
        """Return lightweight child metadata without full parent data."""
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
