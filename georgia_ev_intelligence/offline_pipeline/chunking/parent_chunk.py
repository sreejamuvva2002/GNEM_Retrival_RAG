"""Parent KB record construction for offline chunking."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from georgia_ev_intelligence.shared.data.loader import KBColumns


@dataclass(frozen=True)
class ParentRecord:
    record_id: str
    source_row_id: int
    source_type: str
    # Structured filtering columns
    company: str
    category: str
    industry_group: str
    updated_location: str
    address: str
    latitude: Any
    longitude: Any
    primary_facility_type: str
    ev_supply_chain_role: str
    primary_oems: str
    supplier_or_affiliation_type: str
    employment: Any
    product_service: str
    ev_battery_relevant: str
    classification_method: str
    row_id: int
    # Storage fields
    raw_row: dict
    parent_chunk_text: str

    def payload(self) -> dict[str, Any]:
        """Return full parent payload for PostgreSQL storage."""
        return {
            "record_id": self.record_id,
            "source_row_id": self.source_row_id,
            "source_type": self.source_type,
            "company": self.company,
            "category": self.category,
            "industry_group": self.industry_group,
            "updated_location": self.updated_location,
            "address": self.address,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "primary_facility_type": self.primary_facility_type,
            "ev_supply_chain_role": self.ev_supply_chain_role,
            "primary_oems": self.primary_oems,
            "supplier_or_affiliation_type": self.supplier_or_affiliation_type,
            "employment": self.employment,
            "product_service": self.product_service,
            "ev_battery_relevant": self.ev_battery_relevant,
            "classification_method": self.classification_method,
            "row_id": self.row_id,
            "raw_row": self.raw_row,
            "parent_chunk_text": self.parent_chunk_text,
        }


def build_parent_record(row: pd.Series) -> ParentRecord:
    """Build a ParentRecord from a normalized DataFrame row."""
    source_row_id = _source_row_id(row)
    if source_row_id is None:
        source_row_id = int(row.name) if row.name is not None else 0

    row_id = int(row.get(KBColumns.ROW_ID, source_row_id))

    record_id = _record_id(source_row_id, [
        row.get(KBColumns.COMPANY),
        row.get(KBColumns.UPDATED_LOCATION),
        row.get(KBColumns.PRODUCT_SERVICE),
        row.get(KBColumns.CATEGORY),
        row.get(KBColumns.PRIMARY_FACILITY_TYPE),
        source_row_id,
    ])

    raw_row = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
    parent_chunk_text = build_parent_chunk_text(record_id, source_row_id, row)

    return ParentRecord(
        record_id=record_id,
        source_row_id=source_row_id,
        source_type="excel",
        company=str(row.get(KBColumns.COMPANY, "Unknown")),
        category=str(row.get(KBColumns.CATEGORY, "Unknown")),
        industry_group=str(row.get(KBColumns.INDUSTRY_GROUP, "Unknown")),
        updated_location=str(row.get(KBColumns.UPDATED_LOCATION, "Unknown")),
        address=str(row.get(KBColumns.ADDRESS, "Unknown")),
        latitude=row.get(KBColumns.LATITUDE),
        longitude=row.get(KBColumns.LONGITUDE),
        primary_facility_type=str(row.get(KBColumns.PRIMARY_FACILITY_TYPE, "Unknown")),
        ev_supply_chain_role=str(row.get(KBColumns.EV_SUPPLY_CHAIN_ROLE, "Unknown")),
        primary_oems=str(row.get(KBColumns.PRIMARY_OEMS, "Unknown")),
        supplier_or_affiliation_type=str(row.get(KBColumns.SUPPLIER_OR_AFFILIATION_TYPE, "Unknown")),
        employment=row.get(KBColumns.EMPLOYMENT),
        product_service=str(row.get(KBColumns.PRODUCT_SERVICE, "Unknown")),
        ev_battery_relevant=str(row.get(KBColumns.EV_BATTERY_RELEVANT, "Unknown")),
        classification_method=str(row.get(KBColumns.CLASSIFICATION_METHOD, "Unknown")),
        row_id=row_id,
        raw_row=raw_row,
        parent_chunk_text=parent_chunk_text,
    )


def build_parent_chunk_text(record_id: str, source_row_id: int, row: pd.Series) -> str:
    """Build formatted text representation of the parent record for LLM context."""
    def val(col: str) -> str:
        v = row.get(col, "Unknown")
        return str(v) if v is not None else "Unknown"

    lines = [
        f"Record ID: {record_id}",
        f"Source Row ID: {source_row_id}",
        "",
        f"Company: {val(KBColumns.COMPANY)}",
        f"Category: {val(KBColumns.CATEGORY)}",
        f"Industry Group: {val(KBColumns.INDUSTRY_GROUP)}",
        f"Updated Location: {val(KBColumns.UPDATED_LOCATION)}",
        f"Address: {val(KBColumns.ADDRESS)}",
        f"Latitude: {val(KBColumns.LATITUDE)}",
        f"Longitude: {val(KBColumns.LONGITUDE)}",
        f"Primary Facility Type: {val(KBColumns.PRIMARY_FACILITY_TYPE)}",
        f"EV Supply Chain Role: {val(KBColumns.EV_SUPPLY_CHAIN_ROLE)}",
        f"Primary OEMs: {val(KBColumns.PRIMARY_OEMS)}",
        f"Supplier or Affiliation Type: {val(KBColumns.SUPPLIER_OR_AFFILIATION_TYPE)}",
        f"Employment: {val(KBColumns.EMPLOYMENT)}",
        f"Product / Service: {val(KBColumns.PRODUCT_SERVICE)}",
        f"EV / Battery Relevant: {val(KBColumns.EV_BATTERY_RELEVANT)}",
        f"Classification Method: {val(KBColumns.CLASSIFICATION_METHOD)}",
        f"Original Row ID: {source_row_id}",
    ]
    return "\n".join(lines)


def _source_row_id(row: pd.Series) -> int | None:
    value = row.get(KBColumns.ROW_ID)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _record_id(source_row_id: int, parts: list[Any]) -> str:
    basis = "|".join("" if part is None else str(part) for part in parts)
    md5_hash = hashlib.md5(basis.encode("utf-8")).hexdigest()[:12]
    return f"KB_ROW_{source_row_id:04d}_{md5_hash}"
