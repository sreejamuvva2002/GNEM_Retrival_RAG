"""Child chunk construction for offline embedding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..parent_chunk import ParentRecord


@dataclass(frozen=True)
class ChildChunk:
    chunk_id: str
    parent_record_id: str
    embedding_text: str
    parent: ParentRecord

    def payload(self) -> dict[str, Any]:
        payload = self.parent.payload()
        char_count = len(self.embedding_text)
        payload.update(
            {
                "chunk_id": self.chunk_id,
                "parent_record_id": self.parent_record_id,
                "embedding_text": self.embedding_text,
                "chunk_type": "company_profile",
                "char_count": char_count,
                "token_estimate": round(char_count / 4),
            }
        )
        return _json_safe(payload)


def build_embedding_text(parent: ParentRecord) -> str:
    """Build the child text chunk in the uploaded example format."""
    lines = [f"Company: {parent.company_clean}"]

    if parent.county:
        lines.append(f"Location: {parent.county} County, Georgia")

    if parent.employment is not None:
        lines.append(f"Employment: {parent.employment} employees")

    if parent.industry_name:
        lines.append(f"Industry: {parent.industry_name}")

    if parent.tier_level:
        tier = parent.tier_level
        if parent.tier_confidence:
            tier = f"{tier} ({parent.tier_confidence})"
        lines.append(f"Supply Chain Tier: {tier}")

    if parent.product_service:
        lines.append(f"Products/Services: {parent.product_service}")

    lines.append(
        "OEM Status: OEM in Georgia"
        if parent.oem_ga
        else "OEM Status: Supplier/Service Provider"
    )
    lines.append("Status: Announced" if parent.is_announcement else "Status: Operational")
    return "\n".join(lines)


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            safe[key] = None
        elif isinstance(value, (str, int, float, bool)):
            safe[key] = value
        elif pd.isna(value):
            safe[key] = None
        else:
            safe[key] = str(value)
    return safe
