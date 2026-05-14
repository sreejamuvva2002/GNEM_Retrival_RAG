"""Parent chunk representation and adapter."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ParentRecord:
    record_id: str
    source_row_id: Any
    row_data: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        payload = {
            "record_id": self.record_id,
            "source_row_id": self.source_row_id,
        }
        payload.update(self.row_data)

        return _json_safe(payload)


class ParentChunkAdapter:
    """Adapt raw DataFrame rows into parent KB records."""

    def build(self, row: pd.Series, row_index: int) -> ParentRecord:
        return build_parent_record(row=row, row_index=row_index)


def build_parent_record(row: pd.Series, row_index: int) -> ParentRecord:
    """
    Build parent record from the raw KB row.

    No normalization is applied to any KB value.
    The complete Excel row is preserved as the parent record.
    """
    row_data = row.to_dict()

    raw_source_row_id = row_data.get("_row_id", None)
    source_row_id = raw_source_row_id if not _is_missing(raw_source_row_id) else row_index

    record_id = _record_id(
        row_index=row_index,
        source_row_id=source_row_id,
    )

    return ParentRecord(
        record_id=record_id,
        source_row_id=source_row_id,
        row_data=row_data,
    )


def _record_id(row_index: int, source_row_id: Any) -> str:
    """
    Create stable parent record ID.

    The ID is based on source_row_id and row_index, not the full row content.
    This prevents the parent ID from changing when a cell value is updated.

    This is ID generation only, not KB value normalization.
    """
    raw_basis = repr(
        {
            "row_index": row_index,
            "source_row_id": source_row_id,
        }
    )

    digest = hashlib.md5(raw_basis.encode("utf-8")).hexdigest()[:12]
    return f"KB_ROW_{row_index:04d}_{digest}"


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Convert payload values into JSON-safe values for Qdrant/PostgreSQL.

    This is serialization safety, not KB normalization.
    It does not clean, lowercase, strip, or modify KB meaning.
    """
    return {key: _to_json_safe_value(value) for key, value in payload.items()}


def _to_json_safe_value(value: Any) -> Any:
    """
    Convert one value into a JSON-safe Python value.
    """

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

    # Preserve simple lists/tuples if they appear.
    if isinstance(value, (list, tuple)):
        return [_to_json_safe_value(item) for item in value]

    # Preserve dictionaries if they appear.
    if isinstance(value, dict):
        return {
            str(key): _to_json_safe_value(val)
            for key, val in value.items()
        }

    # Convert timestamps/dates/other objects safely.
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