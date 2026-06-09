"""File metadata provider — reads a snapshot JSON produced by ``snapshot.py``.

Snapshot format::

    {
      "fields": {
        "category": {"match_type": "exact", "is_numeric": false,
                     "is_filterable": true, "components": [],
                     "unique_values": ["Tier 1", "Tier 1/2", "Tier 2/3", ...]},
        ...
      },
      "field_aliases": {"tier": "category", ...}
    }
"""
from __future__ import annotations

import json
from pathlib import Path

from .provider import ColumnMetaView, IndexBackedProvider


def _with_field(col: str, meta: dict) -> dict:
    data = dict(meta)
    data.setdefault("field", col)
    return data


class FileMetadataProvider(IndexBackedProvider):
    """Metadata provider backed by a snapshot JSON file."""

    @classmethod
    def from_path(cls, path: str | Path) -> "FileMetadataProvider":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "FileMetadataProvider":
        fields = data.get("fields", data) if isinstance(data, dict) else {}
        index = {
            col: ColumnMetaView(**_with_field(col, meta))
            for col, meta in fields.items()
        }
        aliases = data.get("field_aliases") if isinstance(data, dict) else None
        return cls(index, aliases)
