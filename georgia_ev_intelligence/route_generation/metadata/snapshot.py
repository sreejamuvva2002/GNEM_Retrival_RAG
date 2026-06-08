"""Metadata snapshot generator (README §21).

Dumps ``schema.build(loader.load())`` to a JSON snapshot that
``FileMetadataProvider`` can load. Regenerate whenever the KB changes::

    python -m georgia_ev_intelligence.route_generation.metadata.snapshot
"""
from __future__ import annotations

import json
from pathlib import Path

from georgia_ev_intelligence.shared import config

from .live_provider import build_live_index
from .provider import DEFAULT_FIELD_ALIASES


def build_snapshot() -> dict:
    """Build the serializable snapshot dict from the live KB metadata."""
    index = build_live_index()
    fields: dict[str, dict] = {}
    for col, view in index.items():
        data = view.model_dump()
        data.pop("field", None)            # the column name is the key
        fields[col] = data
    return {"fields": fields, "field_aliases": dict(DEFAULT_FIELD_ALIASES)}


def dump_snapshot(path: str | Path | None = None) -> Path:
    """Write the snapshot JSON and return the path."""
    target = Path(path or config.METADATA_SNAPSHOT_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(build_snapshot(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


def main() -> None:
    path = dump_snapshot()
    print(f"Wrote metadata snapshot: {path}")


if __name__ == "__main__":
    main()
