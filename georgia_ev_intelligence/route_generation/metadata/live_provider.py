"""Live metadata provider — wraps ``schema.build(loader.load())``.

The index is built once per process (``lru_cache``) because ``loader.load()``
reads the KB Excel and prints. Tests should prefer ``FileMetadataProvider`` to
avoid that I/O.
"""
from __future__ import annotations

from functools import lru_cache

from .provider import ColumnMetaView, IndexBackedProvider


def _to_view(col: str, meta) -> ColumnMetaView:
    """Convert a ``shared.data.schema.ColumnMeta`` dataclass to a view model."""
    return ColumnMetaView(
        field=col,
        match_type=meta.match_type,
        is_numeric=meta.is_numeric,
        is_filterable=meta.is_filterable,
        components=list(meta.components),
        unique_values=list(meta.unique_values),
    )


def build_live_index() -> dict[str, ColumnMetaView]:
    """Build the metadata index from the live KB (reads Excel via loader.load())."""
    from georgia_ev_intelligence.shared.data.loader import load
    from georgia_ev_intelligence.shared.data.schema import build

    raw = build(load())
    return {col: _to_view(col, meta) for col, meta in raw.items()}


@lru_cache(maxsize=1)
def _cached_live_index() -> dict[str, ColumnMetaView]:
    return build_live_index()


class LiveMetadataProvider(IndexBackedProvider):
    """Metadata provider backed by the live KB (cached once per process)."""

    def __init__(
        self,
        index: dict[str, ColumnMetaView] | None = None,
        field_aliases: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            index if index is not None else _cached_live_index(),
            field_aliases,
        )
