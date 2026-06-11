"""Dynamic metadata provider interface.

The validator never hardcodes KB values. Instead it asks a ``MetadataProvider``
for the live distinct values / field metadata produced by
``shared.data.schema.build``. Concrete providers:

* ``LiveMetadataProvider`` — wraps ``schema.build(loader.load())`` (default).
* ``FileMetadataProvider`` — reads a snapshot JSON (decoupled / CI / air-gapped).

``DEFAULT_FIELD_ALIASES`` maps *user language* to *real schema columns* — this is
schema-level mapping, NOT hardcoding data values (which stay dynamic).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ColumnMetaView(BaseModel):
    """Provider-facing view of ``shared.data.schema.ColumnMeta``.

    A pydantic mirror so file snapshots round-trip and providers don't leak the
    underlying dataclass.
    """

    field: str
    match_type: str                                  # "exact" | "partial" | "numeric"
    is_numeric: bool = False
    is_filterable: bool = True
    components: list[str] = Field(default_factory=list)
    unique_values: list[str] = Field(default_factory=list)


@runtime_checkable
class MetadataProvider(Protocol):
    """Read-only access to dynamic KB metadata used during value resolution."""

    def get_allowed_fields(self) -> list[str]:
        """Real columns that may be used as filters (is_filterable=True)."""
        ...

    def get_distinct_values(self, field: str) -> list[str]:
        """Canonical distinct values for ``field`` (empty if unknown/numeric)."""
        ...

    def get_field_meta(self, field: str) -> ColumnMetaView | None:
        """Full metadata for ``field`` (None if the column is unknown)."""
        ...

    def resolve_field_alias(self, hint: str) -> str | None:
        """Map a user-language field hint to a real column, or None."""
        ...


# User phrase -> real schema column. Schema-level only; never data values.
DEFAULT_FIELD_ALIASES: dict[str, str] = {
    # category (supplier tiers)
    "tier": "category",
    "tiers": "category",
    "supplier tier": "category",
    "tier level": "category",
    "category": "category",
    # classification method
    "classification": "classification_method",
    "classification method": "classification_method",
    "classified as": "classification_method",
    # facility type
    "facility": "primary_facility_type",
    "facility type": "primary_facility_type",
    "plant": "primary_facility_type",
    "plant type": "primary_facility_type",
    # supply chain role
    "role": "ev_supply_chain_role",
    "supply chain role": "ev_supply_chain_role",
    "ev role": "ev_supply_chain_role",
    # OEMs
    "oem": "primary_oems",
    "oems": "primary_oems",
    "automaker": "primary_oems",
    "automakers": "primary_oems",
    "manufacturer": "primary_oems",
    # industry
    "industry": "industry_group",
    "industry group": "industry_group",
    "sector": "industry_group",
    # ev battery relevance
    "battery relevant": "ev_battery_relevant",
    "ev relevant": "ev_battery_relevant",
    "ev battery": "ev_battery_relevant",
    # product / service
    "product": "product_service",
    "products": "product_service",
    "service": "product_service",
    "services": "product_service",
    "capability": "product_service",
    "capabilities": "product_service",
    # location
    "location": "updated_location",
    "city": "updated_location",
    "county": "updated_location",
    "state": "state",
    "place": "updated_location",
    "where": "updated_location",
    # company
    "company": "company",
    "companies": "company",
    "name": "company",
    "firm": "company",
    # employment
    "employment": "employment",
    "employees": "employment",
    "headcount": "employment",
    "workforce": "employment",
    "size": "employment",
}


class IndexBackedProvider:
    """Shared implementation over an in-memory ``{column: ColumnMetaView}`` index.

    ``LiveMetadataProvider`` and ``FileMetadataProvider`` differ only in how they
    build the index.
    """

    def __init__(
        self,
        index: dict[str, ColumnMetaView],
        field_aliases: dict[str, str] | None = None,
    ) -> None:
        self._index = index
        self._aliases = {**DEFAULT_FIELD_ALIASES, **(field_aliases or {})}

    def get_allowed_fields(self) -> list[str]:
        return [field for field, meta in self._index.items() if meta.is_filterable]

    def get_distinct_values(self, field: str) -> list[str]:
        meta = self._index.get(field)
        return list(meta.unique_values) if meta else []

    def get_field_meta(self, field: str) -> ColumnMetaView | None:
        return self._index.get(field)

    def resolve_field_alias(self, hint: str) -> str | None:
        key = (hint or "").strip().lower()
        if not key:
            return None
        if key in self._index:               # already a real column name
            return key
        return self._aliases.get(key)
