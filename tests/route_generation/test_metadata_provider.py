"""Tests for the metadata providers and the snapshot round-trip."""
from __future__ import annotations

import pytest

from georgia_ev_intelligence.route_generation.metadata.file_provider import (
    FileMetadataProvider,
)
from georgia_ev_intelligence.route_generation.metadata.provider import (
    ColumnMetaView,
    MetadataProvider,
)


class TestIndexBackedProvider:
    def test_allowed_fields_exclude_non_filterable(self, fixture_metadata):
        allowed = fixture_metadata.get_allowed_fields()
        assert "category" in allowed
        assert "classification_method" not in allowed  # is_filterable=False

    def test_distinct_values(self, fixture_metadata):
        assert "Tier 2/3" in fixture_metadata.get_distinct_values("category")

    def test_unknown_field_meta_is_none(self, fixture_metadata):
        assert fixture_metadata.get_field_meta("nonexistent") is None
        assert fixture_metadata.get_distinct_values("nonexistent") == []

    def test_alias_resolution(self, fixture_metadata):
        assert fixture_metadata.resolve_field_alias("tier") == "category"
        assert fixture_metadata.resolve_field_alias("OEM") == "primary_oems"
        assert fixture_metadata.resolve_field_alias("state") == "state"
        assert fixture_metadata.resolve_field_alias("category") == "category"  # real column
        assert fixture_metadata.resolve_field_alias("zzz") is None
        assert fixture_metadata.resolve_field_alias("") is None

    def test_satisfies_protocol(self, fixture_metadata):
        assert isinstance(fixture_metadata, MetadataProvider)


class TestFileProviderRoundTrip:
    def test_from_dict_roundtrip(self):
        snapshot = {
            "fields": {
                "category": {
                    "match_type": "exact", "is_numeric": False, "is_filterable": True,
                    "components": [], "unique_values": ["Tier 1", "Tier 1/2", "Tier 2/3"],
                },
                "classification_method": {
                    "match_type": "exact", "is_numeric": False, "is_filterable": False,
                    "components": [], "unique_values": ["llm"],
                },
            },
            "field_aliases": {"tier": "category"},
        }
        provider = FileMetadataProvider.from_dict(snapshot)
        assert provider.get_allowed_fields() == ["category"]
        assert provider.get_distinct_values("category") == ["Tier 1", "Tier 1/2", "Tier 2/3"]
        assert provider.resolve_field_alias("tier") == "category"
        meta = provider.get_field_meta("category")
        assert isinstance(meta, ColumnMetaView) and meta.field == "category"


# The live round-trip reads the real KB Excel; skip cleanly if it is absent.
def _kb_excel_available() -> bool:
    from georgia_ev_intelligence.shared.data import loader
    return loader.KB_EXCEL_PATH.exists()


@pytest.mark.skipif(not _kb_excel_available(), reason="KB Excel not available")
def test_live_snapshot_file_roundtrip(tmp_path):
    from georgia_ev_intelligence.route_generation.metadata import snapshot
    from georgia_ev_intelligence.route_generation.metadata.live_provider import (
        LiveMetadataProvider,
    )

    out = snapshot.dump_snapshot(tmp_path / "snap.json")
    assert out.exists()

    live = LiveMetadataProvider()
    filep = FileMetadataProvider.from_path(out)

    # Same fields and identical distinct values for every field.
    assert sorted(live.get_allowed_fields()) == sorted(filep.get_allowed_fields())
    for field in live.get_allowed_fields():
        assert live.get_distinct_values(field) == filep.get_distinct_values(field)

    # Sanity: the known facts survive the round-trip.
    assert "Tier 2/3" in filep.get_distinct_values("category")
    assert filep.get_field_meta("classification_method").is_filterable is False
