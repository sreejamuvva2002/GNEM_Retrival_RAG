"""Shared fixtures for route_generation tests.

Everything here runs without a live LLM or the KB Excel: the metadata fixture is
a small hand-built index (including the slash value "Tier 2/3"), and the fake LLM
router returns canned ``RawRoute`` objects.
"""
from __future__ import annotations

import pytest

from georgia_ev_intelligence.route_generation.metadata.provider import (
    ColumnMetaView,
    IndexBackedProvider,
)
from georgia_ev_intelligence.route_generation.schemas import RawRoute


def make_fixture_index() -> dict[str, ColumnMetaView]:
    """A compact metadata index mirroring the real KB's tricky cases."""
    return {
        "category": ColumnMetaView(
            field="category", match_type="exact", is_filterable=True,
            unique_values=["OEM", "OEM Footprint", "OEM Supply Chain",
                           "Tier 1", "Tier 1/2", "Tier 2/3"],
        ),
        "state": ColumnMetaView(
            field="state", match_type="exact", is_filterable=True,
            unique_values=["Alabama", "Georgia", "New Jersey", "North Carolina",
                           "South Carolina", "Unknown"],
        ),
        "primary_facility_type": ColumnMetaView(
            field="primary_facility_type", match_type="exact", is_filterable=True,
            unique_values=["Manufacturing Plant", "Engineering/ Manufacturing",
                           "Manufacturing/ OEM operations", "R&D Center"],
        ),
        "ev_battery_relevant": ColumnMetaView(
            field="ev_battery_relevant", match_type="exact", is_filterable=True,
            unique_values=["Indirect", "No", "Unknown", "Yes"],
        ),
        "ev_supply_chain_role": ColumnMetaView(
            field="ev_supply_chain_role", match_type="exact", is_filterable=True,
            unique_values=["Battery Cell", "Charging Infrastructure",
                           "General Automotive", "Power Electronics"],
        ),
        "primary_oems": ColumnMetaView(
            field="primary_oems", match_type="partial", is_filterable=True,
            unique_values=["Hyundai Kia", "Rivian", "Blue Bird", "Hyundai Kia Rivian"],
        ),
        "product_service": ColumnMetaView(
            field="product_service", match_type="partial", is_filterable=True,
            unique_values=["battery packs", "wiring harness", "seats / interior trim"],
        ),
        "updated_location": ColumnMetaView(
            field="updated_location", match_type="partial", is_filterable=True,
            components=["West Point", "Harris County", "LaGrange", "Troup County",
                        "Savannah", "Chatham County"],
            unique_values=["West Point, Harris County", "LaGrange, Troup County",
                           "Savannah, Chatham County"],
        ),
        "company": ColumnMetaView(
            field="company", match_type="partial", is_filterable=True,
            unique_values=["rivian", "hyundai motor group", "sk battery america"],
        ),
        "employment": ColumnMetaView(
            field="employment", match_type="partial", is_numeric=False,
            is_filterable=True, unique_values=[],
        ),
        "classification_method": ColumnMetaView(
            field="classification_method", match_type="exact", is_filterable=False,
            unique_values=["llm", "rule"],
        ),
    }


@pytest.fixture
def fixture_metadata() -> IndexBackedProvider:
    """A fake MetadataProvider over ``make_fixture_index``."""
    return IndexBackedProvider(make_fixture_index())


class FakeLLMRouter:
    """Returns canned RawRoute(s) instead of calling Ollama.

    Pass a single ``response`` reused every call, or a list of ``responses``
    consumed in order (the last is reused once exhausted).
    """

    def __init__(self, response: RawRoute | None = None,
                 responses: list[RawRoute] | None = None) -> None:
        self._response = response
        self._responses = list(responses or [])
        self.calls: list[tuple] = []

    def route(self, question: str, allowed_fields=None, field_summaries: str = "") -> RawRoute:
        self.calls.append(("route", question))
        return self._next()

    def route_with_clarification(self, question, prior, answer,
                                 allowed_fields=None, field_summaries: str = "") -> RawRoute:
        self.calls.append(("clarify", question, answer))
        return self._next()

    def _next(self) -> RawRoute:
        if self._responses:
            nxt = self._responses.pop(0)
            if not self._responses:
                self._responses.append(nxt)   # reuse last
            return nxt
        return self._response


@pytest.fixture
def fake_llm_factory():
    """Factory so tests build a FakeLLMRouter with their own canned routes."""
    return FakeLLMRouter
