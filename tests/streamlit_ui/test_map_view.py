"""Tests for map_view marker grouping (co-located companies share one pin)."""
from __future__ import annotations

from georgia_ev_intelligence.streamlit_ui.components import map_view


def test_co_located_companies_collapse_to_one_pin():
    records = [
        {"company": "Hitachi Astemo", "latitude": 32.47052, "longitude": -85.89698},
        {"company": "Hitachi Astemo Americas Inc.", "latitude": 32.47052, "longitude": -85.89698},
        {"company": "IMMI", "latitude": 33.7, "longitude": -84.4},
    ]
    groups = map_view._group_by_location(records)

    assert len(groups) == 2
    shared = groups[(32.47052, -85.89698)]
    assert [r["company"] for r in shared] == [
        "Hitachi Astemo",
        "Hitachi Astemo Americas Inc.",
    ]


def test_records_without_coordinates_are_skipped():
    records = [
        {"company": "Good", "latitude": 33.0, "longitude": -84.0},
        {"company": "No coords", "latitude": None, "longitude": None},
        {"company": "NaN coords", "latitude": "nan", "longitude": "nan"},
    ]
    groups = map_view._group_by_location(records)
    assert len(groups) == 1
    assert list(groups.values())[0][0]["company"] == "Good"


def test_multi_company_popup_lists_every_company_with_count():
    group = [
        {"company": "Alpha", "address": "1 Main St"},
        {"company": "Beta", "product_service": "Widgets"},
    ]
    html = map_view._popup_html(group)
    assert "2 companies at this location" in html
    assert "Alpha" in html and "Beta" in html


def test_single_company_popup_has_no_count_header():
    html = map_view._popup_html([{"company": "Solo", "address": "9 Oak Ave"}])
    assert "companies at this location" not in html
    assert "Solo" in html and "9 Oak Ave" in html
