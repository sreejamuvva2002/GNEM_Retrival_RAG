"""Tests for the router prompt contract."""
from georgia_ev_intelligence.route_generation.schemas import RawRoute
from georgia_ev_intelligence.route_generation.routing.field_mapping import SCHEMA_FIELDS
from georgia_ev_intelligence.route_generation.routing.prompts import SYSTEM_PROMPT


def test_system_prompt_contains_detailed_routing_policy() -> None:
    assert "Available structured field meanings:" in SYSTEM_PROMPT
    assert "Do NOT use exact_lookup for multi-record questions" in SYSTEM_PROMPT
    assert "Use structured_sql for counts, lists, rankings" in SYSTEM_PROMPT
    assert "Do NOT choose clarification_needed just because:" in SYSTEM_PROMPT
    assert '"confidence": 0.0-1.0' in SYSTEM_PROMPT
    assert '"source_text": "exact phrase from user or null"' in SYSTEM_PROMPT


def test_system_prompt_lists_every_schema_field_meaning() -> None:
    # The field-meaning block is built from the registry — single source of truth.
    for field, meaning in SCHEMA_FIELDS.items():
        assert f"- {field}: {meaning}" in SYSTEM_PROMPT


def test_system_prompt_contains_no_kb_data_values() -> None:
    # Schema meanings only — never concrete KB values the executor must resolve.
    lowered = SYSTEM_PROMPT.lower()
    for kb_value in ("tier 1", "tier 1/2", "oem footprint", "battery cell",
                     "thermal management", "hyundai", "rivian"):
        assert kb_value not in lowered


def test_raw_route_accepts_null_reason_from_small_json_models() -> None:
    route = RawRoute.model_validate({"route": "structured_sql", "reason": None})

    assert route.reason == ""
