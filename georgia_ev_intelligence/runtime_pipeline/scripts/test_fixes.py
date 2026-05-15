"""Quick unit test for operation_detector, term_matcher, and keyword_resolver."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from georgia_ev_intelligence.runtime_pipeline.query.operation_detector import (
    detect_operation,
    is_analytical_phrase,
)
from georgia_ev_intelligence.runtime_pipeline.query.term_matcher import (
    _is_tier_compatible_column,
    _extract_slash_phrases,
    _resolve_slash_conflicts,
    _normalise_for_comparison,
    _extract_question_ngrams,
    find_best_live_value_matches,
)
from georgia_ev_intelligence.runtime_pipeline.query.keyword_resolver import (
    resolve_keywords,
    KeywordResolution,
    _is_column_name,
    _classify_phrase_type,
    _is_column_compatible,
)
from georgia_ev_intelligence.shared.data.schema import ColumnMeta

print("=" * 60)
print("UNIT TESTS: operation_detector + term_matcher + keyword_resolver")
print("=" * 60)

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}: {detail}")


# ── Operation detector tests ──

print("\n--- detect_operation ---")

r = detect_operation("Which roles are single point of failure?")
check("SPOF detected", r["type"] == "spof", f"got {r['type']}")
check("SPOF exhaustive", r["requires_exhaustive_retrieval"])

r = detect_operation("Show all Tier 1/2 suppliers in Georgia")
check("Exhaustive list detected", r["type"] == "exhaustive_list", f"got {r['type']}")

r = detect_operation("Which county has the highest total employment?")
check("Aggregate sum detected", r["type"] == "aggregate_sum", f"got {r['type']}")

r = detect_operation("How many Battery Cell companies are there?")
check("Count detected", r["type"] == "count", f"got {r['type']}")

r = detect_operation("Thermal Management suppliers")
check("No operation for product query", r["type"] == "none", f"got {r['type']}")

print("\n--- is_analytical_phrase ---")
check("SPOF is analytical", is_analytical_phrase("single point of failure"))
check("highest is analytical", is_analytical_phrase("highest"))
check("Battery Cell is NOT analytical", not is_analytical_phrase("Battery Cell"))

# ── Tier column compatibility ──

print("\n--- _is_tier_compatible_column ---")
check("category is tier-compatible", _is_tier_compatible_column("category"))
check("ev_supply_chain_role is NOT tier-compatible", not _is_tier_compatible_column("ev_supply_chain_role"))
check("product_service is NOT tier-compatible", not _is_tier_compatible_column("product_service"))

# ── Slash conflict resolution ──

print("\n--- _resolve_slash_conflicts ---")

mock_schema = {
    "category": ColumnMeta(
        unique_values=["Tier 1/2", "Tier 1", "Tier 2", "Tier 2/3"],
        match_type="exact", is_numeric=False, is_filterable=True,
    )
}
mock_found = ["Tier 1/2", "Tier 1", "Tier 2", "Tier 2/3"]
mock_match_types = {
    "Tier 1/2": "exact", "Tier 1": "tier_exact",
    "Tier 2": "tier_exact", "Tier 2/3": "tier_exact",
}
resolved = _resolve_slash_conflicts(
    mock_found, mock_match_types, "category", mock_schema,
    "Show all Tier 1/2 suppliers",
)
check("Keeps anchor 'Tier 1/2'", "Tier 1/2" in resolved, f"got {resolved}")
check("Removes expanded 'Tier 1'", "Tier 1" not in resolved, f"got {resolved}")
check("Removes expanded 'Tier 2'", "Tier 2" not in resolved, f"got {resolved}")
check("Removes expanded 'Tier 2/3'", "Tier 2/3" not in resolved, f"got {resolved}")
check("Only anchor remains", resolved == ["Tier 1/2"], f"got {resolved}")


# ── Keyword resolver tests ──

print("\n--- keyword_resolver ---")

mock_schema_full = {
    "category": ColumnMeta(
        unique_values=["Tier 1/2", "Tier 1", "Tier 2", "Tier 2/3", "OEM"],
        match_type="exact", is_numeric=False, is_filterable=True,
    ),
    "ev_supply_chain_role": ColumnMeta(
        unique_values=["Battery Cell", "Battery Pack", "Thermal Management",
                        "Tier 1 Automotive Components"],
        match_type="exact", is_numeric=False, is_filterable=True,
    ),
    "product_service": ColumnMeta(
        unique_values=["Battery Modules", "EV Charging Stations"],
        match_type="exact", is_numeric=False, is_filterable=True,
    ),
    "updated_location": ColumnMeta(
        unique_values=["Atlanta, Fulton County", "Savannah, Chatham County"],
        match_type="partial", is_numeric=False, is_filterable=True,
    ),
    "company": ColumnMeta(
        unique_values=["Acme Corp", "Beta Industries"],
        match_type="exact", is_numeric=False, is_filterable=True,
    ),
    "employment": ColumnMeta(
        unique_values=[], match_type="numeric", is_numeric=True, is_filterable=False,
    ),
}

# Test 1: "Battery Cell" should be perfect keyword
kw = resolve_keywords("Show Battery Cell companies", mock_schema_full)
check("has_perfect for Battery Cell", kw.has_perfect, f"got {kw.has_perfect}")
check(
    "Battery Cell is perfect",
    any(k.value == "Battery Cell" for k in kw.perfect_keywords),
    f"perfect: {[k.value for k in kw.perfect_keywords]}"
)
check(
    "Battery Cell deterministic filter",
    "ev_supply_chain_role" in kw.deterministic_filters,
    f"det filters: {kw.deterministic_filters}"
)

# Test 2: "Tier 1/2" should resolve to exact match, not expand
kw2 = resolve_keywords("Show all Tier 1/2 suppliers", mock_schema_full)
check("has_perfect for Tier 1/2", kw2.has_perfect, f"got {kw2.has_perfect}")
perfect_values = [k.value for k in kw2.perfect_keywords]
check(
    "Tier 1/2 is perfect",
    "Tier 1/2" in perfect_values,
    f"perfect: {perfect_values}"
)
# Tier 1, Tier 2, Tier 2/3 should be candidates (suppressed by anchor)
candidate_values = [k.value for k in kw2.candidate_keywords]
check(
    "Tier 1 is candidate (not perfect)",
    "Tier 1" not in perfect_values,
    f"perfect: {perfect_values}"
)

# Test 3: "Thermal Management" should be perfect in ev_supply_chain_role
kw3 = resolve_keywords("Thermal Management suppliers", mock_schema_full)
check("has_perfect for Thermal Mgmt", kw3.has_perfect)
check(
    "Thermal Mgmt in ev_supply_chain_role",
    any(k.value == "Thermal Management" and k.column == "ev_supply_chain_role"
        for k in kw3.perfect_keywords),
    f"perfect: {[(k.value, k.column) for k in kw3.perfect_keywords]}"
)

# Test 4: Column name should be rejected
kw4 = resolve_keywords("Show EV Supply Chain Role data", mock_schema_full)
# "ev_supply_chain_role" is a column name, should not be a keyword
# (but the exact phrase "EV Supply Chain Role" might not appear as a unique value,
# so it may not even be scanned. The column name check catches it if the model tries.)
# Let's test directly:
check("is_column_name detects column name",
      _is_column_name("category", mock_schema_full))
check("is_column_name detects ev_supply_chain_role",
      _is_column_name("ev_supply_chain_role", mock_schema_full))
check("is_column_name rejects Battery Cell",
      not _is_column_name("Battery Cell", mock_schema_full))

# Test 5: Analytical phrases should be rejected
check("classify_phrase_type: 'highest' is analytical",
      _classify_phrase_type("highest") == "analytical")
check("classify_phrase_type: 'Battery Cell' is product_component",
      _classify_phrase_type("Battery Cell") == "product_component")
check("classify_phrase_type: 'tier 1/2' is tier",
      _classify_phrase_type("tier 1/2") == "tier")

# Test 6: Column compatibility
check("tier compatible with category", _is_column_compatible("tier", "category"))
check("tier NOT compatible with ev_supply_chain_role",
      not _is_column_compatible("tier", "ev_supply_chain_role"))
check("product_component compatible with ev_supply_chain_role",
      _is_column_compatible("product_component", "ev_supply_chain_role"))
check("product_component NOT compatible with category",
      not _is_column_compatible("product_component", "category"))
check("analytical never compatible",
      not _is_column_compatible("analytical", "category"))
check("general compatible with anything",
      _is_column_compatible("general", "category"))

# Test 7: No direct match → candidates only, not perfect
kw5 = resolve_keywords("Show all companies doing power electronics", mock_schema_full)
check(
    "No direct match for 'power electronics'",
    not any(k.value.lower() == "power electronics" for k in kw5.perfect_keywords),
    f"perfect: {[k.value for k in kw5.perfect_keywords]}"
)

# Test 8: to_debug_dict works
debug = kw2.to_debug_dict()
check("to_debug_dict has perfect", "perfect" in debug)
check("to_debug_dict has candidate", "candidate" in debug)
check("to_debug_dict has rejected", "rejected" in debug)
check("to_debug_dict has deterministic_filters", "deterministic_filters" in debug)

print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
