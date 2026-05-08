"""
Static-analysis tests that enforce the KB-only / no-hardcoded-facts policy
introduced by the Phase 4 refactor.

These tests are intentionally pure-Python (no DB, no network) so they run
on every CI invocation. They scan source files under phase4_agent/ for
forbidden tokens and patterns. The policy is documented in:
  - shared/metadata_schema.py  (approved hardcoded grammar)
  - shared/db.py               (DomainMappingRule write policy)

If you need to introduce a new domain-flavoured token to phase4_agent/,
either:
  (a) add an approved row to gev_domain_mapping_rules and reference it
      via synonym_expander, or
  (b) widen the approved grammar in shared/metadata_schema.py and update
      the test allowlists below.

Do NOT relax these tests to make a failing build green.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PHASE4 = ROOT / "phase4_agent"


# ─────────────────────────────────────────────────────────────────────────────
# H.1.a — no real KB values appear as literals in phase4_agent/
# ─────────────────────────────────────────────────────────────────────────────

# Real OEM / company / county / city / role / tier / facility values that
# must come from the live KB or from gev_domain_mapping_rules. This list is
# representative, not exhaustive — the goal is to catch obvious regressions,
# not to enumerate every possible KB value.
_FORBIDDEN_PATTERNS: tuple[str, ...] = (
    # OEM brand names
    r"\bHyundai\b",
    r"\bRivian\b",
    r"\bKia\b",
    r"\bBMW\b",
    r"\bMercedes-Benz\b",
    r"\bBlue Bird\b",
    r"\bSK Battery\b",
    # County names (sample)
    r"\bTroup County\b",
    r"\bGwinnett County\b",
    r"\bHall County\b",
    r"\bBryan County\b",
    # City names (sample)
    r"\bLaGrange\b",
    r"\bCartersville\b",
    # Real role / tier / facility values quoted as Python string literals
    r'"Battery Cell"',
    r'"Battery Pack"',
    r'"Thermal Management"',
    r'"Charging Infrastructure"',
    r'"General Automotive"',
    r'"Manufacturing Plant"',
    r'"Materials"',
    r'"Tier 1"',
    r'"Tier 1/2"',
    r'"Tier 2/3"',
    r'"OEM \(Footprint\)"',
    # Real company names
    r'"Hyundai Motor Group"',
    r'"Novelis"',
    r'"SK Battery America"',
)


def _python_files() -> list[Path]:
    return sorted(p for p in PHASE4.rglob("*.py") if p.is_file())


def test_no_hardcoded_facts_in_phase4():
    offenders: list[str] = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        for pat in _FORBIDDEN_PATTERNS:
            for m in re.finditer(pat, text):
                line = text.count("\n", 0, m.start()) + 1
                offenders.append(
                    f"{path.relative_to(ROOT)}:{line} matched {pat!r}"
                )
    assert not offenders, (
        "Forbidden hardcoded KB values found in phase4_agent/.\n"
        "These belong in gev_domain_mapping_rules or must be sampled at "
        "runtime via metadata_loader. See shared/metadata_schema.py for the "
        "approved hardcoding policy.\n\n"
        + "\n".join(offenders)
    )


# ─────────────────────────────────────────────────────────────────────────────
# H.1.b — phase4_agent/ never WRITES to gev_domain_mapping_rules
# ─────────────────────────────────────────────────────────────────────────────

_WRITE_PATTERN = re.compile(
    r"(?:session\.add|session\.delete|\.update\(\s*DomainMappingRule|"
    r"DomainMappingRule\s*\(.*?\)\s*\)|"
    r"DomainMappingRule\s*\(.*?\).*?session\.add)",
    flags=re.DOTALL,
)


def test_phase4_does_not_write_domain_mapping_rules():
    offenders: list[str] = []
    for path in _python_files():
        src = path.read_text(encoding="utf-8")
        if "DomainMappingRule" not in src:
            continue
        # Allow read-only patterns: queries are fine, INSERT/UPDATE/DELETE
        # are not. Look for "session.add(<expr containing DomainMappingRule>)".
        for m in re.finditer(r"session\.(add|delete|merge)\s*\(", src):
            after = src[m.end(): m.end() + 240]
            if "DomainMappingRule" in after:
                line = src.count("\n", 0, m.start()) + 1
                offenders.append(
                    f"{path.relative_to(ROOT)}:{line} writes DomainMappingRule"
                )
    assert not offenders, (
        "phase4_agent must never INSERT/UPDATE/DELETE rows in "
        "gev_domain_mapping_rules. Use scripts/approve_mapping_rule.py.\n\n"
        + "\n".join(offenders)
    )


# ─────────────────────────────────────────────────────────────────────────────
# H.1.c — no imports from evaluate.* in phase4_agent/
# ─────────────────────────────────────────────────────────────────────────────

def test_no_evaluate_imports_in_phase4():
    offenders: list[str] = []
    for path in _python_files():
        src = path.read_text(encoding="utf-8")
        for m in re.finditer(r"^\s*(?:from\s+evaluate|import\s+evaluate)\b", src, re.MULTILINE):
            line = src.count("\n", 0, m.start()) + 1
            offenders.append(f"{path.relative_to(ROOT)}:{line} imports evaluate")
    assert not offenders, (
        "phase4_agent must not import from evaluate/. Golden answers are "
        "evaluation-only.\n\n" + "\n".join(offenders)
    )


# ─────────────────────────────────────────────────────────────────────────────
# H.1.d — shared/metadata_schema.py exposes the approved 10 constants
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_CONSTANTS = (
    "CANONICAL_FIELDS",
    "FIELD_TYPES",
    "HARD_FILTER_FIELDS",
    "SUPPORTED_OPERATORS",
    "QUERY_CLASSES",
    "FINAL_SCORE_WEIGHTS",
    "COMPANY_CHUNK_VIEWS",
    "REQUIRED_QDRANT_PAYLOAD",
    "GRAPH_LABELS",
    "GRAPH_RELATIONSHIPS",
)


def test_metadata_schema_is_canonical():
    schema_path = ROOT / "shared" / "metadata_schema.py"
    assert schema_path.exists(), "shared/metadata_schema.py must exist"
    src = schema_path.read_text(encoding="utf-8")
    missing = [c for c in _REQUIRED_CONSTANTS if c not in src]
    assert not missing, (
        f"shared/metadata_schema.py is missing required constants: {missing}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# H.1.e — synonym_expander has no _HEURISTIC_TEMPLATES and only reads approved rules
# ─────────────────────────────────────────────────────────────────────────────

def test_synonym_expander_has_no_heuristic_templates():
    src = (PHASE4 / "synonym_expander.py").read_text(encoding="utf-8")
    assert "_HEURISTIC_TEMPLATES" not in src, (
        "phase4_agent/synonym_expander.py must NOT contain _HEURISTIC_TEMPLATES. "
        "Unapproved auto-mappings violate the 'no feedback = no permanent learning' policy."
    )


def test_synonym_expander_filters_by_approved_status():
    src = (PHASE4 / "synonym_expander.py").read_text(encoding="utf-8")
    # The status filter should reference 'approved' (the schema default
    # 'active' is allowed alongside it for legacy compat).
    assert "approved" in src, (
        "synonym_expander must filter gev_domain_mapping_rules by status='approved'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# H.1.f — kb_query_planner has no phrase-keyed product/role keyword branches
# ─────────────────────────────────────────────────────────────────────────────

_FORBIDDEN_PLANNER_PHRASES = (
    "battery recycling",
    "second-life battery",
    "wiring harness",
    "powder coating",
    "chemical manufacturing",
    "lithium-ion battery",
    "dc-to-dc",
    "anodes",
    "cathodes",
    "copper foil",
    "battery parts",
    "enclosure systems",
    "battery materials",
)


def test_kb_query_planner_has_no_phrase_keyed_branches():
    src = (PHASE4 / "kb_query_planner.py").read_text(encoding="utf-8")
    offenders = [p for p in _FORBIDDEN_PLANNER_PHRASES if p in src.lower()]
    assert not offenders, (
        "phase4_agent/kb_query_planner.py must not branch on hardcoded "
        "product/role phrase wordings. Move routing into "
        "gev_domain_mapping_rules or query_classifier.\n"
        f"Offending phrases: {offenders}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# H.1.g — vector_retriever has no question-specific deterministic answer paths
# ─────────────────────────────────────────────────────────────────────────────

_REMOVED_VR_FUNCTIONS = (
    "_deterministic_dual_oem_capability",
    "_deterministic_areas_with_role_without_roles",
    "_deterministic_area_concentration",
    "_deterministic_areas_facility_without_ev",
)


def test_vector_retriever_has_no_question_specific_paths():
    src = (PHASE4 / "vector_retriever.py").read_text(encoding="utf-8")
    # Allow the removal-comment to reference these names; reject any actual
    # `def <name>(` definition.
    for fn in _REMOVED_VR_FUNCTIONS:
        assert f"def {fn}(" not in src, (
            f"phase4_agent/vector_retriever.py must not define {fn} — "
            "it hardcoded specific OEM / tier / role / facility values."
        )


# ─────────────────────────────────────────────────────────────────────────────
# H.1.h — query_classifier has no hardcoded abstract-phrase hint tuples
# ─────────────────────────────────────────────────────────────────────────────

def test_query_classifier_has_no_abstract_hints_tuple():
    src = (PHASE4 / "query_classifier.py").read_text(encoding="utf-8")
    assert "_ABSTRACT_HINTS" not in src, (
        "phase4_agent/query_classifier.py must not contain _ABSTRACT_HINTS. "
        "Use entities.residual_abstract_terms instead."
    )


# ─────────────────────────────────────────────────────────────────────────────
# H.1.i — entity_extractor has no hardcoded factual lists
# ─────────────────────────────────────────────────────────────────────────────

_REMOVED_EE_NAMES = (
    "_TIER_SYNONYMS",         # was a literal tier-synonym dict
    "_ABSTRACT_PHRASES",      # was a literal abstract-phrase tuple
    "is_oem_dependency",      # phrase-keyed risk subtype
    "is_capacity_risk",       # phrase-keyed risk subtype
    "is_misalignment",        # phrase-keyed risk subtype
)


def test_entity_extractor_has_no_phrase_keyed_lists():
    src = (PHASE4 / "entity_extractor.py").read_text(encoding="utf-8")
    for name in _REMOVED_EE_NAMES:
        assert name not in src, (
            f"phase4_agent/entity_extractor.py must not contain {name!r} — "
            "it was a phrase-keyed routing list. The new path uses "
            "gev_domain_mapping_rules + generic noun-phrase extraction."
        )


# ─────────────────────────────────────────────────────────────────────────────
# H.1.j — text_to_sql / text_to_cypher have no literal KB values in prompts
# ─────────────────────────────────────────────────────────────────────────────

def test_text_to_sql_prompts_have_no_literal_kb_values():
    src = (PHASE4 / "text_to_sql.py").read_text(encoding="utf-8")
    for pat in _FORBIDDEN_PATTERNS:
        assert not re.search(pat, src), (
            f"phase4_agent/text_to_sql.py must not contain {pat!r}. "
            "Examples must be sampled at runtime via metadata_loader."
        )


def test_text_to_cypher_prompts_have_no_literal_kb_values():
    src = (PHASE4 / "text_to_cypher.py").read_text(encoding="utf-8")
    for pat in _FORBIDDEN_PATTERNS:
        assert not re.search(pat, src), (
            f"phase4_agent/text_to_cypher.py must not contain {pat!r}. "
            "Examples must be sampled at runtime via metadata_loader."
        )


# ─────────────────────────────────────────────────────────────────────────────
# H.1.k — metadata_loader.py exists and exposes the documented surface
# ─────────────────────────────────────────────────────────────────────────────

def test_metadata_loader_module_exists():
    path = PHASE4 / "metadata_loader.py"
    assert path.exists(), "phase4_agent/metadata_loader.py must exist"
    src = path.read_text(encoding="utf-8")
    for sym in ("class MetadataLoader", "def distinct", "def sample_distinct_values",
                "def kb_columns_supporting", "def invalidate", "loader = MetadataLoader()"):
        assert sym in src, f"metadata_loader.py missing required symbol: {sym}"


# ─────────────────────────────────────────────────────────────────────────────
# H.1.l — formatters defensively assert on validated rows
# ─────────────────────────────────────────────────────────────────────────────

def test_formatters_assert_validated_rows():
    src = (PHASE4 / "formatters.py").read_text(encoding="utf-8")
    assert "_assert_validated" in src, (
        "phase4_agent/formatters.py must call a defensive assertion on "
        "incoming rows so un-validated data cannot reach the answer."
    )
