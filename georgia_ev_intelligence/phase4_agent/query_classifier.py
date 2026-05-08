"""
Phase 4 — Deterministic question classifier.

Maps a (question, Entities) pair to one of ten QueryClass values. The
classification drives:
  - which retrievers run (sql / dense / sparse / cypher / synonym)
  - whether the cross-encoder reranker fires
  - the per-class evidence-selection policy
  - the override rules in retrieval_fusion (SQL is sole source of truth for
    aggregate / count / rank / top_n / risk)

Rules below are intentionally rule-based, not LLM-based. The entity
extractor has already pulled all the explicit fields (tier, county, OEM,
role, ...) by matching against live KB values; this classifier only
inspects the residual question shape ("how many", "which counties", etc.).

POLICY
------
This file is allowed to contain GENERIC ENGLISH question-shape phrases
(e.g. "how many", "highest total", "linked to", "manufacture") and
references to KB schema field names ("employment", "county"). It is NOT
allowed to contain real KB values (company names, OEM names, county
names, tier values, EV roles, facility types, product terms). Any
domain phrase that should drive retrieval routing belongs in
gev_domain_mapping_rules, not here. The grep audit in
tests/test_07_no_hardcoded_facts.py enforces this.
"""
from __future__ import annotations

import re

from phase4_agent.entity_extractor import Entities
from phase4_agent.retrieval_types import QueryClass


# Generic English question-shape phrases (no domain values). KB schema field
# names like "employment" and "county" are allowed because they are part of
# the approved grammar (see shared/metadata_schema.py).
_COUNT_PHRASES: tuple[str, ...] = (
    "how many",
    "count of",
    "number of companies",
    "number of suppliers",
)

_AGGREGATE_PHRASES: tuple[str, ...] = (
    "highest total",
    "total employment",
    "average employment",
    "median employment",
    "ranked by",
    "rank by",
    "which county has",
    "which counties",
    "highest concentration",
    "highest sum",
)

_NETWORK_PHRASES: tuple[str, ...] = (
    "linked to",
    "supplies to",
    "supply to",
    "downstream of",
    "upstream of",
    "connected to",
    "serving",
    "serves",
    "customer of",
    "customers of",
)

_PRODUCT_CAPABILITY_PHRASES: tuple[str, ...] = (
    "capable of producing",
    "capable of manufacturing",
    "produce ",
    "produces ",
    "producing ",
    "manufacture ",
    "manufactures ",
    "manufacturing ",
    "provide ",
    "provides ",
    "providing ",
    "specializ",
    "expertise in",
    "products include",
    "product descriptions include",
)

# Generic risk-shape phrases (single-point-of-failure language). These are
# generic risk-analysis terminology, not specific to any one question or
# domain entity, so they remain in code rather than the rule store.
_RISK_PHRASES: tuple[str, ...] = (
    "single-point",
    "single point",
    "single-point of failure",
    "single point of failure",
    "single-point-of-failure",
    "served by only",
    "only one company",
    "only a single",
)


def _has_any(text_lower: str, phrases: tuple[str, ...]) -> bool:
    return any(p in text_lower for p in phrases)


def _is_count_query(q_lower: str) -> bool:
    if _has_any(q_lower, _COUNT_PHRASES):
        return True
    return bool(re.match(r"^\s*how many\b", q_lower))


def _residual_abstract_terms(entities: Entities) -> list[str]:
    """
    Read residual abstract terms from the extracted entities. The list is
    populated by entity_extractor (or, after refactor, by a generic noun-
    phrase pass) — this classifier never invents the list itself.
    """
    return list(getattr(entities, "residual_abstract_terms", []) or [])


def classify(question: str, entities: Entities) -> QueryClass:
    """
    Classify a question into one of ten QueryClass values.

    Order matters: the most specific signals win.

      1. risk-shape phrasing or is_risk_query   → RISK
      2. is_top_n                               → TOP_N
      3. residual abstract terms present        → AMBIGUOUS_SEMANTIC
      4. count phrasing                         → COUNT
      5. aggregate phrasing or is_aggregate     → AGGREGATE
      6. ranking phrasing without explicit N    → RANK
      7. OEM entity + network phrasing          → NETWORK
      8. product/capability phrasing            → PRODUCT_CAPABILITY
      9. any structured filter on entities      → EXACT_FILTER
      10. otherwise                              → FALLBACK_SEMANTIC
    """
    q_lower = question.lower()

    if getattr(entities, "is_risk_query", False) or _has_any(q_lower, _RISK_PHRASES):
        return QueryClass.RISK

    if getattr(entities, "is_top_n", False):
        return QueryClass.TOP_N

    # Ambiguity detection runs BEFORE structural classification so a question
    # mixing an abstract phrase with an OEM (e.g. "small-scale <oem> suppliers")
    # still surfaces both interpretations rather than being silently coerced
    # into EXACT_FILTER on the OEM alone.
    if _residual_abstract_terms(entities):
        return QueryClass.AMBIGUOUS_SEMANTIC

    if _is_count_query(q_lower):
        return QueryClass.COUNT

    if getattr(entities, "is_aggregate", False) or _has_any(q_lower, _AGGREGATE_PHRASES):
        return QueryClass.AGGREGATE

    if "rank" in q_lower or "ranking" in q_lower:
        return QueryClass.RANK

    if (entities.oem or entities.oem_list) and _has_any(q_lower, _NETWORK_PHRASES):
        return QueryClass.NETWORK

    if _has_any(q_lower, _PRODUCT_CAPABILITY_PHRASES) or entities.product_keywords:
        return QueryClass.PRODUCT_CAPABILITY

    has_structured = bool(
        entities.tier
        or entities.tier_list
        or entities.county
        or entities.oem
        or entities.oem_list
        or entities.ev_role
        or entities.ev_role_list
        or entities.industry_group
        or entities.facility_type
        or entities.classification_method
        or entities.supplier_affiliation_type
        or entities.min_employment is not None
        or entities.max_employment is not None
        or entities.ev_relevant_filter
        or getattr(entities, "ev_relevance_value", None)
        or entities.company_name
    )
    if has_structured:
        return QueryClass.EXACT_FILTER

    return QueryClass.FALLBACK_SEMANTIC
