"""
Phase 4 — deterministic KB query planning.

The GNEM workbook is structured data. Questions that ask for exact lists,
counts, top-N rankings, product phrase matches, or area set differences should
be handled as KB operations before semantic vector ranking is considered.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from phase4_agent.entity_extractor import Entities


@dataclass(frozen=True)
class KBQueryPlan:
    mode: str = "semantic"
    keywords: list[str] = field(default_factory=list)
    group_by: str | None = None
    limit: int | None = None

    @property
    def deterministic(self) -> bool:
        return self.mode != "semantic"


def _clean_keyword(term: str) -> str:
    return term.strip(" \"'.,:;()[]{}").lower()


def _quoted_terms(question: str) -> list[str]:
    return [
        _clean_keyword(match)
        for match in re.findall(r"'([^']+)'|\"([^\"]+)\"", question)
        for match in match
        if match
    ]


def _keywords_from_entities(entities: Entities) -> list[str]:
    skip = {
        "top", "size", "many", "areas", "area", "whose", "product",
        "products", "service", "services", "description", "descriptions",
        "include", "includes", "including", "they", "signal", "signals",
        "growth", "customer", "base", "growing", "applicable", "currently",
        "international", "battery", "materials", "infrastructure",
        "components", "component", "structural", "thermal", "applicable",
        "such", "now", "chain",
    }
    result: list[str] = []
    for term in entities.product_keywords:
        cleaned = _clean_keyword(term)
        if cleaned and cleaned not in skip and len(cleaned) >= 3:
            result.append(cleaned)
    return result


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = value.lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped


def _product_terms(question: str, entities: Entities) -> list[str]:
    quoted = _quoted_terms(question)
    if quoted:
        return _dedupe(quoted)
    keywords = _keywords_from_entities(entities)
    return _dedupe(keywords)


def build_kb_query_plan(question: str, entities: Entities) -> KBQueryPlan:
    q = question.lower()

    if "highest employment" in q and entities.county:
        return KBQueryPlan(mode="highest_employment", limit=1)

    if entities.is_top_n:
        return KBQueryPlan(mode="top_employment", limit=entities.top_n_limit)

    if entities.is_risk_query:
        return KBQueryPlan(mode="single_supplier_roles")

    if entities.classification_method and "classified" in q:
        return KBQueryPlan(mode="structured_list")

    if entities.ev_role or entities.ev_role_list:
        role_list_intent = any(
            signal in q
            for signal in (
                "classified under",
                "classified as",
                "list every",
                "list all",
                "show all",
                "map all",
            )
        )
        if role_list_intent:
            return KBQueryPlan(mode="role_list")

    if "ev supply chain role" in q and "related to" in q:
        if "wiring harness" in q or "wiring harnesses" in q:
            return KBQueryPlan(mode="role_text_contains", keywords=["wiring harnesses"])

    if "vehicle assembly" in q and ("facilit" in q or "primary oem" in q):
        return KBQueryPlan(mode="role_list", keywords=["Vehicle Assembly"])

    if "oem footprint" in q or "oem supply chain" in q:
        return KBQueryPlan(mode="tier_list")

    if (
        "traditional oem" in q
        and ("ev-native" in q or "ev native" in q or "rivian" in q)
    ) or "dual-platform" in q:
        return KBQueryPlan(mode="dual_oem_capability")

    if (
        "lack battery cell" in q
        or "lacks battery cell" in q
        or "no battery cell" in q
    ) and "general automotive" in q:
        return KBQueryPlan(mode="areas_with_role_without_roles", group_by="county")

    if (
        "manufacturing plant" in q
        and ("no ev-specific" in q or "no ev specific" in q)
    ):
        return KBQueryPlan(mode="areas_facility_without_ev", group_by="city_county")

    if "highest concentration" in q and "materials" in q:
        return KBQueryPlan(mode="area_concentration", group_by="city_county")

    if "chemical manufacturing infrastructure" in q:
        return KBQueryPlan(
            mode="industry_contains",
            keywords=["chemical", "chemicals"],
        )

    if "battery recycling" in q or "second-life battery" in q or "second life battery" in q:
        return KBQueryPlan(
            mode="product_text_contains",
            keywords=["recycler", "recycling", "second-life", "second life"],
        )

    if "battery parts" in q or "enclosure systems" in q:
        return KBQueryPlan(
            mode="ev_product_text_contains",
            keywords=["lithium-ion battery", "battery cells", "battery parts", "battery electrolyte"],
        )

    if "lithium-ion battery materials, cells, or electrolytes" in q:
        return KBQueryPlan(
            mode="ev_product_text_contains",
            keywords=["lithium-ion battery", "battery cells", "battery electrolyte"],
        )

    if (
        "battery materials" in q
        and any(term in q for term in ("anodes", "cathodes", "electrolytes", "copper foil"))
    ):
        return KBQueryPlan(
            mode="ev_product_text_contains",
            keywords=["lithium-ion battery", "battery electrolyte", "copper foil"],
        )

    if any(signal in q for signal in ("research", "development", "prototyping", "innovation-stage")):
        return KBQueryPlan(
            mode="product_contains",
            keywords=["r&d", "research", "development", "prototyping"],
        )

    if "wiring harness" in q or "wiring harnesses" in q:
        return KBQueryPlan(mode="role_or_product_text_contains", keywords=["wiring harnesses"])

    if "dc-to-dc" in q and "capacitor" in q:
        return KBQueryPlan(mode="product_text_contains", keywords=["dc-to-dc", "capacitors"])

    product_intent = any(
        signal in q
        for signal in (
            "product descriptions include",
            "provide",
            "provides",
            "providing",
            "produce",
            "produces",
            "producing",
            "manufacture",
            "manufactures",
            "manufacturing",
        )
    )
    if product_intent:
        if "product descriptions include" in q:
            terms = _product_terms(question, entities)
            if terms:
                return KBQueryPlan(mode="product_text_contains", keywords=terms)
        if "powder coating" in q or "powder coating-related" in q:
            return KBQueryPlan(
                mode="product_contains",
                keywords=["powder coating", "powder coatings"],
            )
        terms = _product_terms(question, entities)
        if terms:
            return KBQueryPlan(mode="product_contains", keywords=terms)

    return KBQueryPlan()
