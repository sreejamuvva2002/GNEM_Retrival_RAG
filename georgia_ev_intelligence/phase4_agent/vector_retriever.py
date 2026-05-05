"""
Phase 4 — Qdrant-only retrieval.

All answer context comes from company records stored in the vector DB.
We load company payloads from Qdrant, apply structured filtering in Python,
and use hybrid vector ranking only when we need semantic ordering.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from phase2_embedding.embedder import embed_single
from phase2_embedding.vector_store import scroll_points, search_dense
from phase4_agent.entity_extractor import Entities
from phase4_agent.filter_interpreter import SoftFilterPlan, interpret_soft_filters
from phase4_agent.reranker import rerank_companies
from shared.config import Config
from shared.logger import get_logger

logger = get_logger("phase4.vector_retriever")

_BASE_COMPANY_FILTERS = {
    "chunk_type": "company",
    "source_type": "gnem_excel",
}
_MASTER_COMPANY_FILTERS = {
    **_BASE_COMPANY_FILTERS,
    "chunk_view": "master",
}
_SEMANTIC_TOP_K = 120
_DEFAULT_LIMIT = 18
_BROAD_LIMIT = 30
_SEMANTIC_LIMIT = 12
_SOFT_FILTER_TRIGGER_COUNT = 8


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def _employment(company: dict[str, Any]) -> float:
    raw = company.get("employment")
    if raw in (None, ""):
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _is_oem_reference(tier_value: str | None) -> bool:
    if not tier_value:
        return False
    tier_norm = _normalize(tier_value)
    if tier_norm in {"oem supply chain", "oem footprint", "oem (footprint)"}:
        return False
    return tier_norm == "oem"


def _tier_matches(company_tier: str, requested_tier: str) -> bool:
    company_norm = _normalize(company_tier)
    requested_norm = _normalize(requested_tier)
    if not requested_norm:
        return True
    if company_norm == requested_norm:
        return True
    # Helpful normalization: Tier 2 requests should include Tier 2/3 companies.
    if requested_norm == "tier 2" and company_norm == "tier 2/3":
        return True
    return False


def _entity_tier_filters(entities: Entities) -> list[str]:
    values = []
    for tier in [entities.tier, *getattr(entities, "tier_list", [])]:
        if tier and not _is_oem_reference(tier):
            norm = _normalize(tier)
            if norm and norm not in {_normalize(v) for v in values}:
                values.append(tier)
    return values


def _company_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "company_row_id": payload.get("company_row_id", ""),
        "company_name": payload.get("company_name", ""),
        "tier": payload.get("tier", ""),
        "ev_supply_chain_role": payload.get("ev_supply_chain_role", ""),
        "primary_oems": payload.get("primary_oems", ""),
        "ev_battery_relevant": payload.get("ev_battery_relevant", ""),
        "industry_group": payload.get("industry_group", ""),
        "facility_type": payload.get("facility_type", ""),
        "location_city": payload.get("location_city", ""),
        "location_county": payload.get("location_county", ""),
        "location_state": payload.get("location_state", "Georgia"),
        "employment": payload.get("employment"),
        "products_services": payload.get("products_services", ""),
        "classification_method": payload.get("classification_method", ""),
        "supplier_affiliation_type": payload.get("supplier_affiliation_type", ""),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "chunk_view": payload.get("chunk_view", "legacy"),
        "matched_view": payload.get("chunk_view", "legacy"),
        "text": payload.get("master_text", payload.get("text", "")),
    }


def _row_key(company: dict[str, Any]) -> tuple[Any, ...]:
    if company.get("company_row_id"):
        return (company.get("company_row_id"),)
    return (
        company.get("company_name"),
        company.get("tier"),
        company.get("ev_supply_chain_role"),
        company.get("primary_oems"),
        company.get("ev_battery_relevant"),
        company.get("industry_group"),
        company.get("facility_type"),
        company.get("location_city"),
        company.get("location_county"),
        company.get("location_state"),
        company.get("employment"),
        company.get("products_services"),
        company.get("classification_method"),
        company.get("supplier_affiliation_type"),
        company.get("latitude"),
        company.get("longitude"),
    )


def _dedupe_exact_rows(companies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for company in companies:
        deduped.setdefault(_row_key(company), company)
    return list(deduped.values())


@lru_cache(maxsize=1)
def _load_all_companies() -> list[dict[str, Any]]:
    records = scroll_points(filters=_MASTER_COMPANY_FILTERS, limit=600)
    if not records:
        logger.info("No master-view chunks found; falling back to legacy company chunks")
        records = scroll_points(filters=_BASE_COMPANY_FILTERS, limit=600)
    companies = [_company_from_payload(r["payload"]) for r in records]
    companies = [c for c in companies if c.get("company_name")]
    companies = _dedupe_exact_rows(companies)
    logger.info("Loaded %d company records from Qdrant", len(companies))
    return companies


def _matches_entities(company: dict[str, Any], entities: Entities) -> bool:
    if entities.company_name and _normalize(company.get("company_name")) != _normalize(entities.company_name):
        return False
    if entities.county and _normalize(company.get("location_county")) != _normalize(entities.county):
        return False
    tier_filters = _entity_tier_filters(entities)
    if tier_filters:
        if not any(_tier_matches(company.get("tier", ""), tier) for tier in tier_filters):
            return False
    if entities.industry_group and _normalize(company.get("industry_group")) != _normalize(entities.industry_group):
        return False
    if entities.facility_type and _normalize(entities.facility_type) not in _normalize(company.get("facility_type")):
        return False
    if entities.classification_method and _normalize(company.get("classification_method")) != _normalize(entities.classification_method):
        return False
    if entities.supplier_affiliation_type and _normalize(company.get("supplier_affiliation_type")) != _normalize(entities.supplier_affiliation_type):
        return False
    excluded_roles = {_normalize(role) for role in getattr(entities, "exclude_ev_role_list", [])}
    if excluded_roles and _normalize(company.get("ev_supply_chain_role")) in excluded_roles:
        return False
    if entities.ev_role and _normalize(company.get("ev_supply_chain_role")) != _normalize(entities.ev_role):
        return False
    if entities.ev_role_list:
        role_set = {_normalize(role) for role in entities.ev_role_list}
        if _normalize(company.get("ev_supply_chain_role")) not in role_set:
            return False
    if entities.oem_list:
        oem_text = _normalize(company.get("primary_oems"))
        if not any(_normalize(oem) in oem_text for oem in entities.oem_list):
            return False
    elif entities.oem:
        if _normalize(entities.oem) not in _normalize(company.get("primary_oems")):
            return False
    emp = _employment(company)
    if entities.min_employment is not None and emp < entities.min_employment:
        return False
    if entities.max_employment is not None and emp > entities.max_employment:
        return False
    if entities.ev_relevant_filter and _normalize(company.get("ev_battery_relevant")) != "yes":
        return False
    return True


def _apply_soft_filter_plan(companies: list[dict[str, Any]], plan: SoftFilterPlan) -> list[dict[str, Any]]:
    if not plan.active:
        return companies

    include_roles = {_normalize(value) for value in plan.include_roles}
    exclude_roles = {_normalize(value) for value in plan.exclude_roles}
    include_affiliations = {_normalize(value) for value in plan.include_supplier_affiliation_types}
    include_classifications = {_normalize(value) for value in plan.include_classification_methods}
    include_facilities = {_normalize(value) for value in plan.include_facility_types}
    include_industries = {_normalize(value) for value in plan.include_industry_groups}
    require_ev = _normalize(plan.require_ev_battery_relevant)

    filtered: list[dict[str, Any]] = []
    for company in companies:
        role = _normalize(company.get("ev_supply_chain_role"))
        affiliation = _normalize(company.get("supplier_affiliation_type"))
        classification = _normalize(company.get("classification_method"))
        facility = _normalize(company.get("facility_type"))
        industry = _normalize(company.get("industry_group"))
        ev_relevance = _normalize(company.get("ev_battery_relevant"))

        if require_ev and ev_relevance != require_ev:
            continue
        if include_roles and role not in include_roles:
            continue
        if exclude_roles and role in exclude_roles:
            continue
        if include_affiliations and affiliation not in include_affiliations:
            continue
        if include_classifications and classification not in include_classifications:
            continue
        if include_facilities and facility not in include_facilities:
            continue
        if include_industries and industry not in include_industries:
            continue

        filtered.append(company)

    return filtered


def _needs_soft_interpretation(question: str, entities: Entities, matches: list[dict[str, Any]]) -> bool:
    if len(matches) <= _SOFT_FILTER_TRIGGER_COUNT:
        return False
    if entities.is_aggregate or entities.is_risk_query or entities.is_top_n:
        return False
    q_lower = question.lower()
    ambiguous_signals = (
        "primary",
        "main",
        "specific",
        "focused",
        "relevant",
        "involvement",
        "capable",
        "suitable",
        "ready",
        "innovation",
        "fragility",
        "risk",
        "support",
    )
    return bool(entities.product_keywords) or any(signal in q_lower for signal in ambiguous_signals)


def _apply_rule_based_semantic_filters(question: str, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    q_lower = question.lower()

    primary_signals = (
        "primary involvement",
        "primary role",
        "direct involvement",
        "directly involved",
        "core involvement",
    )
    ev_focus_signals = ("electric vehicle", "ev ", "battery", "electrification")
    scope_signals = ("supply chain", "relevant", "involvement", "role", "roles")

    if (
        any(signal in q_lower for signal in primary_signals)
        and any(signal in q_lower for signal in ev_focus_signals)
        and any(signal in q_lower for signal in scope_signals)
    ):
        yes_matches = [
            company for company in matches
            if _normalize(company.get("ev_battery_relevant")) == "yes"
        ]
        if yes_matches and len(yes_matches) < len(matches):
            logger.info(
                "Rule-based semantic filter narrowed candidates from %d to %d using ev_battery_relevant=Yes",
                len(matches),
                len(yes_matches),
            )
            return yes_matches

    return matches


def _has_structured_filters(entities: Entities) -> bool:
    return any([
        entities.company_name,
        entities.county,
        _entity_tier_filters(entities),
        entities.industry_group,
        entities.facility_type,
        entities.classification_method,
        entities.supplier_affiliation_type,
        entities.ev_role,
        entities.ev_role_list,
        getattr(entities, "exclude_ev_role_list", []),
        entities.oem,
        entities.oem_list,
        entities.min_employment is not None,
        entities.max_employment is not None,
        entities.ev_relevant_filter,
    ])


def _refine_matches(question: str, entities: Entities, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = _apply_rule_based_semantic_filters(question, matches)
    if not _needs_soft_interpretation(question, entities, matches):
        return matches

    plan = interpret_soft_filters(question, entities, matches)
    refined = _apply_soft_filter_plan(matches, plan)
    if refined and len(refined) < len(matches):
        logger.info(
            "Soft interpretation narrowed candidates from %d to %d%s",
            len(matches),
            len(refined),
            f" ({plan.explanation})" if plan.explanation else "",
        )
        return refined
    return matches


def _company_rank_key(company: dict[str, Any]) -> str:
    return str(company.get("company_row_id") or company.get("company_name") or "")


def _semantic_rank(question: str) -> tuple[dict[str, int], list[dict[str, Any]]]:
    vector = embed_single(question)
    results = search_dense(
        query_vector=vector,
        top_k=_SEMANTIC_TOP_K,
        filters=_BASE_COMPANY_FILTERS,
    )

    rank_by_company_id: dict[str, int] = {}
    ranked_companies: list[dict[str, Any]] = []
    seen_rows: set[tuple[Any, ...]] = set()
    for result in results:
        metadata = result.get("metadata", {})
        company = _company_from_payload(metadata)
        row_key = _row_key(company)
        company_rank_key = _company_rank_key(company)
        if not company_rank_key or row_key in seen_rows:
            continue
        seen_rows.add(row_key)
        rank_by_company_id.setdefault(company_rank_key, len(rank_by_company_id))
        ranked_companies.append(company)

    return rank_by_company_id, ranked_companies


def _is_broad_listing(question: str, entities: Entities) -> bool:
    q_lower = question.lower()
    broad_markers = ("show all", "list all", "list every", "map all", "full supplier network")
    return entities.is_aggregate or entities.is_top_n or any(marker in q_lower for marker in broad_markers)


def _limit_for_question(question: str, entities: Entities) -> int:
    if _is_broad_listing(question, entities):
        return _BROAD_LIMIT
    if entities.product_keywords or entities.oem or entities.oem_list:
        return _SEMANTIC_LIMIT
    return _DEFAULT_LIMIT


def _rerank_cap(question: str, entities: Entities) -> int:
    limit = _limit_for_question(question, entities)
    return max(limit, Config.get().reranker_max_candidates)


def _sort_matches(
    question: str,
    entities: Entities,
    matches: list[dict[str, Any]],
    rank_by_company_id: dict[str, int],
) -> list[dict[str, Any]]:
    if entities.is_top_n:
        return sorted(matches, key=lambda c: _employment(c), reverse=True)

    if entities.is_aggregate:
        return matches

    limit = _limit_for_question(question, entities)
    dense_ranked = sorted(
        matches,
        key=lambda c: (
            rank_by_company_id.get(_company_rank_key(c), 10_000),
            c.get("company_name", ""),
        ),
    )
    rerank_cap = _rerank_cap(question, entities)
    preselected = dense_ranked[:rerank_cap]
    reranked = rerank_companies(question, preselected)
    return reranked[:limit]


def _format_aggregate_context(companies: list[dict[str, Any]], label: str) -> str:
    county_totals: dict[str, dict[str, Any]] = {}
    for company in companies:
        county = company.get("location_county") or "Unknown County"
        bucket = county_totals.setdefault(
            county,
            {"county": county, "total_employment": 0.0, "company_count": 0},
        )
        bucket["total_employment"] += _employment(company)
        bucket["company_count"] += 1

    rows = sorted(
        county_totals.values(),
        key=lambda r: (r["total_employment"], r["company_count"]),
        reverse=True,
    )
    if not rows:
        return "No matching companies found."

    lines = [
        f"{row['county']}: {int(row['total_employment']):,} employees ({row['company_count']} companies)"
        for row in rows
    ]
    return f"{label}\n" + "\n".join(lines)


def _format_risk_context(companies: list[dict[str, Any]]) -> str:
    role_index: dict[str, list[dict[str, Any]]] = {}
    for company in companies:
        role = company.get("ev_supply_chain_role") or "Unknown Role"
        role_index.setdefault(role, []).append(company)

    sole_suppliers = []
    for role, role_companies in role_index.items():
        unique = _dedupe_exact_rows(role_companies)
        if len(unique) == 1:
            company = unique[0].copy()
            company["ev_supply_chain_role"] = role
            sole_suppliers.append(company)

    if not sole_suppliers:
        return "No matching companies found."

    header = "[Qdrant — Single-supplier roles]:"
    lines = [
        f"{c['company_name']} | {c.get('tier','')} | {c.get('ev_supply_chain_role','')} | "
        f"{c.get('location_county','')} | {int(_employment(c)) if _employment(c) else ''}"
        for c in sole_suppliers
    ]
    return header + "\nCompany | Tier | Role | County | Employment\n" + "\n".join(lines)


def retrieve_companies(question: str, entities: Entities) -> list[dict[str, Any]]:
    companies = _load_all_companies()
    filtered = [company for company in companies if _matches_entities(company, entities)]
    filtered = _refine_matches(question, entities, filtered)

    rank_by_company_id, ranked_companies = _semantic_rank(question)
    if filtered:
        return _sort_matches(question, entities, filtered, rank_by_company_id)

    if _has_structured_filters(entities):
        logger.info("Strict retrieval: structured filters produced 0 matches")
        return []

    if ranked_companies:
        reranked = rerank_companies(question, ranked_companies[:_rerank_cap(question, entities)])
        return reranked[:_limit_for_question(question, entities)]

    return []


def retrieve_context(question: str, entities: Entities) -> list[dict[str, Any]] | str:
    companies = _load_all_companies()
    filtered = [company for company in companies if _matches_entities(company, entities)]
    filtered = _refine_matches(question, entities, filtered)

    if entities.is_risk_query:
        return _format_risk_context(filtered or companies)

    if entities.is_aggregate:
        scope = filtered or companies
        label = "[Qdrant — Aggregated company records]:"
        return _format_aggregate_context(scope, label)

    matches = filtered
    rank_by_company_id, ranked_companies = _semantic_rank(question)
    if matches:
        matches = _sort_matches(question, entities, matches, rank_by_company_id)
    elif _has_structured_filters(entities):
        logger.info("Strict retrieval: structured filters produced 0 matches")
        return "No matching companies found."
    else:
        reranked = rerank_companies(question, ranked_companies[:_rerank_cap(question, entities)])
        matches = reranked[:_limit_for_question(question, entities)]

    if not matches:
        return "No matching companies found."

    return matches
