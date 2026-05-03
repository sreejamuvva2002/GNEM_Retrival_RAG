"""
Phase 1 — Query Generator
Generates Tavily search queries per company.
Adopted from Kb_Enrichment/src/searcher.py QUERY_FAMILIES — same proven patterns,
but now optimized for Tavily (advanced depth, no rate-limit hacks needed).
"""
from __future__ import annotations

from typing import Any

# ── Query families — same battle-tested patterns from Kb_Enrichment ─────────
# Family numbers preserved to keep convergence logic recognizable
QUERY_FAMILIES: dict[int, list[str]] = {
    # Core identification
    1: [
        "[COMPANY] Georgia",
        "[COMPANY] [LOCATION]",
        "[COMPANY] Georgia facility",
        "[COMPANY] Georgia operations",
        "[COMPANY] Georgia manufacturing",
    ],
    # Temporal — recent news
    2: [
        "[COMPANY] Georgia 2023",
        "[COMPANY] Georgia 2024",
        "[COMPANY] Georgia 2025",
        "[COMPANY] Georgia 2026",
        "[COMPANY] Georgia latest news",
    ],
    # Investment & jobs — HIGH VALUE
    3: [
        "[COMPANY] Georgia investment",
        "[COMPANY] Georgia jobs created",
        "[COMPANY] Georgia capital expenditure",
        "[COMPANY] Georgia expansion announcement",
        "[COMPANY] Georgia plant capacity",
        "[COMPANY] Georgia billion investment",
    ],
    # Supply chain relationships — HIGH VALUE
    4: [
        "[COMPANY] Georgia supplier",
        "[COMPANY] Georgia OEM customer",
        "[COMPANY] Georgia supply chain",
        "[COMPANY] Kia Georgia",
        "[COMPANY] Hyundai Metaplant Georgia",
        "[COMPANY] SK On Georgia",
        "[COMPANY] HMGMA supplier",
    ],
    # EV / battery specific
    5: [
        "[COMPANY] electric vehicle Georgia",
        "[COMPANY] EV battery Georgia",
        "[COMPANY] battery manufacturing Georgia",
        "[COMPANY] lithium ion Georgia",
        "[COMPANY] EV supply chain Georgia",
    ],
    # Official documents — HIGH VALUE
    6: [
        "[COMPANY] Georgia press release",
        "[COMPANY] Georgia annual report",
        "[COMPANY] Georgia SEC filing",
        "[COMPANY] site:sec.gov Georgia",
        "[COMPANY] site:energy.gov",
        "[COMPANY] site:georgia.org announcement",
        "[COMPANY] site:selectgeorgia.com",
    ],
    # Economic development — HIGH VALUE
    7: [
        "[COMPANY] Georgia economic development",
        "[COMPANY] Georgia GDEcD",
        "[COMPANY] Georgia governor announcement",
        "[COMPANY] Georgia tax incentive",
        "[COMPANY] Georgia Invest Georgia",
    ],
    # News coverage — HIGH VALUE
    8: [
        "[COMPANY] Georgia news",
        "[COMPANY] Georgia Reuters",
        "[COMPANY] Georgia Bloomberg",
        "[COMPANY] Georgia manufacturing news",
        "[COMPANY] Georgia Automotive News",
        "[COMPANY] Georgia plant opening",
    ],
}

# These families get temporal variants (year suffixes)
TEMPORAL_FAMILIES = {3, 6, 7, 8}

# All families — used when no specific subset assigned
ALL_FAMILIES = set(QUERY_FAMILIES.keys())

# Families for Tier 1 / OEM companies (broader search)
HIGH_TIER_FAMILIES = {1, 2, 3, 4, 5, 6, 7, 8}

# Families for Tier 2 / less prominent companies (focused search)
STANDARD_TIER_FAMILIES = {1, 2, 3, 4, 5}

# Tavily search depth — "advanced" costs 2 credits but gives much better results
# Since quota is not a concern: always use advanced
TAVILY_SEARCH_DEPTH = "advanced"
TAVILY_MAX_RESULTS = 10  # Per query — Tavily max is 20, 10 is optimal


def _fill_template(template: str, company: dict[str, Any]) -> str:
    """Replace [COMPANY] and [LOCATION] placeholders."""
    name = company.get("company_name", "")
    location = " ".join(filter(None, [
        company.get("location_city"),
        company.get("location_county"),
        company.get("location_state") or "Georgia",
    ])) or "Georgia"
    return template.replace("[COMPANY]", name).replace("[LOCATION]", location).strip()


def _select_families(company: dict[str, Any]) -> set[int]:
    """Select which query families to run based on company tier."""
    tier = str(company.get("tier", "")).lower()
    if any(t in tier for t in ["oem", "tier 1"]):
        return HIGH_TIER_FAMILIES
    return STANDARD_TIER_FAMILIES


def build_queries(company: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Generate all search queries for a company.
    Returns deduplicated list of query dicts ready for Tavily.

    Each query dict:
        query_text   : str  — the actual search string
        family       : int  — query family number
        company_id   : int  — DB id
        company_name : str
        search_depth : str  — "advanced" always
        max_results  : int
    """
    families = _select_families(company)
    seen: set[str] = set()
    queries: list[dict[str, Any]] = []

    for family_id in sorted(families):
        templates = QUERY_FAMILIES[family_id]
        for template in templates:
            query_text = _fill_template(template, company)
            normalized = query_text.lower().strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            queries.append({
                "query_text": query_text,
                "family": family_id,
                "company_id": company.get("id"),
                "company_name": company.get("company_name", ""),
                "search_depth": TAVILY_SEARCH_DEPTH,
                "max_results": TAVILY_MAX_RESULTS,
            })

    return queries


def estimate_query_count(companies: list[dict[str, Any]]) -> dict[str, int]:
    """Estimate total Tavily API calls needed before running."""
    total = 0
    oem_count = 0
    for company in companies:
        qs = build_queries(company)
        total += len(qs)
        tier = str(company.get("tier", "")).lower()
        if "oem" in tier or "tier 1" in tier:
            oem_count += len(qs)
    return {
        "total_queries": total,
        "oem_tier1_queries": oem_count,
        "standard_queries": total - oem_count,
        "estimated_tavily_credits": total * 2,  # advanced = 2 credits each
    }
