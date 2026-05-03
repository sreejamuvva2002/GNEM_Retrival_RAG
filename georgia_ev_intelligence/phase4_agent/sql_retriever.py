"""
Phase 4 — SQL Retriever
Runs structured PostgreSQL queries against gev_companies and gev_documents.

WHY SQL AS PRIMARY RETRIEVER:
  43/50 evaluation questions are aggregation or filter queries:
    "List all Tier 1/2 suppliers" → WHERE tier = 'Tier 1/2'
    "Which county has highest employment?" → GROUP BY county ORDER BY SUM
    "Companies with <200 employees in Thermal Management" → WHERE + AND

  These cannot be answered by semantic vector search alone.
  SQL gives exact, complete, deterministic results.
"""
from __future__ import annotations
from typing import Any
from shared.db import get_session, Company
from shared.logger import get_logger

logger = get_logger("phase4.sql_retriever")


def _tier_filter(model: type, tier: str):
    """
    Build the correct SQLAlchemy tier filter.

    The database has values like: 'Tier 1', 'Tier 2/3', 'Tier 1/2', 'OEM (Footprint)'

    PROBLEM with ilike('%Tier 1%'):
      'Tier 1 only' question → ilike matches 'Tier 1' AND 'Tier 1/2' AND 'Tier 1/2/3'
      → WRONG: inflates employment totals with cross-tier companies

    SOLUTION:
      Simple tier (no '/'): use exact case-insensitive match  → 'Tier 1' only
      Compound tier ('/'): use ilike (user explicitly asked for mixed tier)
      'OEM*' tiers: use ilike (covers 'OEM (Footprint)', 'OEM Supply Chain')
    """
    from sqlalchemy import func
    tier_clean = tier.strip()
    # Compound tier or OEM tier → broad ilike match
    if "/" in tier_clean or tier_clean.upper().startswith("OEM"):
        return model.tier.ilike(f"%{tier_clean}%")
    # Simple tier (e.g. 'Tier 1', 'Tier 2', 'Tier 3') → exact match only
    return func.lower(model.tier) == tier_clean.lower()


def query_companies(filters: dict[str, Any] | None = None, limit: int = 200) -> list[dict]:
    """
    Fetch companies from PostgreSQL with optional filters.

    Supported filters:
      tier, ev_supply_chain_role, ev_battery_relevant, facility_type,
      location_county, industry_group, primary_oems, products_services,
      min_employment, max_employment, company_name
    """
    session = get_session()
    try:
        q = session.query(Company)

        if filters:
            if t := filters.get("tier"):
                q = q.filter(_tier_filter(Company, t))
            if name := filters.get("company_name"):
                q = q.filter(Company.company_name.ilike(f"%{name}%"))
            if role := filters.get("ev_supply_chain_role"):
                # Handle OR-joined values: "Battery Cell OR Battery Pack"
                if " OR " in str(role):
                    from sqlalchemy import or_
                    parts = [r.strip() for r in role.split(" OR ")]
                    q = q.filter(or_(*[Company.ev_supply_chain_role.ilike(f"%{p}%") for p in parts]))
                else:
                    q = q.filter(Company.ev_supply_chain_role.ilike(f"%{role}%"))
            if ev := filters.get("ev_battery_relevant"):
                q = q.filter(Company.ev_battery_relevant == ev)
            if county := filters.get("location_county"):
                q = q.filter(Company.location_county.ilike(f"%{county}%"))
            if industry := filters.get("industry_group"):
                q = q.filter(Company.industry_group.ilike(f"%{industry}%"))
            if oem := filters.get("primary_oems"):
                q = q.filter(Company.primary_oems.ilike(f"%{oem}%"))
            if product := filters.get("products_services"):
                q = q.filter(Company.products_services.ilike(f"%{product}%"))
            if facility := filters.get("facility_type"):
                q = q.filter(Company.facility_type.ilike(f"%{facility}%"))
            if min_emp := filters.get("min_employment"):
                q = q.filter(Company.employment >= float(min_emp))
            if max_emp := filters.get("max_employment"):
                q = q.filter(Company.employment <= float(max_emp))

        # Sort by employment descending so highest-employment companies appear first
        # in the capped context window — most useful results visible to LLM
        rows = q.order_by(Company.employment.desc().nullslast()).limit(limit).all()
        results = [_company_to_dict(c) for c in rows]
        logger.info("SQL query returned %d companies (filters=%s)", len(results), filters)
        return results
    finally:
        session.close()


def aggregate_employment_by_county(tier: str | None = None) -> list[dict]:

    """Return total employment per county, optionally filtered by tier."""
    from sqlalchemy import func
    session = get_session()
    try:
        q = session.query(
            Company.location_county,
            func.sum(Company.employment).label("total_employment"),
            func.count(Company.id).label("company_count"),
        ).filter(
            Company.location_county.isnot(None),
            Company.employment.isnot(None),
            Company.employment <= 100000,   # exclude global headcount outliers
        )
        if tier:
            q = q.filter(_tier_filter(Company, tier))
        q = q.group_by(Company.location_county).order_by(func.sum(Company.employment).desc())
        rows = q.all()
        return [
            {"county": r.location_county, "total_employment": int(r.total_employment or 0), "company_count": r.company_count}
            for r in rows
        ]
    finally:
        session.close()


def count_by_role() -> list[dict]:
    """Return count of companies per EV supply chain role."""
    from sqlalchemy import func
    session = get_session()
    try:
        rows = session.query(
            Company.ev_supply_chain_role,
            func.count(Company.id).label("cnt"),
        ).filter(
            Company.ev_supply_chain_role.isnot(None)
        ).group_by(Company.ev_supply_chain_role).order_by(func.count(Company.id)).all()
        return [{"role": r.ev_supply_chain_role, "count": r.cnt} for r in rows]
    finally:
        session.close()


def top_companies_by_employment(
    limit: int = 10,
    ev_relevant_only: bool = False,
    tier: str | None = None,
    max_employment: float = 100000,
) -> list[dict]:
    """
    Return top N companies by employment size.
    Used for 'Top 10 companies by employment' questions.
    Excludes global headcount outliers (>100k) by default.
    """
    session = get_session()
    try:
        q = session.query(Company).filter(
            Company.employment.isnot(None),
            Company.employment <= max_employment,
        )
        if ev_relevant_only:
            # 'Yes' or 'Indirect' — not 'No'
            from sqlalchemy import or_
            q = q.filter(
                or_(
                    Company.ev_battery_relevant.ilike("%yes%"),
                    Company.ev_battery_relevant.ilike("%indirect%"),
                )
            )
        if tier:
            q = q.filter(Company.tier.ilike(f"%{tier}%"))
        rows = q.order_by(Company.employment.desc()).limit(limit).all()
        return [_company_to_dict(c) for c in rows]
    finally:
        session.close()


def get_single_supplier_roles() -> list[dict]:
    """Return EV supply chain roles served by exactly ONE company (single-point-of-failure)."""
    from sqlalchemy import func
    session = get_session()
    try:
        subq = session.query(
            Company.ev_supply_chain_role,
            func.count(Company.id).label("cnt"),
        ).filter(
            Company.ev_supply_chain_role.isnot(None)
        ).group_by(Company.ev_supply_chain_role).having(func.count(Company.id) == 1).subquery()

        rows = session.query(Company).join(
            subq, Company.ev_supply_chain_role == subq.c.ev_supply_chain_role
        ).all()
        return [{"role": c.ev_supply_chain_role, "company": c.company_name, "tier": c.tier} for c in rows]
    finally:
        session.close()


def keyword_search_products(keywords: list[str], tier: str | None = None) -> list[dict]:
    """
    Search across ALL text columns for any of the given keywords.
    Includes company_name so 'recycling' finds 'SungEel Recycling Park Georgia',
    and facility_type so 'R&D' and 'manufacturing plant' return results.
    """
    from sqlalchemy import or_
    session = get_session()
    try:
        conditions = []
        for kw in keywords:
            conditions.extend([
                Company.company_name.ilike(f"%{kw}%"),           # SungEel Recycling, R&D firms
                Company.products_services.ilike(f"%{kw}%"),
                Company.ev_supply_chain_role.ilike(f"%{kw}%"),
                Company.ev_battery_relevant.ilike(f"%{kw}%"),
                Company.industry_group.ilike(f"%{kw}%"),
                Company.supplier_affiliation_type.ilike(f"%{kw}%"),
                Company.classification_method.ilike(f"%{kw}%"),
                Company.facility_type.ilike(f"%{kw}%"),          # R&D, Manufacturing plant
            ])
        q = session.query(Company).filter(or_(*conditions))
        if tier:
            q = q.filter(Company.tier.ilike(f"%{tier}%"))
        rows = q.order_by(Company.employment.desc().nullslast()).all()
        seen: set[str] = set()
        results = []
        for c in rows:
            if c.company_name not in seen:
                seen.add(c.company_name)
                results.append(_company_to_dict(c))
        return results
    finally:
        session.close()


def full_text_search(question_words: list[str], tier: str | None = None, limit: int = 50) -> list[dict]:
    """
    Broad fallback search: searches meaningful words across every text column.
    Used when structured SQL returns 0 results.

    Fixes vs. naive split():
      - Strips punctuation ('.', '?', ',') from each word before matching
      - Expands skip list to remove structural/question words that add noise
      - Searches location_county and primary_oems so county + OEM questions work
    """
    import re
    from sqlalchemy import or_
    session = get_session()
    try:
        # Structural words that NEVER match useful database content
        _skip = {
            "the", "and", "or", "in", "of", "to", "for", "by", "with", "is", "are",
            "a", "an", "that", "this", "which", "what", "how", "where", "when",
            "georgia", "company", "companies", "supplier", "suppliers",
            # Question-framing verbs — never in DB values
            "show", "find", "list", "give", "get", "tell", "identify",
            "full", "linked", "broken", "down", "into", "has", "have",
            "its", "their", "from", "who", "are", "does", "been",
        }

        cleaned = []
        for w in question_words:
            # Strip leading/trailing punctuation (?, ., ,, :, ;)
            w_clean = re.sub(r"^[^\w]+|[^\w]+$", "", w)
            if w_clean.lower() not in _skip and len(w_clean) >= 3:
                cleaned.append(w_clean)

        if not cleaned:
            return []

        conditions = []
        for word in cleaned:
            conditions.extend([
                Company.company_name.ilike(f"%{word}%"),
                Company.products_services.ilike(f"%{word}%"),
                Company.ev_supply_chain_role.ilike(f"%{word}%"),
                Company.industry_group.ilike(f"%{word}%"),
                Company.facility_type.ilike(f"%{word}%"),
                Company.ev_battery_relevant.ilike(f"%{word}%"),
                Company.classification_method.ilike(f"%{word}%"),
                Company.primary_oems.ilike(f"%{word}%"),       # OEM name matches
                Company.location_county.ilike(f"%{word}%"),    # county matches
                Company.location_city.ilike(f"%{word}%"),      # city matches
            ])
        q = session.query(Company).filter(or_(*conditions))
        if tier:
            q = q.filter(Company.tier.ilike(f"%{tier}%"))
        rows = q.order_by(Company.employment.desc().nullslast()).limit(limit).all()
        seen: set[str] = set()
        results = []
        for c in rows:
            if c.company_name not in seen:
                seen.add(c.company_name)
                results.append(_company_to_dict(c))
        logger.info("full_text_search returned %d companies for words=%s", len(results), cleaned[:5])
        return results
    finally:
        session.close()


def _company_to_dict(c: Company) -> dict:
    return {
        "company_name":           c.company_name,
        "tier":                   c.tier,
        "ev_supply_chain_role":   c.ev_supply_chain_role,
        "ev_battery_relevant":    c.ev_battery_relevant,
        "industry_group":         c.industry_group,
        "facility_type":          getattr(c, "facility_type", None),
        "location_city":          c.location_city,
        "location_county":        c.location_county,
        "employment":             c.employment,
        "products_services":      c.products_services,
        "primary_oems":           c.primary_oems,
        "classification_method":  c.classification_method,
        "supplier_affiliation_type": c.supplier_affiliation_type,
    }
