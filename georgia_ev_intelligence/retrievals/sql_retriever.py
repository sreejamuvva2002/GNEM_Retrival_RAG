"""
Phase 4 — SQL Retriever
Runs structured PostgreSQL queries against gev_companies and gev_documents.

WHY SQL AS PRIMARY RETRIEVER:
  43/50 evaluation questions are aggregation or filter queries that
  cannot be answered by semantic vector search alone — SQL gives exact,
  complete, deterministic results.

The retriever validates incoming filter operators against the approved
SUPPORTED_OPERATORS set from shared/metadata_schema.py. Field references
should be looked up via CANONICAL_FIELDS so a future column rename only
changes one file.
"""
from __future__ import annotations
from typing import Any
from shared.config import Config
from shared.db import get_session, Company
from shared.logger import get_logger
from shared.metadata_schema import CANONICAL_FIELDS, SUPPORTED_OPERATORS

logger = get_logger("retrievals.sql_retriever")


def _employment_outlier_cap() -> float:
    """
    Read the employment outlier cap from config/settings.yaml. Cap excludes
    rows whose `employment` value is a global parent headcount, not the
    Georgia footprint. Returns 0 (disabled) when not configured.
    """
    try:
        ph4 = getattr(Config.get().settings, "phase4", None)
        cap = getattr(ph4, "employment_outlier_cap", 0) if ph4 is not None else 0
        return float(cap or 0)
    except Exception:
        return 0.0


def _validate_operator(op: str) -> None:
    """Reject operators that are not in the approved SUPPORTED_OPERATORS set."""
    if op not in SUPPORTED_OPERATORS:
        raise ValueError(
            f"unsupported operator {op!r}; allowed: {sorted(SUPPORTED_OPERATORS)}"
        )


def _canonical(name: str) -> str:
    """
    Resolve a logical field name to its canonical KB column name. If `name`
    is already a canonical column it is returned unchanged.
    """
    if name in CANONICAL_FIELDS:
        return CANONICAL_FIELDS[name]
    if name in CANONICAL_FIELDS.values():
        return name
    raise KeyError(f"unknown field {name!r}; not in CANONICAL_FIELDS")


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
                # Support multi-OEM: "rivian OR kia" → OR across all
                if " OR " in str(oem):
                    from sqlalchemy import or_
                    parts = [p.strip() for p in oem.split(" OR ")]
                    q = q.filter(or_(*[Company.primary_oems.ilike(f"%{p}%") for p in parts]))
                else:
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
        )
        outlier_cap = _employment_outlier_cap()
        if outlier_cap > 0:
            q = q.filter(Company.employment <= outlier_cap)
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
    max_employment: float | None = None,
) -> list[dict]:
    """
    Return top N companies by employment size.
    Used for 'Top 10 companies by employment' questions.
    Excludes global headcount outliers when phase4.employment_outlier_cap is
    set in config/settings.yaml. Pass an explicit max_employment to override.
    """
    if max_employment is None:
        max_employment = _employment_outlier_cap()
    session = get_session()
    try:
        q = session.query(Company).filter(Company.employment.isnot(None))
        if max_employment and max_employment > 0:
            q = q.filter(Company.employment <= max_employment)
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
        "id":                     c.id,
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


# ── Pipeline-level wiring: SQLPlan + run_plan ────────────────────────────────
# Used by the new retrieval pipeline (pipeline.py → ambiguity_resolver →
# retrieval_fusion). Wraps the existing helpers above without changing their
# public behaviour.

def filters_from_entities(entities, branch_filters: dict | None = None) -> dict:
    """
    Translate a deterministic Entities object plus optional branch-level
    overlay into the filters dict consumed by query_companies().

    Branch overlay wins on key collisions — that is how an ambiguity branch
    narrows or widens a base interpretation (e.g. branch A adds
    {"max_employment": 200}, branch B adds {"tier": "Tier 2"}).
    """
    filters: dict = {}
    if entities.tier:
        filters["tier"] = entities.tier
    elif entities.tier_list:
        filters["tier"] = entities.tier_list[0]
    if entities.county:
        filters["location_county"] = entities.county
    if entities.industry_group:
        filters["industry_group"] = entities.industry_group
    if entities.facility_type:
        filters["facility_type"] = entities.facility_type
    if entities.classification_method:
        filters["classification_method"] = entities.classification_method
    if entities.supplier_affiliation_type:
        filters["supplier_affiliation_type"] = entities.supplier_affiliation_type
    if entities.company_name:
        filters["company_name"] = entities.company_name
    if entities.min_employment is not None:
        filters["min_employment"] = entities.min_employment
    if entities.max_employment is not None:
        filters["max_employment"] = entities.max_employment
    if entities.ev_relevance_value:
        filters["ev_battery_relevant"] = entities.ev_relevance_value

    role_values: list[str] = []
    if entities.ev_role:
        role_values.append(entities.ev_role)
    role_values.extend(r for r in entities.ev_role_list if r not in role_values)
    if role_values:
        filters["ev_supply_chain_role"] = " OR ".join(role_values)

    oem_values: list[str] = []
    if entities.oem:
        oem_values.append(entities.oem)
    oem_values.extend(o for o in entities.oem_list if o not in oem_values)
    if oem_values:
        filters["primary_oems"] = " OR ".join(oem_values)

    if branch_filters:
        for k, v in branch_filters.items():
            if v is None:
                continue
            filters[k] = v

    return filters


def _wrap_rows_as_candidates(rows: list[dict]) -> list:
    """
    Convert SQL result rows into Candidate objects for the fusion layer.
    The `sql` source contributes a flat 1.0 to that source's normalised
    score (post-min-max it stays 1.0 because every SQL row has the same
    raw score — SQL is binary: matched or not).
    """
    from core_agent.retrieval_types import Candidate
    candidates: list[Candidate] = []
    for row in rows:
        name = (row.get("company_name") or "").strip()
        if not name:
            continue
        cand = Candidate(
            canonical_name=name,
            company_row_id=row.get("id"),
            row=row,
        )
        cand.add_source("sql", 1.0)
        candidates.append(cand)
    return candidates


def run_plan(plan, branch_filters: dict | None = None, entities=None) -> list:
    """
    Execute a SQLPlan against gev_companies and wrap rows as Candidates.

    Branch filters overlay the entity-derived filters when plan.mode=='filter'
    or 'keyword_products'. For aggregate / count / top_n / single_supplier
    modes the plan carries its own parameters.
    """
    mode = plan.mode

    if mode == "filter":
        filters = dict(plan.filters)
        if entities is not None:
            filters = filters_from_entities(entities, branch_filters) | filters
        elif branch_filters:
            filters = filters | branch_filters
        rows = query_companies(filters=filters, limit=plan.limit)
        return _wrap_rows_as_candidates(rows)

    if mode == "aggregate_county":
        # Aggregate rows do not have company_name — return raw rows wrapped
        # as candidates with canonical_name = "<county> (aggregate)" so the
        # audit table records them; downstream evidence selection uses
        # query_class==AGGREGATE to format from the row dicts directly.
        from core_agent.retrieval_types import Candidate
        rows = aggregate_employment_by_county(tier=plan.tier)
        out: list = []
        for r in rows:
            cand = Candidate(
                canonical_name=f"{r.get('county')} (aggregate)",
                row=r,
            )
            cand.add_source("sql", 1.0)
            out.append(cand)
        return out

    if mode == "count_role":
        from core_agent.retrieval_types import Candidate
        rows = count_by_role()
        return [
            (lambda r: (
                lambda c: (c.add_source("sql", 1.0) or c)
            )(Candidate(canonical_name=f"{r.get('role')} (count)", row=r)))(r)
            for r in rows
        ]

    if mode == "top_n_employment":
        rows = top_companies_by_employment(
            limit=plan.limit,
            ev_relevant_only=plan.ev_relevant_only,
            tier=plan.tier,
        )
        return _wrap_rows_as_candidates(rows)

    if mode == "single_supplier":
        from core_agent.retrieval_types import Candidate
        rows = get_single_supplier_roles()
        out: list = []
        for r in rows:
            cand = Candidate(
                canonical_name=r.get("company") or r.get("role") or "(unknown)",
                row=r,
            )
            cand.add_source("sql", 1.0)
            out.append(cand)
        return out

    if mode == "keyword_products":
        rows = keyword_search_products(plan.keywords, tier=plan.tier)
        return _wrap_rows_as_candidates(rows)

    if mode == "full_text":
        rows = full_text_search(plan.keywords, tier=plan.tier, limit=plan.limit)
        return _wrap_rows_as_candidates(rows)

    logger.warning("run_plan: unknown SQL plan mode %s — returning empty", mode)
    return []


def find_company_by_name(name: str) -> dict | None:
    """
    Look up a single canonical gev_companies row by case-insensitive name.
    Used by evidence_validator to confirm vector candidates against KB
    truth before letting them survive into final evidence.
    """
    if not name:
        return None
    session = get_session()
    try:
        row = (
            session.query(Company)
            .filter(Company.company_name.ilike(name.strip()))
            .one_or_none()
        )
        if row is None:
            row = (
                session.query(Company)
                .filter(Company.company_name.ilike(f"%{name.strip()}%"))
                .order_by(Company.employment.desc().nullslast())
                .first()
            )
        return _company_to_dict(row) if row else None
    finally:
        session.close()
