"""
phase4_agent/formatters.py
==============================================================
Template-based response formatters — zero LLM calls.

WHY TEMPLATES INSTEAD OF LLM SYNTHESIS:
  When data is already structured, LLM synthesis adds:
    - 8-20s latency
    - RAM pressure
    - Risk of hallucination ("The database does not contain...")

  Python templates:
    - 0ms latency
    - 0 RAM
    - 100% deterministic — always correct format

WHAT EACH FORMATTER HANDLES:
  aggregate  → "Troup County has 2,280 employees across 7 Tier 1 companies"
  company_list → Ranked list with company / tier / role / employment
  county_top   → "SungEel has 650 employees in Gwinnett County. Role: Materials"
  oem_network  → "6 suppliers linked to Rivian Automotive: [list]"
  facility     → "1 R&D facility found: Racemark International LLC..."
  top_n        → "Top 5 Georgia EV companies by employment: [ranked list]"

FALLBACK:
  If none of the structured templates apply (free-form question),
  return None — pipeline falls through to LLM synthesis.
"""
from __future__ import annotations


# ── Aggregate (GROUP BY county, SUM employment) ────────────────────────────────

def format_aggregate(rows: list[dict], tier: str | None = None) -> str | None:
    """
    Format employment-by-county aggregate results.
    rows: [{"county": ..., "total_employment": ..., "company_count": ...}]

    Returns:
        "Troup County has the highest total employment among Tier 1 suppliers
         with 2,280 employees across 7 companies.
         Full ranking: ..."
    """
    if not rows:
        return "No employment data found for the requested filters."

    top = rows[0]
    county   = top.get("county") or top.get("location_county") or "Unknown County"
    emp      = int(top.get("total_employment") or 0)
    count    = top.get("company_count") or ""

    tier_str = f" among {tier} suppliers" if tier else ""

    lines = [
        f"{county} has the highest total employment{tier_str} "
        f"with {emp:,} employees across {count} companies.",
        "",
        "Full county ranking (highest to lowest):",
    ]
    for i, r in enumerate(rows[:15], 1):
        c   = r.get("county") or r.get("location_county") or ""
        e   = int(r.get("total_employment") or 0)
        n   = r.get("company_count") or ""
        lines.append(f"  {i:2d}. {c}: {e:,} employees ({n} companies)")

    return "\n".join(lines)


# ── Company list (role, product, facility queries) ────────────────────────────

def format_company_list(companies: list[dict], context_label: str = "") -> str | None:
    """
    Format a list of companies as a ranked readable list.

    Returns:
        "6 Georgia companies found [context]:
         1. Hitachi Astemo Americas Inc. | Tier 1/2 | Battery Cell | Harris County | 723 emp
         ..."
    """
    if not companies:
        return "No matching companies found in the Georgia EV supply chain database."

    label = f" {context_label}" if context_label else ""
    lines = [f"{len(companies)} Georgia compan{'y' if len(companies)==1 else 'ies'} found{label}:", ""]

    for i, c in enumerate(companies, 1):
        name    = c.get("company_name") or ""
        tier    = c.get("tier") or ""
        role    = c.get("ev_supply_chain_role") or ""
        county  = c.get("location_county") or ""
        emp     = c.get("employment")
        emp_str = f"{int(float(emp)):,} employees" if emp else "employment unknown"
        product = c.get("products_services") or ""

        line = f"  {i}. {name} | {tier} | {role} | {county} | {emp_str}"
        if product:
            line += f"\n     ↳ {product[:100]}"
        lines.append(line)

    return "\n".join(lines)


# ── County top-company query ──────────────────────────────────────────────────

def format_county_top(rows: list[dict], county: str) -> str | None:
    """
    Format "which company has highest employment in [county]?" result.

    Returns:
        "SungEel Recycling Park Georgia has the highest employment in Gwinnett County
         with 650 employees. EV Supply Chain Role: Materials."
    """
    if not rows:
        return f"No companies found in {county} in the Georgia EV supply chain database."

    # rows already sorted by employment desc by the SQL
    top = rows[0]
    name    = top.get("company_name") or ""
    emp     = top.get("employment")
    emp_str = f"{int(float(emp)):,}" if emp else "unknown"
    role    = top.get("ev_supply_chain_role") or "unknown"
    tier    = top.get("tier") or ""
    facility = top.get("facility_type") or ""

    lines = [
        f"{name} has the highest employment in {county} "
        f"with {emp_str} employees.",
        f"  EV Supply Chain Role : {role}",
        f"  Tier                 : {tier}",
    ]
    if facility:
        lines.append(f"  Facility Type        : {facility}")

    # Show other companies in the county if available
    if len(rows) > 1:
        lines += ["", f"Other companies in {county}:"]
        for r in rows[1:6]:
            n = r.get("company_name") or ""
            e = r.get("employment")
            e_str = f"{int(float(e)):,}" if e else "?"
            rl = r.get("ev_supply_chain_role") or ""
            lines.append(f"  • {n} | {e_str} emp | {rl}")

    return "\n".join(lines)


# ── OEM supplier network ──────────────────────────────────────────────────────

def format_oem_network(rows: list[dict], oem: str) -> str | None:
    """
    Format "show supplier network linked to [OEM]" result.

    Returns:
        "6 Georgia suppliers linked to Rivian Automotive:
         1. Duckyang | Tier 2/3 | General Automotive | Jackson County | 250 emp
         ..."
    """
    if not rows:
        return f"No Georgia suppliers linked to {oem.title()} found in the database."

    # Sort by employment desc
    rows_sorted = sorted(rows, key=lambda x: float(x.get("employment") or 0), reverse=True)

    lines = [
        f"{len(rows_sorted)} Georgia supplier{'s' if len(rows_sorted) != 1 else ''} "
        f"linked to {oem.title()} Automotive:",
        ""
    ]
    for i, c in enumerate(rows_sorted, 1):
        name    = c.get("company_name") or ""
        tier    = c.get("tier") or ""
        role    = c.get("ev_supply_chain_role") or ""
        county  = c.get("location_county") or ""
        emp     = c.get("employment")
        emp_str = f"{int(float(emp)):,} emp" if emp else "emp unknown"
        lines.append(f"  {i}. {name} | {tier} | {role} | {county} | {emp_str}")

    # Total employment
    total_emp = sum(int(float(r.get("employment") or 0)) for r in rows_sorted)
    lines += ["", f"Total employment across {len(rows_sorted)} suppliers: {total_emp:,}"]

    return "\n".join(lines)


# ── Top-N companies by employment ─────────────────────────────────────────────

def format_top_n(rows: list[dict], n: int, tier: str | None = None) -> str | None:
    """
    Format "top N Georgia companies by employment" result.
    """
    if not rows:
        return f"No companies found for the requested filters."

    tier_str = f" {tier}" if tier else ""
    lines = [
        f"Top {len(rows)} Georgia{tier_str} EV supply chain companies by employment:",
        ""
    ]
    for i, c in enumerate(rows, 1):
        name    = c.get("company_name") or ""
        emp     = c.get("employment")
        emp_str = f"{int(float(emp)):,}" if emp else "unknown"
        tier_v  = c.get("tier") or ""
        role    = c.get("ev_supply_chain_role") or ""
        county  = c.get("location_county") or ""
        lines.append(f"  {i:2d}. {name} | {emp_str} emp | {tier_v} | {role} | {county}")

    return "\n".join(lines)


# ── Facility type results ─────────────────────────────────────────────────────

def format_facility(companies: list[dict], facility_type: str) -> str | None:
    """
    Format "which companies operate [facility_type] facilities?" result.
    """
    if not companies:
        return f"No Georgia companies with {facility_type} facilities found in the database."

    lines = [
        f"{len(companies)} Georgia compan{'y' if len(companies)==1 else 'ies'} "
        f"with {facility_type} facilities:",
        ""
    ]
    for i, c in enumerate(companies, 1):
        name    = c.get("company_name") or ""
        tier    = c.get("tier") or ""
        role    = c.get("ev_supply_chain_role") or ""
        county  = c.get("location_county") or ""
        emp     = c.get("employment")
        emp_str = f"{int(float(emp)):,} emp" if emp else ""
        product = c.get("products_services") or ""
        lines.append(f"  {i}. {name} | {tier} | {role} | {county} | {emp_str}")
        if product:
            lines.append(f"     ↳ {product[:100]}")

    return "\n".join(lines)
