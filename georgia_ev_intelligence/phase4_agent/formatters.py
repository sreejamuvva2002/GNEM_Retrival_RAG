"""
phase4_agent/formatters.py
==============================================================
Template-based response formatters — zero LLM calls.

Added in V4: format_branched_answer() wraps existing per-class formatters
into the two-section "Two KB-supported interpretations of <term>" template
required when an ambiguity branch produces 2 RetrievalBranch outputs.

WHY TEMPLATES INSTEAD OF LLM SYNTHESIS:
  When data is already structured, LLM synthesis adds latency, RAM
  pressure, and hallucination risk. Python templates are 0ms, 0 RAM, and
  100% deterministic.

WHAT EACH FORMATTER HANDLES (output shape only — no domain values
appear here; rows are populated by the upstream retriever):
  aggregate    → "<county> has the highest total employment ..."
  company_list → ranked list with company / tier / role / employment
  county_top   → "<company> has the highest employment in <county> ..."
  oem_network  → "<n> Georgia suppliers linked to <oem> ..."
  facility     → "<n> Georgia companies with <facility_type> facilities ..."
  top_n        → "Top <n> Georgia EV supply chain companies by employment ..."

INVARIANT: every row passed to a formatter must have already been
validated by phase4_agent/evidence_validator (it sets
`row['validated'] = True`). Formatters assert this on entry to catch
pipeline mistakes. Pre-validated rows protect against the formatter
silently echoing un-checked text.

FALLBACK:
  If none of the structured templates apply (free-form question),
  return None — pipeline falls through to LLM synthesis.
"""
from __future__ import annotations


def _assert_validated(rows: list[dict]) -> None:
    """
    Defensive: refuse to format rows that did not pass evidence_validator.

    Aggregate rows (county / role / count outputs) are SQL-deterministic
    and may not carry the per-row validated flag — they're allowed through
    when they look like aggregates (no `company_name`).
    """
    for r in rows or []:
        if "company_name" not in r:
            continue  # aggregate row — SQL-authoritative, not a per-company row
        if not r.get("validated", False):
            raise AssertionError(
                "formatter received un-validated row: "
                f"{r.get('company_name', '<no name>')!r}"
            )


# ── Aggregate (GROUP BY county, SUM employment) ────────────────────────────────

def format_aggregate(rows: list[dict], tier: str | None = None) -> str | None:
    """
    Format employment-by-county aggregate results.

    Input rows come straight from sql_retriever.aggregate_employment_by_county
    and have shape:
        {"county": str, "total_employment": int, "company_count": int}
    """
    if not rows:
        return "No employment data found for the requested filters."
    _assert_validated(rows)

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

    Output shape (no domain values appear in source):
        "<n> Georgia companies found [<context>]:
           1. <name> | <tier> | <role> | <county> | <emp> employees
           ..."
    """
    if not companies:
        return "No matching companies found in the Georgia EV supply chain database."
    _assert_validated(companies)

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

    Output shape:
        "<top company> has the highest employment in <county> with <n> employees.
           EV Supply Chain Role : <role>
           Tier                 : <tier>
           Facility Type        : <facility_type>"
    """
    if not rows:
        return f"No companies found in {county} in the Georgia EV supply chain database."
    _assert_validated(rows)

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

    Output shape:
        "<n> Georgia suppliers linked to <oem> Automotive:
           1. <name> | <tier> | <role> | <county> | <n> emp
           ..."
    """
    if not rows:
        return f"No Georgia suppliers linked to {oem.title()} found in the database."
    _assert_validated(rows)

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
    _assert_validated(rows)

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
    _assert_validated(companies)

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


# ── Branched answer (ambiguity top-2) ─────────────────────────────────────────

def format_branch_section(branch) -> str:
    """
    Render one RetrievalBranch as a labelled section.

    Uses format_company_list() under the hood so the format is consistent
    with other deterministic outputs. When the branch has zero evidence
    rows, the placeholder line is explicit so the user can see that a
    KB-grounded interpretation was tried but returned nothing.
    """
    rows = [c.row for c in (branch.evidence or []) if getattr(c, "row", None)]
    label = f"Branch {branch.branch_id} — {branch.interpreted_meaning}"
    if not rows:
        return f"{label}\n(No KB rows matched this interpretation.)\n"
    body = format_company_list(rows, context_label="for this interpretation") or ""
    return f"{label}\n{body}\n"


def format_branched_answer(branches) -> str | None:
    """
    Compose the final two-section answer when ambiguity branching produced
    two RetrievalBranch objects. Returns None when there is at most one
    branch (caller falls through to the single-branch formatter or LLM
    synthesis).

    Output template:

        Two KB-supported interpretations of "<term>" were considered.

        Branch A — <interpretation A>
        <evidence rows>

        Branch B — <interpretation B>
        <evidence rows>
    """
    if not branches or len(branches) < 2:
        return None

    # Best-effort term extraction — interpretation strings begin with
    # "<term> → <mapping>"; we use the substring before the first arrow.
    first = branches[0].interpreted_meaning or ""
    term = first.split("→", 1)[0].strip() or "the abstract term"

    parts = [f'Two KB-supported interpretations of "{term}" were considered.', ""]
    for branch in branches:
        parts.append(format_branch_section(branch))
    return "\n".join(parts)
