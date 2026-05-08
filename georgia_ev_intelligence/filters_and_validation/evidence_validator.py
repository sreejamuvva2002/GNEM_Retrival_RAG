"""
Phase 4 — Hard-filter validator + answer verifier.

Two responsibilities:

  1. validate_all(...) — runs after retrieval merge, before fusion.
     Each Candidate is checked against the canonical gev_companies row
     (joined by company_row_id when known, else by canonical name). If
     any required entity field disagrees the candidate is rejected with
     a recorded `rejection_reason`. This is what blocks vector-only
     candidates from leaking through when the question has structural
     filters (tier, county, OEM, role, etc.).

  2. verify_answer(...) — runs after the answer text is generated.
     Extracts every company name and numeric claim from the answer and
     checks it against the selected evidence. Names not present and
     numbers not derivable from evidence are flagged as
     hallucination_risk; the pipeline downgrades the support level.

The validator never invents fields or values: only the entity fields the
extractor populated are checked, and only against actual gev_companies
columns.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from filters_and_validation.query_entity_extractor import Entities, _company_names
from core_agent.retrieval_types import (
    ALLOW_VECTOR_ONLY,
    AnswerVerification,
    Candidate,
    QueryClass,
)
from retrievals.sql_retriever import find_company_by_name
from shared.logger import get_logger

logger = get_logger("filters_and_validation.evidence_validator")


# ── Local match helpers (deliberately not imported from vector_retriever
# so this module stays usable even when vector_retriever is being slimmed) ─

def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalized_text(value: Any) -> str:
    text = _normalize(value)
    return text.replace("‑", "-").replace("–", "-").replace("—", "-")


def _tokenize(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalized_text(value))
        if len(token) >= 2
    }


def _tier_matches(company_tier: str, requested_tier: str) -> bool:
    company_norm = _normalize(company_tier)
    requested_norm = _normalize(requested_tier)
    if not requested_norm:
        return True
    if company_norm == requested_norm:
        return True
    if (
        "oem" in company_norm
        and "oem" in requested_norm
        and _tokenize(company_norm) == _tokenize(requested_norm)
    ):
        return True
    if requested_norm == "tier 2" and company_norm == "tier 2/3":
        return True
    if "/" in requested_norm and requested_norm in company_norm:
        return True
    if "/" in company_norm and requested_norm in company_norm:
        return True
    return False


def _role_matches(company_role: Any, requested_role: str) -> bool:
    role = _normalized_text(company_role)
    requested = _normalized_text(requested_role)
    if not requested:
        return True
    if role == requested:
        return True
    if requested in role or role in requested:
        return True
    requested_tokens = _tokenize(requested)
    role_tokens = _tokenize(role)
    return bool(requested_tokens) and requested_tokens.issubset(role_tokens)


def _employment(row: dict[str, Any]) -> float:
    raw = row.get("employment")
    if raw in (None, ""):
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


# ── KB row joining ───────────────────────────────────────────────────────────

def _kb_row_for(cand: Candidate) -> dict[str, Any] | None:
    """
    Return the canonical gev_companies row for a candidate.

    Order of preference: candidate.row already looks like gev_companies (has
    'id'), candidate.company_row_id, candidate.canonical_name lookup.
    """
    if cand.row and cand.row.get("id") and cand.row.get("company_name"):
        return cand.row
    name = cand.canonical_name
    if not name:
        return None
    fetched = find_company_by_name(name)
    return fetched


# ── Candidate-level validation ───────────────────────────────────────────────

def _check_entity_against_row(
    entities: Entities,
    branch_filters: dict[str, Any],
    row: dict[str, Any],
    query_class: QueryClass,
) -> str | None:
    """
    Returns the rejection reason if any entity field disagrees with the
    canonical row. Returns None when the row passes every check.
    """
    # Branch-level overrides win on conflicts.
    tier_required = branch_filters.get("tier") or entities.tier
    tier_list = entities.tier_list or ([tier_required] if tier_required else [])
    if tier_list:
        row_tier = row.get("tier") or ""
        if not any(_tier_matches(row_tier, t) for t in tier_list):
            return f"tier mismatch (row='{row_tier}', required={tier_list})"

    county = branch_filters.get("location_county") or entities.county
    if county and _normalize(row.get("location_county")) != _normalize(county):
        return f"county mismatch (row='{row.get('location_county')}', required='{county}')"

    industry = branch_filters.get("industry_group") or entities.industry_group
    if industry and _normalize(row.get("industry_group")) != _normalize(industry):
        return f"industry mismatch (row='{row.get('industry_group')}', required='{industry}')"

    facility = branch_filters.get("facility_type") or entities.facility_type
    if facility and _normalize(facility) not in _normalize(row.get("facility_type")):
        return f"facility mismatch (row='{row.get('facility_type')}', required='{facility}')"

    classification = branch_filters.get("classification_method") or entities.classification_method
    if classification and _normalize(row.get("classification_method")) != _normalize(classification):
        return "classification_method mismatch"

    affiliation = branch_filters.get("supplier_affiliation_type") or entities.supplier_affiliation_type
    if affiliation and _normalize(row.get("supplier_affiliation_type")) != _normalize(affiliation):
        return "supplier_affiliation_type mismatch"

    role_list = entities.ev_role_list or ([entities.ev_role] if entities.ev_role else [])
    if role_list:
        row_role = row.get("ev_supply_chain_role") or ""
        if not any(_role_matches(row_role, r) for r in role_list):
            return f"ev_role mismatch (row='{row_role}', required={role_list})"
    excluded = entities.exclude_ev_role_list or []
    if excluded:
        row_role = row.get("ev_supply_chain_role") or ""
        if any(_role_matches(row_role, r) for r in excluded):
            return f"ev_role excluded (row='{row_role}', excluded={excluded})"

    oem_list = entities.oem_list or ([entities.oem] if entities.oem else [])
    if oem_list:
        oem_text = _normalize(row.get("primary_oems"))
        if not any(_normalize(o) in oem_text for o in oem_list):
            return f"oem mismatch (row_oems='{row.get('primary_oems')}', required={oem_list})"

    min_emp = branch_filters.get("min_employment", entities.min_employment)
    max_emp = branch_filters.get("max_employment", entities.max_employment)
    emp = _employment(row)
    if min_emp is not None and emp < float(min_emp):
        return f"employment below min ({emp} < {min_emp})"
    if max_emp is not None and emp > float(max_emp):
        return f"employment above max ({emp} > {max_emp})"

    ev_value = branch_filters.get("ev_battery_relevant") or entities.ev_relevance_value
    if ev_value and _normalize(row.get("ev_battery_relevant")) != _normalize(ev_value):
        return f"ev_battery_relevant mismatch (row='{row.get('ev_battery_relevant')}', required='{ev_value}')"

    if entities.ev_relevant_filter:
        rel = _normalize(row.get("ev_battery_relevant"))
        if rel not in {"yes", "indirect"}:
            return "ev_relevant_filter required (row not Yes/Indirect)"

    # Product keywords are only enforced when the question is product-capability.
    if query_class == QueryClass.PRODUCT_CAPABILITY and entities.product_keywords:
        haystack = " ".join([
            _normalized_text(row.get("products_services")),
            _normalized_text(row.get("ev_supply_chain_role")),
            _normalized_text(row.get("company_name")),
        ])
        if not any(_normalized_text(k) in haystack for k in entities.product_keywords):
            return "no product keyword overlap with row"

    return None


def validate_candidate(
    cand: Candidate,
    entities: Entities,
    query_class: QueryClass,
    branch_filters: dict[str, Any] | None = None,
    bypass: bool = False,
) -> bool:
    """
    Apply hard-filter validation. Mutates `cand` to record the result.
    Returns True when the candidate survives, False when rejected.

    Aggregate / count / single-supplier rows are passed through unchanged
    because their `row` is already a deterministic SQL output and contains
    no per-company entity fields to validate.
    """
    if query_class in (QueryClass.AGGREGATE, QueryClass.COUNT, QueryClass.RISK):
        # SQL is authoritative for these classes — the row is already a
        # deterministic aggregate / SPOF result. Mark it validated so
        # downstream formatters can assert it on entry.
        cand.hard_filter_passed = True
        if cand.row is not None:
            cand.row["validated"] = True
        return True

    bf = branch_filters or {}
    row = _kb_row_for(cand)

    if row is None:
        if bypass or query_class in ALLOW_VECTOR_ONLY:
            cand.hard_filter_passed = True
            if cand.row is None:
                cand.row = {}
            cand.row["validated"] = True
            return True
        cand.hard_filter_passed = False
        cand.rejection_reason = "no canonical gev_companies row found"
        return False

    # Promote the canonical row onto the candidate so downstream code reads
    # ground-truth field values, not the chunk view.
    cand.row = row
    cand.company_row_id = row.get("id") or cand.company_row_id

    reason = _check_entity_against_row(entities, bf, row, query_class)
    if bypass or reason is None:
        cand.hard_filter_passed = True
        cand.row["validated"] = True
        return True

    cand.hard_filter_passed = False
    cand.rejection_reason = reason
    return False


def validate_all(
    candidates: list[Candidate],
    entities: Entities,
    query_class: QueryClass,
    branch_filters: dict[str, Any] | None = None,
    bypass: bool = False,
) -> list[Candidate]:
    """Filter candidates in-place. Returns the surviving list."""
    survivors: list[Candidate] = []
    rejected = 0
    for cand in candidates:
        if validate_candidate(cand, entities, query_class, branch_filters, bypass=bypass):
            survivors.append(cand)
        else:
            rejected += 1
    if rejected:
        logger.info(
            "evidence_validator: kept %d, rejected %d (qclass=%s)",
            len(survivors), rejected, query_class.value,
        )
    return survivors


# ── Answer verification ──────────────────────────────────────────────────────

_NUMBER_PATTERN = re.compile(r"\b\d{2,}(?:[,]\d{3})*(?:\.\d+)?\b")


def _company_names_in_text(text: str) -> set[str]:
    """Return canonical company names that literally appear in the text."""
    text_lower = text.lower()
    found: set[str] = set()
    for name in _company_names():
        if not name:
            continue
        if name.lower() in text_lower:
            found.add(name)
    return found


def _evidence_names(evidence: Iterable[Candidate]) -> set[str]:
    out: set[str] = set()
    for cand in evidence:
        if cand.canonical_name:
            out.add(cand.canonical_name)
        row_name = cand.row.get("company_name") if cand.row else None
        if row_name:
            out.add(row_name)
    return out


def _evidence_numbers(evidence: Iterable[Candidate]) -> set[str]:
    """Collect numbers (counts, sums, employment) that appear in evidence rows."""
    raw: set[str] = set()
    employments: list[float] = []
    for cand in evidence:
        row = cand.row or {}
        emp = row.get("employment")
        try:
            if emp not in (None, ""):
                employments.append(float(emp))
        except (TypeError, ValueError):
            pass
        for v in row.values():
            if isinstance(v, (int, float)) and v >= 10:
                raw.add(str(int(v)) if float(v).is_integer() else str(v))
    raw.add(str(len(list(evidence))))  # row count
    if employments:
        raw.add(str(int(sum(employments))))
    return raw


def verify_answer(
    answer_text: str,
    evidence: Iterable[Candidate],
) -> AnswerVerification:
    """
    Cross-check the answer against the validated evidence rows.

    Names that look like real companies (canonical KB list) but are absent
    from `evidence` count towards hallucination_risk. Numbers in the answer
    that cannot be reproduced from evidence rows or simple aggregations
    (sum of employment, row count) are flagged as suspect.

    Note: this is a *risk* signal, not a hard block. The pipeline uses it
    to downgrade the audit row's support_level — it does not rewrite the
    answer.
    """
    evidence_list = list(evidence)
    ev_names = _evidence_names(evidence_list)
    ev_numbers = _evidence_numbers(evidence_list)

    answer_names = _company_names_in_text(answer_text)
    missing = sorted(n for n in answer_names if n not in ev_names)

    suspect_numbers: list[str] = []
    answer_numbers = _NUMBER_PATTERN.findall(answer_text)
    for n in answer_numbers:
        # Strip thousand-sep commas so "2,280" matches "2280".
        n_clean = n.replace(",", "")
        if n_clean in ev_numbers:
            continue
        # Allow numbers that are derivable as a sum of an evidence subset of
        # employment values — too expensive to check exhaustively, so we only
        # accept exact matches and simple totals here.
        if n_clean.isdigit() and int(n_clean) <= len(evidence_list):
            continue
        suspect_numbers.append(n)

    risk = len(missing) + len(suspect_numbers)
    if risk == 0:
        status = "ok"
    elif risk <= 2:
        status = "risky"
    else:
        status = "failed"
    return AnswerVerification(
        status=status,
        hallucination_risk=risk,
        missing_names=missing,
        suspect_numbers=suspect_numbers,
    )
