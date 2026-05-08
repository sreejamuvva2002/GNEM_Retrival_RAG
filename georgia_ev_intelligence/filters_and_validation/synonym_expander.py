"""
Phase 4 — KB-grounded synonym expansion for residual abstract terms.

The expander resolves a residual abstract phrase (e.g. "small scale",
"innovation stage") into KB-supported candidate mappings using EXACTLY
two sources of truth, in priority order:

    1. direct  — the phrase is itself a canonical column name or a value
                 currently stored in a hard-filter column. No interpretation
                 happens; we just point at the literal KB value.

    2. rule    — gev_domain_mapping_rules has an approved row whose
                 `valid_when` predicate fires for this question.

If neither source returns anything, the term is flagged "unresolved" and
the pipeline records it in the audit log. No mapping is invented.

WHY NO HEURISTIC FALLBACK
-------------------------
A previous version of this module shipped an inline heuristic-templates
dict that hardcoded abstract-term mappings without human approval. Those
mappings were model guesses, and they let the agent silently overfit to
one question's wording. They were removed per the refactor plan
("no feedback = no permanent learning"). To enable a new mapping, add
an approved row via the admin script — never inline.

LLM-suggested mappings are NEVER inserted into the rule store. The audit
pipeline may capture them with `support_basis="llm_suggestion"` and
confidence capped at 0.50; only an explicit human approval (out-of-band
admin tool) writes a row to gev_domain_mapping_rules.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from filters_and_validation.query_entity_extractor import (
    Entities,
    _classification_methods,
    _facility_types,
    _industry_groups,
    _kb_companies,
    _supplier_affiliation_types,
    _tier_names,
)
from filters_and_validation.metadata_loader import loader as kb_loader
from core_agent.retrieval_types import CandidateMapping, ResolvedTerm
from shared.db import DomainMappingRule, get_session
from shared.logger import get_logger
from shared.metadata_schema import CANONICAL_FIELDS, HARD_FILTER_FIELDS

logger = get_logger("filters_and_validation.synonym_expander")


# Approved rule-status values. The schema's column default is "active" for
# legacy compatibility; new admin scripts should set "approved". Both are
# accepted in the read path so the migration can flip statuses lazily.
_APPROVED_STATUS_VALUES: frozenset[str] = frozenset({"approved", "active"})


# ── KB schema introspection (metadata only) ──────────────────────────────────

# Canonical KB column names — pulled from the central metadata schema so this
# module never carries its own hardcoded copy.
KB_COLUMN_NAMES: tuple[str, ...] = tuple(sorted(CANONICAL_FIELDS.values()))


@lru_cache(maxsize=1)
def _all_product_phrases() -> set[str]:
    """All non-empty products_services strings, lowercased."""
    return {
        str(c.get("products_services") or "").lower()
        for c in _kb_companies()
        if c.get("products_services")
    }


def _value_set(column: str) -> set[str]:
    """Lowercased set of distinct values for an indexed KB column."""
    if column == "tier":
        return {v.lower() for v in _tier_names()}
    if column == "facility_type":
        return {v.lower() for v in _facility_types()}
    if column == "industry_group":
        return {v.lower() for v in _industry_groups()}
    if column == "classification_method":
        return {v.lower() for v in _classification_methods()}
    if column == "supplier_affiliation_type":
        return {v.lower() for v in _supplier_affiliation_types()}
    return set()


# ── Direct schema match ──────────────────────────────────────────────────────

def _direct_schema_mapping(term: str) -> CandidateMapping | None:
    """If the term is itself a column name or a known column value, accept it."""
    t = term.lower().strip()
    if t in KB_COLUMN_NAMES:
        return CandidateMapping(
            mapping=f"{t} (column)",
            mapped_column=t,
            mapped_value=None,
            support_basis="kb_column",
            confidence=1.0,
        )
    for col in (
        "tier",
        "facility_type",
        "industry_group",
        "classification_method",
        "supplier_affiliation_type",
    ):
        if t in _value_set(col):
            return CandidateMapping(
                mapping=f"{col} = '{t}'",
                mapped_column=col,
                mapped_value=t,
                support_basis="kb_value",
                confidence=1.0,
            )
    if any(t in phrase for phrase in _all_product_phrases()):
        return CandidateMapping(
            mapping=f"products_services contains '{t}'",
            mapped_column="products_services",
            mapped_value=t,
            support_basis="kb_value",
            confidence=0.85,
        )
    return None


# ── Approved rule store lookup ───────────────────────────────────────────────

def _rule_predicate_holds(rule: DomainMappingRule, question_lower: str) -> bool:
    """
    Honour the rule's `valid_when` and `invalid_when` text predicates.

    Predicates are simple substring checks: a comma-separated list of cue
    phrases. The rule fires if at least one valid_when cue is present and no
    invalid_when cue is present. Empty `valid_when` → always valid.
    """
    valid_cues = [c.strip().lower() for c in (rule.valid_when or "").split(",") if c.strip()]
    invalid_cues = [c.strip().lower() for c in (rule.invalid_when or "").split(",") if c.strip()]
    if any(c in question_lower for c in invalid_cues):
        return False
    if valid_cues and not any(c in question_lower for c in valid_cues):
        return False
    return True


def _rule_mappings_for(term: str, question_lower: str) -> list[CandidateMapping]:
    """
    Fetch approved rules whose term matches and whose predicate holds.

    Only rows with `status` in _APPROVED_STATUS_VALUES are read. Rows in
    `proposed`, `deprecated`, or any other status are ignored.
    """
    try:
        session = get_session()
    except Exception as exc:
        logger.debug("rule lookup skipped (db unavailable): %s", exc)
        return []
    try:
        rows = (
            session.query(DomainMappingRule)
            .filter(DomainMappingRule.status.in_(list(_APPROVED_STATUS_VALUES)))
            .filter(DomainMappingRule.term.ilike(term))
            .all()
        )
    except Exception as exc:
        logger.debug("rule lookup query failed: %s", exc)
        return []
    finally:
        session.close()

    out: list[CandidateMapping] = []
    for r in rows:
        if not _rule_predicate_holds(r, question_lower):
            continue
        # Only allow rules pointing at a hard-filter column we recognise. This
        # is a defence-in-depth check: even if a rule sneaks in pointing at
        # an unknown column, the retriever cannot apply it.
        col_norm = (r.mapped_column or "").strip().lower()
        if col_norm and col_norm not in HARD_FILTER_FIELDS:
            logger.warning(
                "synonym_expander: ignoring rule id=%s — mapped_column %r is not a hard filter field",
                r.id, r.mapped_column,
            )
            continue
        out.append(CandidateMapping(
            mapping=f"{r.mapped_column} {r.mapped_value_or_condition}",
            mapped_column=r.mapped_column,
            mapped_value=r.mapped_value_or_condition,
            support_basis="rule",
            confidence=float(r.confidence or 1.0),
        ))
    return out


# ── KB column candidates for an unresolved term ──────────────────────────────

def _kb_columns_supporting_term(term: str) -> list[str]:
    """
    Hard-filter columns whose distinct values contain `term` as a substring.

    Used by ambiguity_resolver to build interpretation branches for terms
    that the rule store does not cover. Returns [] when no column has any
    value containing the term — in that case the term is genuinely unmapped
    and the pipeline records it as such.
    """
    try:
        return kb_loader.kb_columns_supporting(term)
    except Exception as exc:
        logger.debug("kb_columns_supporting failed for %r: %s", term, exc)
        return []


# ── Public API ───────────────────────────────────────────────────────────────

def resolve(
    terms: Iterable[str],
    entities: Entities,
    question: str,
) -> list[ResolvedTerm]:
    """
    Resolve abstract residual terms using the direct → rule → unresolved
    decision chain. No heuristic invention.

    Decision rules per term:

        0 candidates                                   → unresolved (drop)
        1 candidate, confidence ≥ 0.95                 → apply (direct)
        1 candidate, support_basis="rule"              → apply (rule)
        ≥ 2 candidates                                 → branch_top_2

    Note: the `entities` argument is currently unused but kept in the
    signature so call sites can pass extracted entities for future
    rule-firing logic without another breaking change.
    """
    _ = entities  # reserved for future predicate inputs
    q_lower = question.lower()
    resolved: list[ResolvedTerm] = []

    for raw in terms:
        term = raw.lower().strip()
        if not term:
            continue

        candidates: list[CandidateMapping] = []

        direct = _direct_schema_mapping(term)
        if direct is not None:
            candidates.append(direct)

        candidates.extend(_rule_mappings_for(term, q_lower))

        # Deduplicate by (mapped_column, mapped_value) keeping max confidence.
        deduped: dict[tuple[str | None, str | None], CandidateMapping] = {}
        for c in candidates:
            key = (c.mapped_column, c.mapped_value)
            prior = deduped.get(key)
            if prior is None or c.confidence > prior.confidence:
                deduped[key] = c
        merged = sorted(deduped.values(), key=lambda c: c.confidence, reverse=True)

        if not merged:
            resolved.append(ResolvedTerm(
                term=term,
                status="unresolved",
                selected_policy="drop",
            ))
            continue

        if len(merged) == 1:
            sole = merged[0]
            status = (
                "direct"
                if sole.support_basis in {"kb_column", "kb_value"} and sole.confidence >= 0.95
                else "rule"
            )
            resolved.append(ResolvedTerm(
                term=term,
                status=status,
                candidate_mappings=[sole],
                selected_policy="apply",
            ))
            continue

        top2 = merged[:2]
        resolved.append(ResolvedTerm(
            term=term,
            status="ambiguous",
            candidate_mappings=top2,
            selected_policy="branch_top_2",
        ))

    if resolved:
        logger.info(
            "synonym_expander: %d term(s) resolved | %s",
            len(resolved),
            ", ".join(f"{r.term}→{r.status}" for r in resolved),
        )
    return resolved


def kb_columns_supporting(term: str) -> list[str]:
    """Public re-export so ambiguity_resolver can find KB-grounded columns."""
    return _kb_columns_supporting_term(term)
