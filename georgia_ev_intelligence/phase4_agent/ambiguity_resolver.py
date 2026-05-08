"""
Phase 4 — Ambiguity branching.

Composes one or two RetrievalBranch objects from a deterministic Entities
extract, the ResolvedTerm list returned by synonym_expander, and the
QueryClass. The branches are run in parallel by the pipeline; both reach
the answer formatter when there are two.

Single-branch path is the default. Two branches fire only when at least
one ResolvedTerm has selected_policy == "branch_top_2".

The interpreted_meaning string is the human-readable label that the
formatter places at the top of each branch's section in the final answer.
"""
from __future__ import annotations

from typing import Any

from phase4_agent.entity_extractor import Entities
from phase4_agent.retrieval_types import (
    CandidateMapping,
    CypherPlan,
    QdrantPlan,
    QueryClass,
    ResolvedTerm,
    RetrievalBranch,
    SQLPlan,
)
from phase4_agent.synonym_expander import kb_columns_supporting
from shared.metadata_schema import FIELD_TYPES


# ── Mapping → branch overlay translation ─────────────────────────────────────

def _filters_from_mapping(m: CandidateMapping) -> tuple[dict[str, Any], list[str]]:
    """
    Translate a CandidateMapping into:
      - a filters overlay dict to merge into the branch's filters
      - a list of keyword strings to add to keyword_query

    Returns (filters, keywords). Both may be empty.
    """
    filters: dict[str, Any] = {}
    keywords: list[str] = []
    col = (m.mapped_column or "").lower()
    val = m.mapped_value

    if col == "employment" and val:
        v = val.strip()
        if v.startswith("<"):
            try:
                filters["max_employment"] = int(v.lstrip("<= ").strip())
            except ValueError:
                pass
        elif v.startswith(">"):
            try:
                filters["min_employment"] = int(v.lstrip(">= ").strip())
            except ValueError:
                pass
    elif col == "tier" and val:
        # Use as a hint for SQL filter; sql_retriever applies ilike for compound tiers.
        filters["tier"] = val
    elif col == "facility_type" and val:
        filters["facility_type"] = val
    elif col == "industry_group" and val:
        filters["industry_group"] = val
    elif col == "classification_method" and val:
        filters["classification_method"] = val
    elif col == "supplier_affiliation_type" and val:
        filters["supplier_affiliation_type"] = val
    elif col == "ev_battery_relevant" and val:
        filters["ev_battery_relevant"] = val
    elif col == "ev_supply_chain_role" and val:
        # values come in either as a single role or "term1|term2|..."
        for piece in val.split("|"):
            piece = piece.strip()
            if piece:
                keywords.append(piece)
    elif col == "products_services" and val:
        keywords.append(val.strip())

    return filters, keywords


# ── Plan composition per query class ─────────────────────────────────────────

def _build_plans_for_class(
    qclass: QueryClass,
    entities: Entities,
    branch_filters: dict[str, Any],
    extra_keywords: list[str],
) -> tuple[SQLPlan | None, CypherPlan | None, QdrantPlan | None]:
    """Compose SQL/Cypher/Qdrant plans appropriate for the query class."""
    sql_plan: SQLPlan | None = None
    cypher_plan: CypherPlan | None = None
    qdrant_plan: QdrantPlan | None = None

    keyword_terms = list(extra_keywords)
    keyword_terms.extend(entities.product_keywords or [])
    keyword_query = " ".join(dict.fromkeys(t for t in keyword_terms if t))

    semantic_query = keyword_query  # the deterministic version; pipeline can override

    if qclass in (QueryClass.EXACT_FILTER, QueryClass.COUNT):
        sql_plan = SQLPlan(mode="filter", filters=dict(branch_filters), limit=200)

    elif qclass == QueryClass.AGGREGATE:
        sql_plan = SQLPlan(
            mode="aggregate_county",
            tier=branch_filters.get("tier") or entities.tier,
        )

    elif qclass == QueryClass.RANK:
        # Generic ranking — fall back to top-N by employment.
        sql_plan = SQLPlan(
            mode="top_n_employment",
            limit=10,
            tier=branch_filters.get("tier") or entities.tier,
            ev_relevant_only=bool(entities.ev_relevant_filter),
        )

    elif qclass == QueryClass.TOP_N:
        sql_plan = SQLPlan(
            mode="top_n_employment",
            limit=int(getattr(entities, "top_n_limit", 10) or 10),
            tier=branch_filters.get("tier") or entities.tier,
            ev_relevant_only=bool(entities.ev_relevant_filter),
        )

    elif qclass == QueryClass.RISK:
        sql_plan = SQLPlan(mode="single_supplier")

    elif qclass == QueryClass.NETWORK:
        oem = entities.oem or (entities.oem_list[0] if entities.oem_list else None)
        if oem:
            cypher_plan = CypherPlan(mode="oem_network", oem_name=oem)
        # SQL still useful as ground truth for the per-company row data.
        sql_plan = SQLPlan(mode="filter", filters=dict(branch_filters), limit=200)

    elif qclass == QueryClass.PRODUCT_CAPABILITY:
        # SQL keyword search across text columns + Qdrant hybrid for recall.
        if keyword_terms:
            sql_plan = SQLPlan(
                mode="keyword_products",
                keywords=keyword_terms,
                tier=branch_filters.get("tier") or entities.tier,
            )
        qdrant_plan = QdrantPlan(
            semantic_query=semantic_query or " ".join(keyword_terms),
            keyword_query=keyword_query,
            payload_filters=dict(branch_filters),
        )

    elif qclass in (QueryClass.AMBIGUOUS_SEMANTIC, QueryClass.FALLBACK_SEMANTIC):
        # Hybrid Qdrant + SQL filter validation; SQL filter only if structured
        # signal is present in branch_filters (tier/county/etc.).
        if branch_filters:
            sql_plan = SQLPlan(mode="filter", filters=dict(branch_filters), limit=200)
        qdrant_plan = QdrantPlan(
            semantic_query=semantic_query,
            keyword_query=keyword_query,
            payload_filters=dict(branch_filters),
        )

    return sql_plan, cypher_plan, qdrant_plan


# ── Public API ───────────────────────────────────────────────────────────────

def _promote_unresolved_to_ambiguous(resolved_terms: list[ResolvedTerm]) -> list[ResolvedTerm]:
    """
    Convert "unresolved" terms into "ambiguous" terms when the KB itself can
    ground at least one interpretation column for the term.

    For example: question contains "research" and the KB has the term
    appearing in `ev_supply_chain_role` values and `facility_type` values.
    The synonym expander returns "unresolved" because no rule covers
    "research"; here we synthesise candidate mappings pointing at those
    KB columns so ambiguity_resolver can branch over them.

    No mapping is invented: every synthesised candidate's `mapped_column` is
    a real KB column AND the column's distinct values contain `term`. The
    `mapped_value` is the term itself, used as a `contains` filter.
    """
    out: list[ResolvedTerm] = []
    for rt in resolved_terms:
        if rt.status != "unresolved" or rt.candidate_mappings:
            out.append(rt)
            continue
        cols = kb_columns_supporting(rt.term)[:2]
        if not cols:
            out.append(rt)  # genuinely unmapped — keep as-is, contributes no filter
            continue
        synth = [
            CandidateMapping(
                mapping=f"{col} contains {rt.term!r}",
                mapped_column=col,
                mapped_value=rt.term,
                support_basis=f"kb_value:{col}",
                confidence=0.5,
            )
            for col in cols
        ]
        if len(synth) == 1:
            out.append(ResolvedTerm(
                term=rt.term,
                status="ambiguous",
                candidate_mappings=synth,
                selected_policy="apply",
            ))
        else:
            out.append(ResolvedTerm(
                term=rt.term,
                status="ambiguous",
                candidate_mappings=synth,
                selected_policy="branch_top_2",
            ))
    return out


def detect(resolved_terms: list[ResolvedTerm]) -> list[ResolvedTerm]:
    """Return only the terms whose policy is branch_top_2 (i.e. ambiguous)."""
    promoted = _promote_unresolved_to_ambiguous(resolved_terms)
    return [r for r in promoted if r.selected_policy == "branch_top_2"]


def _label_for(mapping: CandidateMapping, term: str) -> str:
    return f"{term} → {mapping.mapping}"


def branches(
    entities: Entities,
    resolved_terms: list[ResolvedTerm],
    qclass: QueryClass,
    question: str,
) -> list[RetrievalBranch]:
    """
    Build 1 or 2 RetrievalBranch objects.

    Branching rule: at most one ambiguous term branches at a time. If
    multiple terms are ambiguous we still produce only 2 branches: the
    primary ambiguous term's top-1 vs top-2 mappings. Other ambiguous
    terms are dropped from the branch overlay (logged as `dropped` in the
    audit row) — branching all combinations would explode and add no
    interpretable value to the user.
    """
    # Promote unresolved-but-KB-supported terms into ambiguous candidates
    # before applying / branching. After this step every term either has
    # candidate_mappings or is genuinely unmapped (no contribution).
    promoted_terms = _promote_unresolved_to_ambiguous(resolved_terms)

    # Apply non-branching resolved terms to the base filter overlay.
    base_filters: dict[str, Any] = {}
    base_keywords: list[str] = []
    for rt in promoted_terms:
        if rt.selected_policy != "apply" or not rt.candidate_mappings:
            continue
        m = rt.candidate_mappings[0]
        f, kw = _filters_from_mapping(m)
        base_filters.update(f)
        base_keywords.extend(kw)

    ambiguous = [r for r in promoted_terms if r.selected_policy == "branch_top_2"]

    if not ambiguous:
        sql_plan, cypher_plan, qdrant_plan = _build_plans_for_class(
            qclass, entities, base_filters, base_keywords,
        )
        return [
            RetrievalBranch(
                branch_id="A",
                interpreted_meaning="default interpretation",
                filters=base_filters,
                keyword_query=" ".join(dict.fromkeys(
                    list(base_keywords) + (entities.product_keywords or [])
                )),
                semantic_query=question,
                sql_plan=sql_plan,
                cypher_plan=cypher_plan,
                qdrant_plan=qdrant_plan,
            )
        ]

    # Use the first ambiguous term as the branching pivot.
    pivot = ambiguous[0]
    out: list[RetrievalBranch] = []
    for i, mapping in enumerate(pivot.candidate_mappings[:2]):
        bid = "A" if i == 0 else "B"
        delta_filters, delta_keywords = _filters_from_mapping(mapping)
        merged_filters = dict(base_filters)
        merged_filters.update(delta_filters)
        merged_keywords = list(base_keywords) + delta_keywords

        sql_plan, cypher_plan, qdrant_plan = _build_plans_for_class(
            qclass, entities, merged_filters, merged_keywords,
        )
        out.append(RetrievalBranch(
            branch_id=bid,
            interpreted_meaning=_label_for(mapping, pivot.term),
            filters=merged_filters,
            keyword_query=" ".join(dict.fromkeys(
                merged_keywords + (entities.product_keywords or [])
            )),
            semantic_query=question,
            sql_plan=sql_plan,
            cypher_plan=cypher_plan,
            qdrant_plan=qdrant_plan,
        ))
    return out
