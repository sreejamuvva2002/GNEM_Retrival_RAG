"""
Phase 4 — Shared dataclasses for the new retrieval pipeline.

These types are imported by every new retrieval module (query_classifier,
synonym_expander, ambiguity_resolver, sql_retriever, cypher_retriever,
qdrant_search, evidence_validator, retrieval_fusion, audit_logger,
formatters, pipeline) so they live in one place to avoid circular imports.

Naming note: this file is intentionally retrieval_types.py, not types.py,
to avoid shadowing the stdlib `types` module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ── Query class enum ─────────────────────────────────────────────────────────

class QueryClass(str, Enum):
    """
    Coarse intent class for a question. Drives retriever selection,
    reranker gating, and evidence-selection policy.

    Values are lowercase strings so they round-trip cleanly through JSON
    audit records and Postgres TEXT columns.
    """
    EXACT_FILTER       = "exact_filter_query"
    AGGREGATE          = "aggregate_query"
    COUNT              = "count_query"
    RANK               = "rank_query"
    TOP_N              = "top_n_query"
    NETWORK            = "network_query"
    PRODUCT_CAPABILITY = "product_capability_query"
    RISK               = "risk_query"
    AMBIGUOUS_SEMANTIC = "ambiguous_semantic_query"
    FALLBACK_SEMANTIC  = "fallback_semantic_query"


# Classes whose authoritative source is SQL — Qdrant is discarded.
SQL_AUTHORITATIVE: set[QueryClass] = {
    QueryClass.EXACT_FILTER,
    QueryClass.AGGREGATE,
    QueryClass.COUNT,
    QueryClass.RANK,
    QueryClass.TOP_N,
    QueryClass.RISK,
}

# Classes that allow vector candidates without a confirmed SQL row.
ALLOW_VECTOR_ONLY: set[QueryClass] = {
    QueryClass.PRODUCT_CAPABILITY,
    QueryClass.FALLBACK_SEMANTIC,
}

# Classes where the cross-encoder reranker runs.
RERANK_ON: set[QueryClass] = {
    QueryClass.PRODUCT_CAPABILITY,
    QueryClass.FALLBACK_SEMANTIC,
    QueryClass.AMBIGUOUS_SEMANTIC,
}


# ── Candidate ────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    """
    One candidate row that may be selected as evidence.

    A candidate is keyed by canonical company identity (company_row_id when
    known, source_row_hash when known, fallback to lower-case canonical_name).
    Multiple retrievers can contribute to the same candidate — their scores
    accumulate in `scores`, their names in `sources`.

    `row` is the canonical gev_companies row when the candidate has been
    matched against Postgres; otherwise it is the raw payload from Qdrant or
    Neo4j. The evidence_validator promotes a candidate to "validated" only
    after it has confirmed `row` against a real gev_companies row.
    """
    canonical_name:  str
    company_row_id:  Optional[int] = None
    source_row_hash: Optional[str] = None
    row:             dict[str, Any] = field(default_factory=dict)

    sources:         set[str] = field(default_factory=set)   # {'sql','dense','sparse','cypher','synonym'}
    scores:          dict[str, float] = field(default_factory=dict)
    fused_score:     float = 0.0

    hard_filter_passed: bool = True
    rejection_reason:   Optional[str] = None
    final_selected:     bool = False

    def add_source(self, source: str, score: float) -> None:
        self.sources.add(source)
        # If two retrievers of the same type see the row, keep the larger score.
        prior = self.scores.get(source, 0.0)
        if score > prior:
            self.scores[source] = score


# ── Plans ────────────────────────────────────────────────────────────────────

@dataclass
class SQLPlan:
    """
    A structured query plan executed by sql_retriever.run_plan.

    Exactly one of `mode` selects the shape:
      - filter            : query_companies(filters)
      - aggregate_county  : aggregate_employment_by_county(tier)
      - count_role        : count_by_role()
      - top_n_employment  : top_companies_by_employment(limit, ...)
      - single_supplier   : get_single_supplier_roles()
      - keyword_products  : keyword_search_products(keywords, tier)
      - full_text         : full_text_search(words, tier)
    """
    mode:    str = "filter"
    filters: dict[str, Any] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    limit:   int = 200
    tier:    Optional[str] = None
    ev_relevant_only: bool = False


@dataclass
class CypherPlan:
    """
    A graph query plan executed by cypher_retriever.run_plan.

    Modes:
      - tier         : query_companies_by_tier(tier)
      - location     : query_companies_by_location(county)
      - oem_network  : query_oem_suppliers(oem_name)
      - single_supplier : query_single_supplier_roles()
      - county_density : query_county_tier_density()
    """
    mode:      str
    tier:      Optional[str] = None
    county:    Optional[str] = None
    oem_name:  Optional[str] = None


@dataclass
class QdrantPlan:
    """
    A Qdrant search plan executed by qdrant_search.

    The semantic_query is used for dense, the keyword_query is used for sparse.
    `payload_filters` is a dict of equality / contains constraints that gets
    translated into a Qdrant Filter object with payload pushdown.
    """
    semantic_query:  str
    keyword_query:   str
    payload_filters: dict[str, Any] = field(default_factory=dict)
    k:               int = 120


# ── Mapping & ambiguity ──────────────────────────────────────────────────────

@dataclass
class CandidateMapping:
    """One possible KB-supported interpretation of an abstract term."""
    mapping:        str          # human-readable: "Employment < 200"
    mapped_column:  Optional[str] = None  # 'employment' / 'tier' / 'industry_group' / ...
    mapped_value:   Optional[str] = None  # raw value or condition (e.g. "< 200")
    support_basis:  str = ""     # 'kb_value' / 'kb_column' / 'rule' / 'heuristic' / 'llm_suggestion'
    confidence:     float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "mapping":       self.mapping,
            "mapped_column": self.mapped_column,
            "mapped_value":  self.mapped_value,
            "support_basis": self.support_basis,
            "confidence":    self.confidence,
        }


@dataclass
class ResolvedTerm:
    """Result of synonym_expander.resolve() for one residual term."""
    term:               str
    status:             str          # 'direct' | 'rule' | 'heuristic' | 'ambiguous' | 'unmapped'
    candidate_mappings: list[CandidateMapping] = field(default_factory=list)
    selected_policy:    str = ""     # 'apply' | 'branch_top_2' | 'drop'

    def as_dict(self) -> dict[str, Any]:
        return {
            "term":               self.term,
            "status":             self.status,
            "candidate_mappings": [m.as_dict() for m in self.candidate_mappings],
            "selected_policy":    self.selected_policy,
        }


# ── Branch ───────────────────────────────────────────────────────────────────

@dataclass
class RetrievalBranch:
    """
    One interpretation branch. When the question is unambiguous a single
    branch (id="A") is created. When ambiguous, two branches (A and B) are
    created and run in parallel; both contribute to the final answer.
    """
    branch_id:           str             # "A" | "B"
    interpreted_meaning: str             # human-readable mapping label
    filters:             dict[str, Any] = field(default_factory=dict)  # additive overlay on Entities
    keyword_query:       str = ""
    semantic_query:      str = ""
    sql_plan:            Optional[SQLPlan] = None
    cypher_plan:         Optional[CypherPlan] = None
    qdrant_plan:         Optional[QdrantPlan] = None

    evidence:            list[Candidate] = field(default_factory=list)
    support_level:       str = ""        # 'strong' | 'partial' | 'weak' | 'none'
    answer_section:      str = ""        # rendered prose for this branch (may be filled by formatter)


# ── Answer verification ──────────────────────────────────────────────────────

@dataclass
class AnswerVerification:
    """Result of evidence_validator.verify_answer()."""
    status:              str          # 'ok' | 'risky' | 'failed'
    hallucination_risk:  int = 0      # count of suspect names + suspect numbers
    missing_names:       list[str] = field(default_factory=list)
    suspect_numbers:     list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status":             self.status,
            "hallucination_risk": self.hallucination_risk,
            "missing_names":      self.missing_names,
            "suspect_numbers":    self.suspect_numbers,
        }


# ── Audit records ────────────────────────────────────────────────────────────

@dataclass
class AuditRecord:
    """Top-level audit row written once per question to gev_retrieval_audit."""
    run_id:                   str
    question:                 str
    query_class:              str
    extracted_entities:       dict[str, Any]
    hard_filters:             dict[str, Any] = field(default_factory=dict)
    ambiguous_terms:          list[dict[str, Any]] = field(default_factory=list)
    selected_interpretations: list[dict[str, Any]] = field(default_factory=list)
    synonym_mappings:         list[dict[str, Any]] = field(default_factory=list)
    retrieval_methods_used:   list[str] = field(default_factory=list)
    sql_query:                Optional[str] = None
    cypher_query:             Optional[str] = None
    qdrant_dense_query:       Optional[str] = None
    qdrant_sparse_query:      Optional[str] = None
    final_evidence:           list[dict[str, Any]] = field(default_factory=list)
    answer_text:              str = ""
    support_level:             str = ""
    hallucination_risk:        int = 0
    audit_comment:             str = ""
    elapsed_ms:                int = 0
