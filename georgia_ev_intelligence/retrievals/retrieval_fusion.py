"""
Phase 4 — Candidate merging, score fusion, and evidence selection.

This module is the post-retrieval, pre-answer brain. It runs after every
retriever has produced its candidates and after the evidence_validator
has rejected anything that fails hard filters. Three responsibilities:

  1. merge()  — deduplicate candidates by canonical company identity,
                accumulating per-source scores and source provenance.

  2. fuse()   — apply override rules (SQL is authoritative for aggregate /
                count / rank / top_n / risk classes), normalise per-source
                scores within the candidate pool, compute the weighted
                fused score per §D of the plan.

  3. select() — apply per-class evidence policy (top-N by fused score,
                full validated SQL output, etc.) and mark `final_selected`.

Reranker invocation is also gated here via `apply_reranker_if_needed` —
it only runs for query classes where text-level matching is decisive
(PRODUCT_CAPABILITY, FALLBACK_SEMANTIC, AMBIGUOUS_SEMANTIC).
"""
from __future__ import annotations

from typing import Any

from retrievals.reranker import rerank_companies
from core_agent.retrieval_types import (
    ALLOW_VECTOR_ONLY,
    Candidate,
    QueryClass,
    RERANK_ON,
    SQL_AUTHORITATIVE,
)
from shared.logger import get_logger
from shared.metadata_schema import FINAL_SCORE_WEIGHTS

logger = get_logger("retrievals.retrieval_fusion")


# Score weights are owned by shared/metadata_schema.py (single source of
# truth). The mapping below translates a per-source name (used by retrievers
# in this module) to the canonical weight key. Sub-scores absent from a
# candidate are skipped and the remaining weights are renormalised so
# single-source candidates are not unfairly penalised.
#
# 'hybrid' and 'cypher' are not separate weights in the score formula:
#   - 'hybrid' is treated as the dense+sparse pair when only the combined
#      score is reported (e.g. by Qdrant RRF).
#   - 'cypher' is a structured ground-truth source and contributes under the
#      structured-filter weight, same as SQL.
_WEIGHTS: dict[str, float] = {
    "sql":      FINAL_SCORE_WEIGHTS["structured_filter_score"],
    "reranker": FINAL_SCORE_WEIGHTS["reranker_score"],
    "sparse":   FINAL_SCORE_WEIGHTS["bm25_sparse_score"],
    "dense":    FINAL_SCORE_WEIGHTS["dense_vector_score"],
    "synonym":  FINAL_SCORE_WEIGHTS["synonym_mapping_confidence"],
    "metadata": FINAL_SCORE_WEIGHTS["metadata_match_score"],
}


def _candidate_key(c: Candidate) -> str:
    """Stable identity key used for deduplication."""
    if c.company_row_id is not None:
        return f"id:{c.company_row_id}"
    if c.source_row_hash:
        return f"hash:{c.source_row_hash}"
    return f"name:{(c.canonical_name or '').strip().lower()}"


# ── Merge ────────────────────────────────────────────────────────────────────

def merge(per_source_results: list[list[Candidate]]) -> list[Candidate]:
    """
    Deduplicate candidates across sources keeping per-source scores.

    Two retrievers contributing the same row (e.g. SQL row + Qdrant chunk
    of the same company) collapse into one Candidate whose `sources` is
    the union and whose `scores` keeps the max per source.

    The candidate that has the most informative `row` wins for the merged
    representation: SQL rows (which include `id`) are preferred over
    Qdrant payload projections.
    """
    by_key: dict[str, Candidate] = {}
    for batch in per_source_results:
        for cand in batch:
            key = _candidate_key(cand)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = cand
                continue
            existing.sources |= cand.sources
            for src, score in cand.scores.items():
                existing.add_source(src, score)
            # Prefer the row that has more KB columns populated (specifically
            # an `id` field — only SQL rows carry that).
            ex_score = sum(1 for v in existing.row.values() if v not in (None, ""))
            cd_score = sum(1 for v in cand.row.values() if v not in (None, ""))
            ex_has_id = bool(existing.row.get("id"))
            cd_has_id = bool(cand.row.get("id"))
            if (cd_has_id and not ex_has_id) or (cd_has_id == ex_has_id and cd_score > ex_score):
                existing.row = cand.row
                existing.company_row_id = existing.company_row_id or cand.company_row_id
                existing.source_row_hash = existing.source_row_hash or cand.source_row_hash
    return list(by_key.values())


# ── Normalisation + fusion ───────────────────────────────────────────────────

def _min_max_normalise(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-9:
        return [1.0 if v > 0 else 0.0 for v in values]
    return [(v - lo) / (hi - lo) for v in values]


def _normalise_scores_in_place(candidates: list[Candidate]) -> None:
    """Min-max normalise each source's scores across the candidate pool."""
    sources_present: set[str] = set()
    for c in candidates:
        sources_present |= set(c.scores.keys())
    for src in sources_present:
        raw = [c.scores.get(src, 0.0) for c in candidates]
        normed = _min_max_normalise(raw)
        for c, n in zip(candidates, normed):
            if src in c.scores:
                c.scores[src] = n


def _fused_score_for(c: Candidate) -> float:
    """
    Apply the §D weighted formula. Missing sub-scores skip their weight,
    and the remaining weights are renormalised so the final score stays
    in [0, 1].
    """
    contributions: list[tuple[float, float]] = []  # (weight, score)
    for source, weight in _WEIGHTS.items():
        if source in c.scores:
            contributions.append((weight, c.scores[source]))
    if "hybrid" in c.scores and "dense" not in c.scores and "sparse" not in c.scores:
        contributions.append((_WEIGHTS["dense"] + _WEIGHTS["sparse"], c.scores["hybrid"]))
    if "cypher" in c.scores and "sql" not in c.scores:
        contributions.append((_WEIGHTS["sql"], c.scores["cypher"]))

    if not contributions:
        return 0.0
    total_weight = sum(w for w, _ in contributions)
    if total_weight < 1e-9:
        return 0.0
    return sum(w * s for w, s in contributions) / total_weight


def fuse(candidates: list[Candidate], query_class: QueryClass) -> list[Candidate]:
    """
    Apply override rules then weighted fusion.

    Override rules (§D):
      - SQL_AUTHORITATIVE classes → discard everything that does not have
        'sql' (or 'cypher') in its sources.
      - All other classes → fuse normally.

    Returns candidates sorted by fused_score descending.
    """
    if not candidates:
        return []

    if query_class in SQL_AUTHORITATIVE:
        kept: list[Candidate] = []
        for c in candidates:
            if "sql" in c.sources or "cypher" in c.sources:
                kept.append(c)
            else:
                c.hard_filter_passed = False
                c.rejection_reason = c.rejection_reason or (
                    f"vector-only candidate not allowed for {query_class.value}"
                )
        candidates = kept

    if query_class == QueryClass.PRODUCT_CAPABILITY:
        # Vector candidates without an SQL row should already have been
        # rejected by evidence_validator, but enforce again here for safety
        # in case the validator was bypassed.
        kept = []
        for c in candidates:
            if c.company_row_id is not None or "sql" in c.sources:
                kept.append(c)
            else:
                c.hard_filter_passed = False
                c.rejection_reason = c.rejection_reason or (
                    "product_capability requires SQL-validated row"
                )
        candidates = kept

    _normalise_scores_in_place(candidates)
    for c in candidates:
        c.fused_score = _fused_score_for(c)
    candidates.sort(key=lambda c: c.fused_score, reverse=True)
    return candidates


# ── Reranker gate ────────────────────────────────────────────────────────────

def apply_reranker_if_needed(
    question: str,
    candidates: list[Candidate],
    query_class: QueryClass,
    max_inputs: int = 60,
) -> list[Candidate]:
    """
    Run the cross-encoder reranker and store its score under the 'reranker'
    source. Skipped for SQL-authoritative classes (numeric / aggregate /
    risk) where the reranker would only add latency.
    """
    if query_class not in RERANK_ON or not candidates:
        return candidates

    # Build flat dicts the existing reranker_companies expects.
    pool = candidates[: max_inputs]
    rows = [c.row | {"_cand_idx": i} for i, c in enumerate(pool)]
    reranked = rerank_companies(question, rows)

    by_idx = {r.get("_cand_idx"): r.get("_reranker_score") for r in reranked}
    for i, c in enumerate(pool):
        score = by_idx.get(i)
        if score is not None:
            c.add_source("reranker", float(score))
    return candidates


# ── Evidence selection ───────────────────────────────────────────────────────

def select(
    candidates: list[Candidate],
    query_class: QueryClass,
    limit: int | None = None,
) -> list[Candidate]:
    """
    Apply the per-class evidence policy. Mutates `final_selected` so the
    audit row records exactly which candidates the answer was generated
    from.
    """
    if not candidates:
        return []

    chosen: list[Candidate]
    if query_class in (QueryClass.EXACT_FILTER, QueryClass.NETWORK):
        # Take all SQL/Cypher-validated rows; sort by employment desc when
        # available so the answer surfaces the largest first.
        rows = [c for c in candidates if c.hard_filter_passed]
        rows.sort(key=lambda c: float(c.row.get("employment") or 0), reverse=True)
        chosen = rows

    elif query_class in (QueryClass.RANK, QueryClass.TOP_N):
        rows = [c for c in candidates if c.hard_filter_passed]
        rows.sort(key=lambda c: float(c.row.get("employment") or 0), reverse=True)
        chosen = rows

    elif query_class == QueryClass.RISK:
        chosen = [c for c in candidates if c.hard_filter_passed]

    elif query_class in (QueryClass.AGGREGATE, QueryClass.COUNT):
        chosen = [c for c in candidates if c.hard_filter_passed]

    else:
        # PRODUCT_CAPABILITY, AMBIGUOUS_SEMANTIC, FALLBACK_SEMANTIC: rely on
        # fused score (which already incorporates rerank when applicable).
        chosen = [c for c in candidates if c.hard_filter_passed]

    for c in chosen:
        c.final_selected = True
    return chosen
