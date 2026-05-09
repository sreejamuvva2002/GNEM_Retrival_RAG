"""
Two-stage KB-grounded RAG pipeline.

Steps:
  1. Load KB, schema, dense retriever (all cached after first call)
  2. Stage 1 — DeepSeek generates 5-10 semantic probes from question + schema metadata
  3. Probe retrieval — dense + BM25 + column-targeted + exact entity search per probe;
     RRF fusion → 100-150 high-recall candidate rows
  4. KB term extraction — dynamically discover vocabulary from candidate rows
  5. Stage 2 — DeepSeek rewrites the question using only discovered KB terms
  6. Scoring & fallback — low-recall signals trigger safe fallback queries
  7. Final retrieval — term_matcher + rag_retriever.run() for each rewritten query;
     RRF fusion across queries
  8. Intent transformation, evidence formatting, synthesis (unchanged from prior arch)
"""
from __future__ import annotations

import functools

import pandas as pd
from dataclasses import dataclass, field

from . import config
from . import (
    kb_loader, schema_index as si,
    term_matcher, query_rewriter,
    rag_retriever, kb_term_extractor,
    evidence_selector, synthesizer,
)
from .schema_index import ColumnMeta, SKIP_COLUMNS
from .dense_retriever import DenseRetriever
from .retriever import apply_intent, _support_level


@dataclass
class PipelineResult:
    question: str
    key_terms_matched: list[str]
    filters_applied: dict[str, list[str]]
    missing_terms: list[str]
    retrieval_method: str
    evidence_rows: list[dict]
    evidence_count: int
    intent: dict
    answer: str
    hallucination_risk: str
    support_level: str
    rewritten_question: str = field(default="")
    stage2_confidence: str = field(default="")
    probe_warnings: list[str] = field(default_factory=list)


# ── Cached singletons ─────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _get_kb() -> pd.DataFrame:
    return kb_loader.load()


@functools.lru_cache(maxsize=1)
def _get_schema() -> dict[str, ColumnMeta]:
    return si.build(_get_kb())


@functools.lru_cache(maxsize=1)
def _get_dense_retriever() -> DenseRetriever:
    return DenseRetriever(
        df=_get_kb(),
        model_name=config.EMBEDDING_MODEL,
        skip_cols=SKIP_COLUMNS,
    )


@functools.lru_cache(maxsize=1)
def _get_bm25_index():
    """Build BM25 index once per process; returns None if rank-bm25 not installed."""
    return rag_retriever.build_bm25_index(_get_kb())


# ── Probe retrieval ───────────────────────────────────────────────────────────

def _run_probe_retrieval(
    probes: list[str],
    explicit_filters: dict[str, str],
    target_columns: list[str],
    df: pd.DataFrame,
    dense_retriever: DenseRetriever,
    bm25_index,
) -> pd.DataFrame:
    """
    High-recall multi-probe retrieval.  For each probe run dense + BM25 +
    column-targeted search, fuse with RRF per probe, then globally fuse all
    probes + explicit entity hits into a single candidate set.
    """
    all_frames: list[pd.DataFrame] = []

    for probe in probes:
        probe_frames: list[pd.DataFrame] = []

        dense_df = dense_retriever.search(
            probe, top_k=config.PROBE_TOP_K_DENSE, threshold=0.0
        )
        if not dense_df.empty:
            probe_frames.append(dense_df)

        bm25_df = rag_retriever.bm25_search(
            probe, df, bm25_index, top_k=config.PROBE_TOP_K_BM25
        )
        if not bm25_df.empty:
            probe_frames.append(bm25_df)

        if target_columns:
            col_df = rag_retriever.column_targeted_search(
                probe, df, target_columns, top_k=config.PROBE_TOP_K_COLUMN
            )
            if not col_df.empty:
                probe_frames.append(col_df)

        if probe_frames:
            probe_fused = rag_retriever.rrf_fuse(probe_frames, k=60, top_k=config.PROBE_FUSED_TOP_K)
            all_frames.append(probe_fused)

    # Exact entity search for detected explicit filters
    if explicit_filters:
        entity_df = rag_retriever.exact_entity_search(df, explicit_filters)
        if not entity_df.empty:
            all_frames.append(entity_df)

    if not all_frames:
        return pd.DataFrame()

    # Global RRF across all probes
    candidates = rag_retriever.rrf_fuse(all_frames, k=60, top_k=config.PROBE_FUSED_TOP_K)
    return candidates


# ── Two-stage rewriter orchestrator ──────────────────────────────────────────

def _run_two_stage_rewriter(
    question: str,
    df: pd.DataFrame,
    schema: dict[str, ColumnMeta],
    dense_retriever: DenseRetriever,
    bm25_index,
) -> tuple[dict, pd.DataFrame]:
    """
    Run the full two-stage query rewriting flow.

    Returns (stage2_result, candidate_df) where stage2_result always has
    "final_rewritten_queries" containing at least the original question as
    a safe fallback.
    """
    empty_candidates = pd.DataFrame()

    # ── Stage 1: Semantic Probe Generation ──────────────────────────────────
    stage1 = query_rewriter.stage1_probe_generation(question, schema)
    if stage1 is None:
        # Stage 1 failed — return a minimal stage2 with just the original question
        return _minimal_fallback(question), empty_candidates

    probes         = stage1["semantic_probes"]
    explicit_f     = stage1["explicit_filters"]
    target_cols    = stage1["target_columns"]

    # ── Probe Retrieval ──────────────────────────────────────────────────────
    candidates = _run_probe_retrieval(
        probes, explicit_f, target_cols, df, dense_retriever, bm25_index
    )

    # Weak-retrieval fallback: add original question dense results
    if len(candidates) < config.PROBE_MIN_ROWS:
        fallback_dense = dense_retriever.search(question, top_k=50, threshold=0.0)
        if not fallback_dense.empty and "_row_id" in fallback_dense.columns:
            candidates = rag_retriever.rrf_fuse(
                [candidates, fallback_dense], k=60, top_k=config.PROBE_FUSED_TOP_K
            )

    # ── KB Term Extraction ───────────────────────────────────────────────────
    kb_terms = kb_term_extractor.extract(
        candidates,
        schema,
        probes,
        min_frequency=config.KB_TERM_MIN_FREQUENCY,
        top_n=config.KB_TERM_TOP_N,
    )

    # ── Stage 2: KB-Grounded Rewrite ────────────────────────────────────────
    discovered_count = len(kb_terms.get("kb_discovered_terms", []))
    if discovered_count >= config.KB_TERM_MIN_DISCOVERED:
        stage2 = query_rewriter.stage2_kb_grounded_rewrite(
            question, schema, stage1, kb_terms, explicit_f
        )
    else:
        stage2 = None

    if stage2 is None:
        stage2 = query_rewriter.build_fallback_stage2(question, stage1, kb_terms)

    # ── Scoring & Fallback ───────────────────────────────────────────────────
    probe_score = query_rewriter.score_retrieval(candidates, explicit_f, kb_terms)
    if probe_score["weak"]:
        stage2["confidence"] = "low"
        stage2.setdefault("warnings", []).append("weak_retrieval_fallback_activated")
        queries = stage2["final_rewritten_queries"]
        if question not in queries:
            stage2["final_rewritten_queries"] = [question] + queries

    return stage2, candidates


def _minimal_fallback(question: str) -> dict:
    return {
        "stage": "kb_grounded_query_rewrite",
        "intent": "other",
        "explicit_filters": {},
        "target_columns": [],
        "mapped_user_phrases": [],
        "final_rewritten_queries": [question],
        "negative_queries_or_terms_to_avoid": [],
        "requires_exhaustive_retrieval": False,
        "confidence": "low",
        "warnings": ["stage1_probe_generation_failed"],
    }


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run(question: str) -> PipelineResult:
    df     = _get_kb()
    schema = _get_schema()
    dr     = _get_dense_retriever()
    bm25   = _get_bm25_index()

    # Two-stage query rewriting
    stage2, _probe_candidates = _run_two_stage_rewriter(question, df, schema, dr, bm25)

    rewritten_queries = stage2["final_rewritten_queries"]
    exhaustive        = stage2.get("requires_exhaustive_retrieval", False)
    top_k             = 100 if exhaustive else config.RAG_TOP_K
    threshold         = 0.0  if exhaustive else config.SEMANTIC_THRESHOLD

    # Final retrieval: run rag_retriever.run() for each rewritten query,
    # then RRF-fuse all results together
    result_frames: list[pd.DataFrame] = []
    all_filters:   dict[str, list[str]] = {}

    for q in rewritten_queries:
        match   = term_matcher.match(q, schema)
        rag_r   = rag_retriever.run(q, df, schema, dr, match)
        result_frames.append(rag_r.accumulated_df)
        all_filters.update(rag_r.filters_applied)

    # Also run dense search at configured threshold for each rewritten query
    for q in rewritten_queries:
        sem_df = dr.search(q, top_k=top_k, threshold=threshold)
        if not sem_df.empty:
            result_frames.append(sem_df.drop(columns=["_score"], errors="ignore"))

    if not result_frames:
        # Absolute last resort
        fallback = dr.search(question, top_k=config.RAG_TOP_K, threshold=0.0)
        result_frames = [fallback.drop(columns=["_score"], errors="ignore")]

    evidence_df = rag_retriever.rrf_fuse(
        result_frames, k=60, top_k=config.MAX_EVIDENCE_ROWS
    )
    if evidence_df.empty:
        evidence_df = result_frames[0] if result_frames else pd.DataFrame()

    # Effective question = first rewritten query (best single query for intent/synthesis)
    effective_question = rewritten_queries[0] if rewritten_queries else question

    # Apply intent transformation
    base = evidence_df if not evidence_df.empty else df
    result_df, intent = apply_intent(
        filtered=base,
        question=effective_question,
        full_df=df,
        intent=None,
    )

    total_matched = len(evidence_df)
    missing_terms = stage2.get("warnings", [])
    support = _support_level(total_matched, missing_terms, all_filters)

    clean_df = result_df.drop(columns=["_score", "_row_id"], errors="ignore")
    _, evidence_strings = evidence_selector.select(clean_df)

    answer, risk = synthesizer.synthesize(effective_question, evidence_strings)

    confidence = stage2.get("confidence", "low")
    if all_filters:
        method = "RAG (Two-Stage + Keyword)"
    elif confidence != "low":
        method = "RAG (Two-Stage Semantic)"
    else:
        method = "RAG (Two-Stage Fallback)"

    return PipelineResult(
        question=question,
        key_terms_matched=list(all_filters.keys()),
        filters_applied=all_filters,
        missing_terms=missing_terms,
        retrieval_method=method,
        evidence_rows=clean_df.to_dict(orient="records"),
        evidence_count=len(clean_df),
        intent=intent,
        answer=answer,
        hallucination_risk=risk,
        support_level=support,
        rewritten_question=effective_question if effective_question != question else "",
        stage2_confidence=confidence,
        probe_warnings=missing_terms,
    )
