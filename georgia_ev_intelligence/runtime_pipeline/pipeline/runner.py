"""
Two-stage KB-grounded RAG pipeline.

Steps:
  1. Load KB, schema, semantic retriever (all cached after first call)
  2. Stage 1 — local LLM generates 5-10 semantic probes from question + schema metadata
  3. Probe retrieval — semantic vector + BM25 + column-targeted + exact entity search per probe;
     RRF fusion → 100-150 high-recall candidate rows
  4. KB term extraction — dynamically discover vocabulary from candidate rows
  5. Stage 2 — local LLM rewrites the question using only discovered KB terms
  6. Scoring & fallback — low-recall signals trigger safe fallback queries
  7. Final retrieval — term_matcher + rag_retriever.run() for each rewritten query;
     RRF fusion across queries
  8. Intent transformation, evidence formatting, synthesis
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field

import pandas as pd

from ...shared import config
from ...shared.data import loader as kb_loader
from ...shared.data import schema as si
from ...shared.data.schema import ColumnMeta

# ── Controller imports (clean facades) ───────────────────────────────────────
from ..query.controller import (
    analyze_query,
    match_terms,
    run_two_stage_rewrite,
    validate_filters_column_compatibility,
    merge_filters,
    dataframe_filters_only,
    is_exhaustive_request,
    detect_effective_intent,
    build_deterministic_base,
    ANALYTICAL_INTENTS,
)
from ..retrieval.controller import (
    SemanticRetriever,
    build_semantic_retriever,
    retriever_backend_label,
    fuse_results,
    exact_entity_search,
    retrieve_with_match,
    build_bm25_index,
    select_evidence,
)
from ..reasoning.controller import (
    apply_intent,
    support_level,
)
from ..generation import synthesizer


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
    unmatched_words: list[str] = field(default_factory=list)
    stage2_explicit_filters: dict = field(default_factory=dict)
    stage2_target_columns: list[str] = field(default_factory=list)
    stage2_mapped_phrases: list[dict] = field(default_factory=list)
    deterministic_operation: dict = field(default_factory=dict)
    keyword_resolution: dict = field(default_factory=dict)
    debug_info: dict = field(default_factory=dict)


# ── Cached singletons ─────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _get_kb() -> pd.DataFrame:
    return kb_loader.load()


@functools.lru_cache(maxsize=1)
def _get_schema() -> dict[str, ColumnMeta]:
    return si.build(_get_kb())


@functools.lru_cache(maxsize=1)
def _get_semantic_retriever() -> SemanticRetriever:
    return build_semantic_retriever(_get_kb())


@functools.lru_cache(maxsize=1)
def _get_bm25_index():
    """Build BM25 index once per process; returns None if rank-bm25 is not installed."""
    return build_bm25_index(_get_kb())


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run(question: str) -> PipelineResult:
    df = _get_kb()
    schema = _get_schema()
    semantic_retriever = _get_semantic_retriever()
    bm25 = _get_bm25_index()

    # ── Step 0: Deterministic keyword resolution + operation detection ──
    analysis = analyze_query(question, schema)
    kw_resolution = analysis.keyword_resolution
    det_operation = analysis.deterministic_operation

    # ── Step 1: Two-stage query rewriting ──
    stage2, _probe_candidates = run_two_stage_rewrite(
        question,
        df,
        schema,
        semantic_retriever,
        bm25,
    )

    rewritten_queries = stage2.get("final_rewritten_queries", [question])
    if not rewritten_queries:
        rewritten_queries = [question]

    exhaustive = is_exhaustive_request(question, stage2)
    if det_operation.get("requires_exhaustive_retrieval", False):
        exhaustive = True

    top_k = 100 if exhaustive else config.RAG_TOP_K
    threshold = 0.0 if exhaustive else config.SEMANTIC_THRESHOLD

    # ── Step 2: Final retrieval for each rewritten query ──
    result_frames: list[pd.DataFrame] = []
    all_filters: dict[str, list[str]] = {}
    all_unmatched: list[str] = []

    for q in rewritten_queries:
        match = match_terms(q, schema)
        rag_r = retrieve_with_match(q, df, schema, semantic_retriever, match)

        if not rag_r.accumulated_df.empty:
            result_frames.append(rag_r.accumulated_df)

        all_filters = merge_filters(all_filters, rag_r.filters_applied, df=df)
        all_unmatched.extend(match.unmatched_words)

    # Snapshot pre-compatibility filters for debug logging
    pre_compat_filters = {col: list(vals) for col, vals in all_filters.items()}

    # ── Column-compatibility validation for filters ──
    all_filters, compat_warnings = validate_filters_column_compatibility(
        all_filters, question,
    )

    # ── Merge perfect keyword deterministic filters ──
    if kw_resolution.has_perfect:
        for col, vals in kw_resolution.deterministic_filters.items():
            if col in all_filters:
                existing = set(v.lower() for v in all_filters[col])
                for v in vals:
                    if v.lower() not in existing:
                        all_filters[col].append(v)
            else:
                all_filters[col] = list(vals)

    # When term_matcher found no filters, apply LLM-identified explicit filters.
    stage2_ef_raw = stage2.get("explicit_filters", {})
    stage2_ef = dataframe_filters_only(stage2_ef_raw, df)

    if not all_filters and stage2_ef:
        entity_df = exact_entity_search(df, stage2_ef)
        if not entity_df.empty:
            result_frames.append(entity_df)
            all_filters = merge_filters(all_filters, stage2_ef, df=df)

    # Also run semantic vector search at configured threshold for each rewritten query.
    for q in rewritten_queries:
        sem_df = semantic_retriever.search(q, top_k=top_k, threshold=threshold)
        if not sem_df.empty:
            result_frames.append(sem_df.drop(columns=["_score"], errors="ignore"))

    if not result_frames:
        fallback = semantic_retriever.search(question, top_k=config.RAG_TOP_K, threshold=0.0)
        if not fallback.empty:
            result_frames = [fallback.drop(columns=["_score"], errors="ignore")]
        else:
            result_frames = [pd.DataFrame()]

    # ── Intent detection ──
    effective_question = rewritten_queries[0] if rewritten_queries else question
    if det_operation["type"] != "none":
        intent_hint = {"type": det_operation["type"]}
        if det_operation.get("direction"):
            intent_hint["direction"] = det_operation["direction"]
    else:
        intent_hint = detect_effective_intent(question, effective_question)

    # ── Evidence selection ──
    if intent_hint.get("type") in ANALYTICAL_INTENTS:
        evidence_df = build_deterministic_base(
            df=df,
            schema=schema,
            filters=all_filters,
            fallback_to_full=True,
        )
    elif exhaustive and all_filters:
        evidence_df = build_deterministic_base(
            df=df,
            schema=schema,
            filters=all_filters,
            fallback_to_full=False,
        )
        if evidence_df.empty:
            evidence_df = fuse_results(
                result_frames,
                k=60,
                top_k=max(config.MAX_EVIDENCE_ROWS, top_k),
            )
    else:
        evidence_df = fuse_results(
            result_frames,
            k=60,
            top_k=max(config.MAX_EVIDENCE_ROWS, top_k) if exhaustive else config.MAX_EVIDENCE_ROWS,
        )

    if evidence_df.empty:
        evidence_df = result_frames[0] if result_frames else pd.DataFrame()

    # ── Apply intent transformation ──
    base = evidence_df if not evidence_df.empty else df
    result_df, intent = apply_intent(
        filtered=base,
        question=question,
        full_df=df,
        intent=intent_hint,
    )

    total_matched = len(evidence_df)
    missing_terms = stage2.get("warnings", [])
    support = support_level(total_matched, missing_terms, all_filters)

    clean_df = result_df.drop(columns=["_score", "_row_id"], errors="ignore")

    # ── Evidence formatting and synthesis ──
    _, evidence_strings = select_evidence(clean_df)

    answer, risk = synthesizer.synthesize(
        question=question,
        evidence=evidence_strings,
        exhaustive=exhaustive,
    )

    confidence = stage2.get("confidence", "low")
    if all_filters:
        method = f"RAG (Two-Stage + Keyword + {retriever_backend_label()})"
    elif confidence != "low":
        method = f"RAG (Two-Stage Semantic + {retriever_backend_label()})"
    else:
        method = f"RAG (Two-Stage Fallback + {retriever_backend_label()})"

    # ── Build debug info ──
    debug_info = {
        "original_question": question,
        "keyword_resolution": kw_resolution.to_debug_dict(),
        "deterministic_operation": det_operation,
        "stage2_confidence": confidence,
        "requires_exhaustive_retrieval": exhaustive,
        "stage2_final_queries": rewritten_queries,
        "raw_mapped_phrases": stage2.get("mapped_user_phrases", []),
        "term_matcher_filters_before_compat": pre_compat_filters,
        "term_matcher_filters_after_compat": all_filters,
        "compat_filter_warnings": compat_warnings,
        "stage2_warnings": stage2.get("warnings", []),
        "intent_used": intent,
    }

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
        unmatched_words=list(dict.fromkeys(all_unmatched)),
        stage2_explicit_filters=stage2.get("explicit_filters", {}),
        stage2_target_columns=stage2.get("target_columns", []),
        stage2_mapped_phrases=stage2.get("mapped_user_phrases", []),
        deterministic_operation=det_operation,
        keyword_resolution=kw_resolution.to_debug_dict(),
        debug_info=debug_info,
    )
