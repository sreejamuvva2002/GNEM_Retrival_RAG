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
import re
from dataclasses import dataclass, field

import pandas as pd

from ...shared import config
from ...shared.data import loader as kb_loader
from ...shared.data import schema as si
from ..query import term_matcher
from ..query import rewriter as query_rewriter
from ..query import kb_term_extractor, operation_detector
from ..retrieval import evidence as evidence_selector
from ..retrieval import rag as rag_retriever
from ..generation import synthesizer
from ...shared.data.schema import ColumnMeta
from ..query.keyword_resolver import resolve_keywords, KeywordResolution
from ..retrieval.semantic import (
    SemanticRetriever,
    build_semantic_retriever,
    retriever_backend_label,
)
from ..reasoning.retriever import apply_intent, _support_level, _detect_intent, _build_and_mask
from ..query.term_matcher import _is_tier_compatible_column


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
    return rag_retriever.build_bm25_index(_get_kb())


# ── Utility helpers ───────────────────────────────────────────────────────────

_EXHAUSTIVE_LIST_RE = re.compile(
    r"\b("
    r"all|every|complete|complete\s+list|full\s+list|entire|"
    r"show\s+all|list\s+all|identify\s+all|provide\s+all|"
    r"how\s+many|count|number\s+of|total"
    r")\b",
    re.IGNORECASE,
)

_ANALYTICAL_INTENTS = {"aggregate_sum", "rank", "count", "spof"}


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        s = str(value).strip()
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _validate_filters_column_compatibility(
    filters: dict[str, list[str]],
    question: str,
) -> tuple[dict[str, list[str]], list[str]]:
    """
    Validate that term_matcher filters are column-compatible.

    For tier-related questions, remove tier-derived filter values from
    non-tier-compatible columns (e.g., EV Supply Chain Role).
    """
    from ..query.term_matcher import _extract_requested_tiers

    warnings: list[str] = []
    requested_tiers = _extract_requested_tiers(question)

    if not requested_tiers:
        return filters, warnings

    cleaned: dict[str, list[str]] = {}
    for col, vals in filters.items():
        if _is_tier_compatible_column(col):
            cleaned[col] = vals
        else:
            # Check if any values in this column were tier-derived
            non_tier_vals = []
            for v in vals:
                v_lower = v.lower()
                is_tier_value = any(
                    re.search(rf"\btier\s*{re.escape(t)}\b", v_lower)
                    for t in requested_tiers
                )
                if is_tier_value:
                    warnings.append(
                        f"removed_tier_filter_from_incompatible_column="
                        f"{col}:{v}"
                    )
                else:
                    non_tier_vals.append(v)

            if non_tier_vals:
                cleaned[col] = non_tier_vals

    return cleaned, warnings


def _merge_filters(
    base: dict[str, list[str]],
    new_filters: dict[str, list[str] | str],
    *,
    df: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    """
    Merge filter dictionaries without overwriting existing values.

    Important:
    - Values in the same column are OR-ed later by _build_col_mask().
    - Columns are AND-ed by _build_and_mask().
    - Metadata-only keys such as "__dataset_scope" are ignored for dataframe filters.
    """
    merged = {col: list(vals) for col, vals in base.items()}

    for col, vals in (new_filters or {}).items():
        if str(col).startswith("__"):
            continue
        if df is not None and col not in df.columns:
            continue

        if isinstance(vals, list):
            val_list = [str(v) for v in vals if str(v).strip()]
        else:
            val_list = [str(vals)] if str(vals).strip() else []

        if not val_list:
            continue

        merged.setdefault(col, [])
        merged[col].extend(val_list)
        merged[col] = _dedupe(merged[col])

    return merged


def _dataframe_filters_only(
    filters: dict[str, str | list[str]],
    df: pd.DataFrame,
) -> dict[str, list[str]]:
    """Keep only filters that can actually be applied to df columns."""
    out: dict[str, list[str]] = {}
    for col, vals in (filters or {}).items():
        if str(col).startswith("__") or col not in df.columns:
            continue
        if isinstance(vals, list):
            out[col] = _dedupe([str(v) for v in vals if str(v).strip()])
        elif str(vals).strip():
            out[col] = [str(vals)]
    return out


def _is_exhaustive_request(question: str, stage2: dict) -> bool:
    return bool(
        stage2.get("requires_exhaustive_retrieval", False)
        or _EXHAUSTIVE_LIST_RE.search(question or "")
    )


def _detect_effective_intent(original_question: str, effective_question: str) -> dict:
    """
    Prefer original user wording for intent detection because rewritten queries can
    lose words such as "highest", "total", "how many", "single point of failure".
    """
    original_intent = _detect_intent(original_question)
    if original_intent.get("type") != "filter":
        return original_intent

    rewritten_intent = _detect_intent(effective_question)
    if rewritten_intent.get("type") != "filter":
        return rewritten_intent

    return original_intent


def _build_deterministic_base(
    df: pd.DataFrame,
    schema: dict[str, ColumnMeta],
    filters: dict[str, list[str]],
    *,
    fallback_to_full: bool,
) -> pd.DataFrame:
    """
    Build a full deterministic dataframe base.

    If filters exist and match rows, use the fully filtered KB.
    If filters do not match:
      - fallback_to_full=True  -> return full KB
      - fallback_to_full=False -> return empty DataFrame
    """
    if not filters:
        return df if fallback_to_full else pd.DataFrame()

    mask = _build_and_mask(df, filters, schema)
    if int(mask.sum()) > 0:
        return df[mask].copy()

    return df if fallback_to_full else pd.DataFrame()


# ── Probe retrieval ───────────────────────────────────────────────────────────

def _run_probe_retrieval(
    probes: list[str],
    explicit_filters: dict[str, str],
    target_columns: list[str],
    df: pd.DataFrame,
    semantic_retriever: SemanticRetriever,
    bm25_index,
) -> pd.DataFrame:
    """
    High-recall multi-probe retrieval.

    For each probe:
      - semantic vector search
      - BM25 search
      - column-targeted search
      - RRF fuse per probe

    Then globally fuse all probe results plus explicit entity hits.
    """
    all_frames: list[pd.DataFrame] = []

    for probe in probes:
        probe_frames: list[pd.DataFrame] = []

        semantic_df = semantic_retriever.search(
            probe,
            top_k=config.PROBE_TOP_K_SEMANTIC,
            threshold=0.0,
        )
        if not semantic_df.empty:
            probe_frames.append(semantic_df)

        bm25_df = rag_retriever.bm25_search(
            probe,
            df,
            bm25_index,
            top_k=config.PROBE_TOP_K_BM25,
        )
        if not bm25_df.empty:
            probe_frames.append(bm25_df)

        if target_columns:
            col_df = rag_retriever.column_targeted_search(
                probe,
                df,
                target_columns,
                top_k=config.PROBE_TOP_K_COLUMN,
            )
            if not col_df.empty:
                probe_frames.append(col_df)

        if probe_frames:
            probe_fused = rag_retriever.rrf_fuse(
                probe_frames,
                k=60,
                top_k=config.PROBE_FUSED_TOP_K,
            )
            if not probe_fused.empty:
                all_frames.append(probe_fused)

    # Exact entity search for detected explicit filters.
    # Do not pass metadata-only filters such as "__dataset_scope".
    explicit_df_filters = _dataframe_filters_only(explicit_filters, df)
    if explicit_df_filters:
        entity_df = rag_retriever.exact_entity_search(df, explicit_df_filters)
        if not entity_df.empty:
            all_frames.append(entity_df)

    if not all_frames:
        return pd.DataFrame()

    return rag_retriever.rrf_fuse(
        all_frames,
        k=60,
        top_k=config.PROBE_FUSED_TOP_K,
    )


# ── Two-stage rewriter orchestrator ──────────────────────────────────────────

def _run_two_stage_rewriter(
    question: str,
    df: pd.DataFrame,
    schema: dict[str, ColumnMeta],
    semantic_retriever: SemanticRetriever,
    bm25_index,
) -> tuple[dict, pd.DataFrame]:
    """
    Run the full two-stage query rewriting flow.

    Returns:
      (stage2_result, candidate_df)

    stage2_result always contains final_rewritten_queries with at least the
    original question as a safe fallback.
    """
    empty_candidates = pd.DataFrame()

    # Stage 1: Semantic probe generation
    stage1 = query_rewriter.stage1_probe_generation(question, schema)
    if stage1 is None:
        return _minimal_fallback(question), empty_candidates

    probes = stage1.get("semantic_probes", [question])
    explicit_f = stage1.get("explicit_filters", {})
    target_cols = stage1.get("target_columns", [])

    # Probe retrieval
    candidates = _run_probe_retrieval(
        probes=probes,
        explicit_filters=explicit_f,
        target_columns=target_cols,
        df=df,
        semantic_retriever=semantic_retriever,
        bm25_index=bm25_index,
    )

    # Weak probe fallback: add original-question semantic vector results.
    if len(candidates) < config.PROBE_MIN_ROWS:
        fallback_semantic = semantic_retriever.search(question, top_k=50, threshold=0.0)
        if not fallback_semantic.empty:
            frames = [f for f in [candidates, fallback_semantic] if not f.empty]
            candidates = rag_retriever.rrf_fuse(
                frames,
                k=60,
                top_k=config.PROBE_FUSED_TOP_K,
            )

    # KB term extraction
    kb_terms = kb_term_extractor.extract(
        candidates,
        schema,
        probes,
        min_frequency=config.KB_TERM_MIN_FREQUENCY,
        top_n=config.KB_TERM_TOP_N,
    )

    # Stage 2: KB-grounded rewrite
    discovered_count = len(kb_terms.get("kb_discovered_terms", []))
    if discovered_count >= config.KB_TERM_MIN_DISCOVERED:
        stage2 = query_rewriter.stage2_kb_grounded_rewrite(
            question,
            schema,
            stage1,
            kb_terms,
            explicit_f,
        )
    else:
        stage2 = None

    if stage2 is None:
        stage2 = query_rewriter.build_fallback_stage2(question, stage1, kb_terms)

    # Scoring & fallback
    probe_score = query_rewriter.score_retrieval(candidates, explicit_f, kb_terms)
    if probe_score.get("weak", False):
        stage2["confidence"] = "low"
        stage2.setdefault("warnings", []).append("weak_retrieval_fallback_activated")

    # Always preserve original question as the first recall anchor if not present.
    queries = [str(q).strip() for q in stage2.get("final_rewritten_queries", []) if str(q).strip()]
    if question not in queries:
        queries = [question] + queries
    stage2["final_rewritten_queries"] = _dedupe(queries)

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
        "requires_exhaustive_retrieval": bool(_EXHAUSTIVE_LIST_RE.search(question or "")),
        "confidence": "low",
        "warnings": ["stage1_probe_generation_failed"],
    }


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run(question: str) -> PipelineResult:
    df = _get_kb()
    schema = _get_schema()
    semantic_retriever = _get_semantic_retriever()
    bm25 = _get_bm25_index()

    # ── Step 0: Deterministic keyword resolution (BEFORE LLM rewriter) ──
    kw_resolution = resolve_keywords(question, schema)

    # ── Step 1: Detect deterministic operation from original question ──
    det_operation = operation_detector.detect_operation(question)

    # ── Step 2: Two-stage query rewriting ──
    stage2, _probe_candidates = _run_two_stage_rewriter(
        question,
        df,
        schema,
        semantic_retriever,
        bm25,
    )

    rewritten_queries = stage2.get("final_rewritten_queries", [question])
    if not rewritten_queries:
        rewritten_queries = [question]

    exhaustive = _is_exhaustive_request(question, stage2)
    # Operation detector can also require exhaustive retrieval
    if det_operation.get("requires_exhaustive_retrieval", False):
        exhaustive = True

    top_k = 100 if exhaustive else config.RAG_TOP_K
    threshold = 0.0 if exhaustive else config.SEMANTIC_THRESHOLD

    # Final retrieval: run rag_retriever.run() for each rewritten query,
    # then fuse all retrieval results.
    result_frames: list[pd.DataFrame] = []
    all_filters: dict[str, list[str]] = {}
    all_unmatched: list[str] = []
    pre_compat_filters: dict[str, list[str]] = {}  # For debug logging

    for q in rewritten_queries:
        match = term_matcher.match(q, schema)
        rag_r = rag_retriever.run(q, df, schema, semantic_retriever, match)

        if not rag_r.accumulated_df.empty:
            result_frames.append(rag_r.accumulated_df)

        all_filters = _merge_filters(all_filters, rag_r.filters_applied, df=df)
        all_unmatched.extend(match.unmatched_words)

    # Snapshot pre-compatibility filters for debug logging
    pre_compat_filters = {col: list(vals) for col, vals in all_filters.items()}

    # ── Column-compatibility validation for filters ──
    all_filters, compat_warnings = _validate_filters_column_compatibility(
        all_filters, question,
    )

    # ── Merge perfect keyword deterministic filters ──
    # Perfect keywords from the keyword resolver take precedence.
    # They represent exact live KB value matches in compatible columns.
    if kw_resolution.has_perfect:
        # Merge perfect keyword filters INTO all_filters.
        # Perfect keywords override any conflicting term_matcher results
        # for the same column.
        for col, vals in kw_resolution.deterministic_filters.items():
            if col in all_filters:
                # Keep only values that are either from the resolver or
                # independently confirmed by term_matcher
                existing = set(v.lower() for v in all_filters[col])
                for v in vals:
                    if v.lower() not in existing:
                        all_filters[col].append(v)
            else:
                all_filters[col] = list(vals)

    # When term_matcher found no filters, apply LLM-identified explicit filters.
    # Keep only filters that correspond to real dataframe columns.
    stage2_ef_raw = stage2.get("explicit_filters", {})
    stage2_ef = _dataframe_filters_only(stage2_ef_raw, df)

    if not all_filters and stage2_ef:
        entity_df = rag_retriever.exact_entity_search(df, stage2_ef)
        if not entity_df.empty:
            result_frames.append(entity_df)
            all_filters = _merge_filters(all_filters, stage2_ef, df=df)

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
    # Operation detector takes priority over LLM-derived intent.
    effective_question = rewritten_queries[0] if rewritten_queries else question
    if det_operation["type"] != "none":
        # Use deterministic operation for intent
        intent_hint = {"type": det_operation["type"]}
        if det_operation.get("direction"):
            intent_hint["direction"] = det_operation["direction"]
    else:
        # Keep original user wording first for intent detection.
        intent_hint = _detect_effective_intent(question, effective_question)

    # Analytical intents must never operate on RRF-capped evidence.
    # Exhaustive list questions should use deterministic filters when filters exist.
    if intent_hint.get("type") in _ANALYTICAL_INTENTS:
        evidence_df = _build_deterministic_base(
            df=df,
            schema=schema,
            filters=all_filters,
            fallback_to_full=True,
        )
    elif exhaustive and all_filters:
        evidence_df = _build_deterministic_base(
            df=df,
            schema=schema,
            filters=all_filters,
            fallback_to_full=False,
        )
        if evidence_df.empty:
            # If deterministic filters fail, fall back to high-recall fused retrieval.
            evidence_df = rag_retriever.rrf_fuse(
                result_frames,
                k=60,
                top_k=max(config.MAX_EVIDENCE_ROWS, top_k),
            )
    else:
        evidence_df = rag_retriever.rrf_fuse(
            result_frames,
            k=60,
            top_k=max(config.MAX_EVIDENCE_ROWS, top_k) if exhaustive else config.MAX_EVIDENCE_ROWS,
        )

    if evidence_df.empty:
        evidence_df = result_frames[0] if result_frames else pd.DataFrame()

    # Apply intent transformation.
    # For analytical paths, evidence_df is full deterministic/full KB base.
    # For normal filter/list paths, evidence_df is fused evidence.
    base = evidence_df if not evidence_df.empty else df
    result_df, intent = apply_intent(
        filtered=base,
        question=question,
        full_df=df,
        intent=intent_hint,
    )

    total_matched = len(evidence_df)
    missing_terms = stage2.get("warnings", [])
    support = _support_level(total_matched, missing_terms, all_filters)

    clean_df = result_df.drop(columns=["_score", "_row_id"], errors="ignore")

    # Evidence formatting may still cap rows depending on evidence_selector.py.
    # For analytical results, apply_intent should already reduce to final rows.
    _, evidence_strings = evidence_selector.select(clean_df)

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
