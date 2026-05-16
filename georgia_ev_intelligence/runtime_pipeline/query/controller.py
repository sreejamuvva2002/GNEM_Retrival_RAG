"""
Query layer controller / facade.

Aggregates query understanding operations into a single entry point:
  - Deterministic keyword resolution
  - Analytical operation detection
  - Two-stage query rewriting
  - Filter validation and merging
  - Exhaustive request detection

Functions previously scattered across runner.py and multiple query submodules
are consolidated here. Consumers import from this controller instead of
reaching into individual query submodules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from ...shared import config
from ...shared.data.schema import ColumnMeta
from ..retrieval.filters import build_and_mask
from ..reasoning.retriever import detect_intent
from . import term_matcher
from . import rewriter as query_rewriter
from . import kb_term_extractor, operation_detector
from .term_matcher import MatchResult, is_tier_compatible_column, extract_requested_tiers
from .keyword_resolver import resolve_keywords, KeywordResolution


# ── Regex constants ──────────────────────────────────────────────────────────

EXHAUSTIVE_LIST_RE = re.compile(
    r"\b("
    r"all|every|complete|complete\s+list|full\s+list|entire|"
    r"show\s+all|list\s+all|identify\s+all|provide\s+all|"
    r"how\s+many|count|number\s+of|total"
    r")\b",
    re.IGNORECASE,
)

ANALYTICAL_INTENTS = {"aggregate_sum", "rank", "count", "spof"}


# ── QueryAnalysis result dataclass ───────────────────────────────────────────

@dataclass
class QueryAnalysis:
    """Result of deterministic query analysis (Stage 0)."""
    keyword_resolution: KeywordResolution
    deterministic_operation: dict
    question: str


# ── Convenience wrappers ─────────────────────────────────────────────────────

def analyze_query(question: str, schema_index: dict[str, ColumnMeta]) -> QueryAnalysis:
    """
    Run deterministic query analysis: keyword resolution + operation detection.
    This runs BEFORE the LLM-based rewriter.
    """
    kw_resolution = resolve_keywords(question, schema_index)
    det_operation = operation_detector.detect_operation(question)
    return QueryAnalysis(
        keyword_resolution=kw_resolution,
        deterministic_operation=det_operation,
        question=question,
    )


def match_terms(question: str, schema_index: dict[str, ColumnMeta]) -> MatchResult:
    """Delegate to term_matcher.match()."""
    return term_matcher.match(question, schema_index)


# ── Utility helpers (moved from runner.py) ───────────────────────────────────

def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        s = str(value).strip()
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def validate_filters_column_compatibility(
    filters: dict[str, list[str]],
    question: str,
) -> tuple[dict[str, list[str]], list[str]]:
    """
    Validate that term_matcher filters are column-compatible.

    For tier-related questions, remove tier-derived filter values from
    non-tier-compatible columns (e.g., EV Supply Chain Role).
    """
    warnings: list[str] = []
    requested_tiers = extract_requested_tiers(question)

    if not requested_tiers:
        return filters, warnings

    cleaned: dict[str, list[str]] = {}
    for col, vals in filters.items():
        if is_tier_compatible_column(col):
            cleaned[col] = vals
        else:
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


def merge_filters(
    base: dict[str, list[str]],
    new_filters: dict[str, list[str] | str],
    *,
    df: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    """
    Merge filter dictionaries without overwriting existing values.

    - Values in the same column are OR-ed later by build_col_mask().
    - Columns are AND-ed by build_and_mask().
    - Metadata-only keys such as "__dataset_scope" are ignored.
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
        merged[col] = dedupe(merged[col])

    return merged


def dataframe_filters_only(
    filters: dict[str, str | list[str]],
    df: pd.DataFrame,
) -> dict[str, list[str]]:
    """Keep only filters that can actually be applied to df columns."""
    out: dict[str, list[str]] = {}
    for col, vals in (filters or {}).items():
        if str(col).startswith("__") or col not in df.columns:
            continue
        if isinstance(vals, list):
            out[col] = dedupe([str(v) for v in vals if str(v).strip()])
        elif str(vals).strip():
            out[col] = [str(vals)]
    return out


def is_exhaustive_request(question: str, stage2: dict) -> bool:
    return bool(
        stage2.get("requires_exhaustive_retrieval", False)
        or EXHAUSTIVE_LIST_RE.search(question or "")
    )


def detect_effective_intent(original_question: str, effective_question: str) -> dict:
    """
    Prefer original user wording for intent detection because rewritten queries can
    lose words such as "highest", "total", "how many", "single point of failure".
    """
    original_intent = detect_intent(original_question)
    if original_intent.get("type") != "filter":
        return original_intent

    rewritten_intent = detect_intent(effective_question)
    if rewritten_intent.get("type") != "filter":
        return rewritten_intent

    return original_intent


def build_deterministic_base(
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

    mask = build_and_mask(df, filters, schema)
    if int(mask.sum()) > 0:
        return df[mask].copy()

    return df if fallback_to_full else pd.DataFrame()


# ── Probe retrieval ──────────────────────────────────────────────────────────

def run_probe_retrieval(
    probes: list[str],
    explicit_filters: dict[str, str],
    target_columns: list[str],
    df: pd.DataFrame,
    semantic_retriever,
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
    from ..retrieval import rag as rag_retriever

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
    explicit_df_filters = dataframe_filters_only(explicit_filters, df)
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

def minimal_fallback(question: str) -> dict:
    return {
        "stage": "kb_grounded_query_rewrite",
        "intent": "other",
        "explicit_filters": {},
        "target_columns": [],
        "mapped_user_phrases": [],
        "final_rewritten_queries": [question],
        "negative_queries_or_terms_to_avoid": [],
        "requires_exhaustive_retrieval": bool(EXHAUSTIVE_LIST_RE.search(question or "")),
        "confidence": "low",
        "warnings": ["stage1_probe_generation_failed"],
    }


def run_two_stage_rewrite(
    question: str,
    df: pd.DataFrame,
    schema: dict[str, ColumnMeta],
    semantic_retriever,
    bm25_index,
) -> tuple[dict, pd.DataFrame]:
    """
    Run the full two-stage query rewriting flow.

    Returns:
      (stage2_result, candidate_df)

    stage2_result always contains final_rewritten_queries with at least the
    original question as a safe fallback.
    """
    from ..retrieval import rag as rag_retriever

    empty_candidates = pd.DataFrame()

    # Stage 1: Semantic probe generation
    stage1 = query_rewriter.stage1_probe_generation(question, schema)
    if stage1 is None:
        return minimal_fallback(question), empty_candidates

    probes = stage1.get("semantic_probes", [question])
    explicit_f = stage1.get("explicit_filters", {})
    target_cols = stage1.get("target_columns", [])

    # Probe retrieval
    candidates = run_probe_retrieval(
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
    stage2["final_rewritten_queries"] = dedupe(queries)

    return stage2, candidates
