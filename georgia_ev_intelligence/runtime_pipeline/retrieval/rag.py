"""
RAG retrieval utilities.

The existing `run()` function combines single-query keyword filtering with the
shared semantic vector retriever used by the final pipeline retrieval step.

New utilities support the two-stage query rewriter's high-recall probe phase:
  bm25_search()           — BM25 sparse keyword search
  build_bm25_index()      — build a BM25Okapi index once for the full KB
  column_targeted_search() — word-overlap search restricted to target columns
  rrf_fuse()              — Reciprocal Rank Fusion across multiple result frames
  exact_entity_search()   — exact substring match on detected filter columns
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ...shared import config
from ...shared.data.schema import ColumnMeta, SKIP_COLUMNS
from .semantic import SemanticRetriever
from ..query.term_matcher import MatchResult
from ..reasoning.retriever import _build_and_mask, _best_single_filter

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


# ── Original RAGResult + run() (unchanged) ────────────────────────────────────

@dataclass
class RAGResult:
    accumulated_df: pd.DataFrame
    filters_applied: dict[str, list[str]] = field(default_factory=dict)


def run(
    question: str,
    df: pd.DataFrame,
    schema_index: dict[str, ColumnMeta],
    semantic_retriever: SemanticRetriever,
    match: MatchResult,
) -> RAGResult:
    """
    Single-query retrieval: keyword filtering (from term_matcher) + semantic
    vector search. Used for the final retrieval step after rewriting.
    """
    frames: list[pd.DataFrame] = []
    filters_applied: dict[str, list[str]] = {}

    if match.filters:
        and_mask = _build_and_mask(df, match.filters, schema_index)
        if and_mask.sum() > 0:
            frames.append(df[and_mask].copy())
            filters_applied = {col: list(vals) for col, vals in match.filters.items()}
        else:
            fallback_df, fallback_filters = _best_single_filter(
                df, match.filters, schema_index, question
            )
            if not fallback_df.empty:
                frames.append(fallback_df)
                filters_applied = fallback_filters

    semantic_df = semantic_retriever.search(
        question, top_k=config.RAG_TOP_K, threshold=config.SEMANTIC_THRESHOLD
    )
    if not semantic_df.empty:
        frames.append(semantic_df.drop(columns=["_score"], errors="ignore"))

    if not frames:
        fallback = semantic_retriever.search(question, top_k=config.RAG_TOP_K, threshold=0.0)
        return RAGResult(
            accumulated_df=fallback.drop(columns=["_score"], errors="ignore"),
            filters_applied={},
        )

    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["_row_id"])
        .reset_index(drop=True)
    )
    return RAGResult(accumulated_df=combined, filters_applied=filters_applied)


# ── BM25 helpers ──────────────────────────────────────────────────────────────

def _row_to_bm25_text(row: pd.Series) -> str:
    """Convert a KB row to a space-separated token string for BM25."""
    parts: list[str] = []
    for col, val in row.items():
        if col in SKIP_COLUMNS or col == "_score" or pd.isna(val):
            continue
        parts.append(f"{col} {val}")
    return " ".join(parts).lower()


def build_bm25_index(df: pd.DataFrame):
    """
    Build a BM25Okapi index over the full KB DataFrame.
    Returns None if rank-bm25 is not installed (bm25_search will skip silently).
    """
    if not _BM25_AVAILABLE:
        return None
    texts = [_row_to_bm25_text(row) for _, row in df.iterrows()]
    tokenized = [t.split() for t in texts]
    return _BM25Okapi(tokenized)


def bm25_search(
    query: str,
    df: pd.DataFrame,
    bm25_index,
    top_k: int = 50,
) -> pd.DataFrame:
    """
    BM25 keyword search.  Returns a DataFrame with a `_score` column.
    Returns an empty DataFrame if the index is None (rank-bm25 unavailable).
    """
    if bm25_index is None or df.empty:
        return pd.DataFrame()

    tokens = query.lower().split()
    scores = bm25_index.get_scores(tokens)

    n = min(top_k, len(df))
    top_indices = np.argsort(scores)[-n:][::-1]
    top_indices = [int(i) for i in top_indices if scores[i] > 0]

    if not top_indices:
        return pd.DataFrame()

    result = df.iloc[top_indices].copy()
    result["_score"] = [float(scores[i]) for i in top_indices]
    return result.reset_index(drop=True)


# ── Column-targeted search ────────────────────────────────────────────────────

def column_targeted_search(
    query: str,
    df: pd.DataFrame,
    target_columns: list[str],
    top_k: int = 50,
) -> pd.DataFrame:
    """
    Word-overlap search restricted to target_columns.
    Useful when a query should focus on specific KB fields (e.g. product_service).
    """
    valid_cols = [c for c in target_columns if c in df.columns]
    if not valid_cols or df.empty:
        return pd.DataFrame()

    query_words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9\-]*", query.lower()))

    def score_row(row: pd.Series) -> float:
        text_words: set[str] = set()
        for col in valid_cols:
            val = row.get(col)
            if pd.notna(val):
                text_words.update(re.findall(r"[a-zA-Z][a-zA-Z0-9\-]*", str(val).lower()))
        return float(len(query_words & text_words))

    scores = df.apply(score_row, axis=1)
    top_indices = scores.nlargest(top_k).index.tolist()
    top_indices = [i for i in top_indices if scores[i] > 0]

    if not top_indices:
        return pd.DataFrame()

    result = df.loc[top_indices].copy()
    result["_score"] = scores[top_indices].values
    return result.reset_index(drop=True)


# ── Exact entity search ───────────────────────────────────────────────────────

def exact_entity_search(
    df: pd.DataFrame,
    explicit_filters: dict[str, str],
) -> pd.DataFrame:
    """
    Substring match on the detected explicit filter columns.
    Used to ensure that location/company constraints always retrieve
    at least the rows that literally contain those entities.
    """
    if not explicit_filters or df.empty:
        return pd.DataFrame()

    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in explicit_filters.items():
        if col not in df.columns:
            continue
        mask &= df[col].astype(str).str.lower().str.contains(
            str(val).lower(), na=False, regex=False
        )

    result = df[mask].copy()
    return result.reset_index(drop=True)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def rrf_fuse(
    frames: list[pd.DataFrame],
    k: int = 60,
    top_k: int = 150,
) -> pd.DataFrame:
    """
    Reciprocal Rank Fusion over multiple result DataFrames.

    Each frame is assumed to be ordered by descending relevance.
    RRF score for a row at rank r in a list: 1 / (k + r + 1)
    (r is 0-indexed so the top-ranked row gets 1/(k+1))

    Rows are identified by their _row_id column.  Frames without _row_id
    are silently skipped.
    """
    row_scores: dict[str, float] = defaultdict(float)
    row_store:  dict[str, dict]  = {}

    for frame in frames:
        if frame is None or frame.empty or "_row_id" not in frame.columns:
            continue
        for rank, (_, row) in enumerate(frame.iterrows()):
            row_id = str(row["_row_id"])
            row_scores[row_id] += 1.0 / (k + rank + 1)
            if row_id not in row_store:
                row_store[row_id] = row.to_dict()

    if not row_store:
        return pd.DataFrame()

    sorted_ids = sorted(row_scores, key=lambda rid: row_scores[rid], reverse=True)
    top_ids    = sorted_ids[:top_k]

    rows   = [row_store[rid] for rid in top_ids]
    result = pd.DataFrame(rows).reset_index(drop=True)
    # Drop retrieval-internal score column; callers may re-add their own
    result = result.drop(columns=["_score"], errors="ignore")
    return result
