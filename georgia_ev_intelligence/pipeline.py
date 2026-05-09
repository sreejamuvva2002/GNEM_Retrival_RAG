"""
ReAct + Dense Retrieval + Query Rewriting pipeline — no external databases required.

Steps:
  1. Load KB (cached after first load)
  2. Build schema index (cached)
  3. Build dense retriever — embeddings built once at startup (cached)
  3a. Detect ambiguous words via term_matcher (fast, no LLM)
  3b. Rewrite question with exact KB terms if ambiguous words found (qwen2.5:14b)
  4. ReAct loop: LLM calls get_schema / filter_kb / semantic_search iteratively
  5. Apply intent transformation (rank / count / aggregate / spof)
  6. Format evidence rows
  7. Synthesize answer via LLM (1 call)
  8. Return structured result
"""
from __future__ import annotations
import functools
import pandas as pd
from dataclasses import dataclass, field

from . import config
from . import (
    kb_loader, schema_index as si,
    term_matcher, query_rewriter,
    react_agent, retriever, evidence_selector, synthesizer,
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
    rewritten_question: str = field(default="")  # empty = no rewriting was needed


# ── Cached singletons ─────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _get_kb() -> pd.DataFrame:
    return kb_loader.load()


@functools.lru_cache(maxsize=1)
def _get_schema() -> dict[str, ColumnMeta]:
    return si.build(_get_kb())


@functools.lru_cache(maxsize=1)
def _get_dense_retriever() -> DenseRetriever:
    """Build once per process — encoding 205 rows takes ~2s on first run."""
    return DenseRetriever(
        df=_get_kb(),
        model_name=config.EMBEDDING_MODEL,
        skip_cols=SKIP_COLUMNS,
    )


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run(question: str) -> PipelineResult:
    df     = _get_kb()
    schema = _get_schema()
    dr     = _get_dense_retriever()

    # Step 3a: Detect ambiguous words (fast — pure string matching, no LLM)
    match = term_matcher.match(question, schema)

    # Step 3b: Rewrite question if ambiguous words found
    #   - Uses qwen2.5:14b locally (temperature=0.0, grounded in live KB schema)
    #   - Falls back to original question on any error
    effective_question = query_rewriter.rewrite(question, schema, match.unmatched_words)

    # Step 4: ReAct agent collects evidence via tool calls using the effective question
    react_result = react_agent.run(effective_question, df, schema, dr)

    accumulated_df  = react_result.accumulated_df
    filters_applied = react_result.filters_applied

    # Step 5: Apply intent transformation (rank / count / aggregate / spof)
    base = accumulated_df if not accumulated_df.empty else df
    result_df, intent = apply_intent(
        filtered=base,
        question=effective_question,
        full_df=df,
        intent=None,
    )

    # Support level
    total_matched = len(accumulated_df)
    support = _support_level(total_matched, match.unmatched_words, filters_applied)

    # Step 6: Format evidence (drop internal columns)
    clean_df = result_df.drop(columns=["_score", "_row_id"], errors="ignore")
    _, evidence_strings = evidence_selector.select(clean_df)

    # Step 7: Synthesis (uses effective_question so the synthesizer sees the grounded question)
    answer, risk = synthesizer.synthesize(effective_question, evidence_strings)

    method = (
        "ReAct + Dense Retrieval"
        if filters_applied
        else "ReAct + Semantic Fallback"
    )

    return PipelineResult(
        question=question,
        key_terms_matched=list(filters_applied.keys()),
        filters_applied=filters_applied,
        missing_terms=match.unmatched_words,
        retrieval_method=method,
        evidence_rows=clean_df.to_dict(orient="records"),
        evidence_count=len(clean_df),
        intent=intent,
        answer=answer,
        hallucination_risk=risk,
        support_level=support,
        rewritten_question=effective_question if effective_question != question else "",
    )
