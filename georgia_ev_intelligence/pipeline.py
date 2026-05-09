"""
8-step KB retrieval pipeline — no external databases required.

Steps:
  1. Load KB (cached after first load)
  2. Build schema index (cached)
  3. Match terms from question against KB values
  4. Retrieve matching rows + apply intent (filter / rank / aggregate)
  5. Select & format evidence
  6. Synthesize answer via LLM (1 call)
  7. Assess hallucination risk
  8. Return structured result
"""
from __future__ import annotations
import functools
import pandas as pd
from dataclasses import dataclass, field

from . import kb_loader, schema_index as si, term_matcher, retriever, evidence_selector, synthesizer
from .schema_index import ColumnMeta
from .term_matcher import MatchResult
from .retriever import RetrievalResult


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


# ── Cached singletons ─────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _get_kb() -> pd.DataFrame:
    return kb_loader.load()


@functools.lru_cache(maxsize=1)
def _get_schema() -> dict[str, ColumnMeta]:
    return si.build(_get_kb())


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run(question: str) -> PipelineResult:
    df = _get_kb()
    schema = _get_schema()

    # Step 3: Term matching
    match: MatchResult = term_matcher.match(question, schema)

    # Step 4: Retrieval
    result: RetrievalResult = retriever.retrieve(question, df, schema, match)

    # Step 5: Evidence formatting
    _, evidence_strings = evidence_selector.select(result.rows)

    # Step 6-7: Synthesis
    answer, risk = synthesizer.synthesize(question, evidence_strings)

    # Derive retrieval method label
    types = set(match.match_types.values())
    if "exact" in types and len(types) == 1:
        method = "Exact keyword match"
    elif "component" in types:
        method = "Partial keyword match + component extraction"
    elif types:
        method = "Mixed: " + " + ".join(sorted(types))
    else:
        method = "No match — full KB fallback"

    return PipelineResult(
        question=question,
        key_terms_matched=list(match.match_types.keys()),
        filters_applied=result.filters_applied,
        missing_terms=match.unmatched_words,
        retrieval_method=method,
        evidence_rows=result.rows.drop(columns=["_row_id"], errors="ignore").to_dict(orient="records"),
        evidence_count=len(result.rows),
        intent=result.intent,
        answer=answer,
        hallucination_risk=risk,
        support_level=result.support_level,
    )
