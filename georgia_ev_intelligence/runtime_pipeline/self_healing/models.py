"""Dataclasses for the self-healing loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext


# Allowed retrieval-judge verdicts.
VERDICT_GOOD = "good"
VERDICT_INSUFFICIENT = "insufficient"
VERDICT_IRRELEVANT = "irrelevant"
VALID_VERDICTS = {VERDICT_GOOD, VERDICT_INSUFFICIENT, VERDICT_IRRELEVANT}


@dataclass
class JudgeVerdict:
    """Gate A — pre-generation retrieval grading."""

    verdict: str
    relevant: bool
    sufficient: bool
    reason: str
    suggested_query: str = ""


@dataclass
class VerifyResult:
    """Gate B — post-generation groundedness verification."""

    grounded: bool
    unsupported_claims: List[str] = field(default_factory=list)
    missing_companies: List[str] = field(default_factory=list)


@dataclass
class DeterministicCheck:
    """Free, no-LLM checks: count consistency + company grounding."""

    ok: bool
    problems: List[str] = field(default_factory=list)
    ungrounded_companies: List[str] = field(default_factory=list)


@dataclass
class FinalAnswer:
    """Result of the existing generate + parse + company-filter step,
    produced by a callback into ChatService."""

    answer: str
    used_companies: List[str]
    parse_ok: bool
    filtered_parents: List[ParentContext]


@dataclass
class AttemptRecord:
    """One pass through the loop, for the observable healing trace."""

    attempt: int
    effective_query: str
    subqueries: List[str]
    reranker_top_k: int
    verdict: str
    reason: str
    action: str  # generate | widen | rewrite | regenerate | success | retry | exhausted
    grounded: Optional[bool] = None
    unsupported_count: int = 0
    top_rerank_score: Optional[float] = None


@dataclass
class HealingTrace:
    attempts: List[AttemptRecord] = field(default_factory=list)
    final_outcome: str = ""  # success | best_effort
    llm_calls: int = 0

    def as_dict(self) -> dict:
        return {
            "final_outcome": self.final_outcome,
            "llm_calls": self.llm_calls,
            "attempts": [
                {
                    "attempt": a.attempt,
                    "effective_query": a.effective_query,
                    "subqueries": a.subqueries,
                    "reranker_top_k": a.reranker_top_k,
                    "verdict": a.verdict,
                    "reason": a.reason,
                    "action": a.action,
                    "grounded": a.grounded,
                    "unsupported_count": a.unsupported_count,
                    "top_rerank_score": a.top_rerank_score,
                }
                for a in self.attempts
            ],
        }


@dataclass
class LoopResult:
    """What the loop hands back to ChatService."""

    answer: str
    parent_contexts: List[ParentContext]
    effective_query: str
    warn: str = ""
    healing_trace: HealingTrace = field(default_factory=HealingTrace)
    confidence: Optional[float] = None
