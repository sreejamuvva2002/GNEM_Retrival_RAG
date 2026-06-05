"""The self-healing loop: retrieve -> judge -> generate -> verify -> bounded retry.

The loop is dependency-injected so it is unit-testable without a real pipeline
or Ollama:

- ``retrieve_fn(query, reranker_top_k) -> HybridRetrievalResult``  widen-able retrieval
- ``rerank_fn(query, parents, top_k) -> list[(ParentContext, float)]``  union re-rank
- ``final_answer_fn(original_query, effective_query, parents, chat_memory,
   correction_notes) -> FinalAnswer``  the existing generate+parse+filter step
- ``generate_answer_fn(prompt, timeout) -> str``  Ollama, used by the gate prompts

Control flow mirrors the approved plan; UC references are in the tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from georgia_ev_intelligence.runtime_pipeline.debug_trace import record_step
from georgia_ev_intelligence.runtime_pipeline.generation.llm_client import generate_answer
from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext
from georgia_ev_intelligence.shared.config import settings

from . import judge as judge_mod
from .models import (
    VERDICT_GOOD,
    VERDICT_INSUFFICIENT,
    VERDICT_IRRELEVANT,
    AttemptRecord,
    FinalAnswer,
    HealingTrace,
    JudgeVerdict,
    LoopResult,
    VerifyResult,
)

RetrieveFn = Callable[..., object]  # (query, reranker_top_k) -> HybridRetrievalResult
RerankFn = Callable[[str, List[ParentContext], int], List[Tuple[ParentContext, float]]]
FinalAnswerFn = Callable[..., FinalAnswer]
GenerateFn = Callable[..., str]
OnStep = Callable[[str], None]


@dataclass(frozen=True)
class SelfHealingConfig:
    max_attempts: int = 3
    base_reranker_top_k: int = 45
    widen_step: int = 20
    decompose_enabled: bool = True
    max_subqueries: int = 4
    verify_enabled: bool = True
    snippet_count: int = 10
    snippet_chars: int = 500
    regen_max: int = 1
    gate_timeout: int = 60

    @classmethod
    def from_settings(cls, base_reranker_top_k: int) -> "SelfHealingConfig":
        return cls(
            max_attempts=settings.SELF_HEALING_MAX_ATTEMPTS,
            base_reranker_top_k=base_reranker_top_k,
            widen_step=settings.SELF_HEALING_WIDEN_STEP,
            decompose_enabled=settings.SELF_HEALING_DECOMPOSE_ENABLED,
            max_subqueries=settings.SELF_HEALING_MAX_SUBQUERIES,
            verify_enabled=settings.SELF_HEALING_VERIFY_ENABLED,
            snippet_count=settings.SELF_HEALING_JUDGE_SNIPPET_COUNT,
            snippet_chars=settings.SELF_HEALING_SNIPPET_CHARS,
            regen_max=settings.SELF_HEALING_REGEN_MAX,
        )


@dataclass
class _Candidate:
    """Best-so-far answer, kept for the best-effort fail-safe."""

    final: FinalAnswer
    parents: List[ParentContext]
    effective_query: str
    unsupported: int
    confidence: Optional[float]


class SelfHealingLoop:
    def __init__(
        self,
        *,
        retrieve_fn: RetrieveFn,
        rerank_fn: RerankFn,
        final_answer_fn: FinalAnswerFn,
        config: SelfHealingConfig,
        generate_answer_fn: GenerateFn = generate_answer,
        decompose_fn: Callable[..., List[str]] = judge_mod.decompose_query,
        judge_fn: Callable[..., JudgeVerdict] = judge_mod.judge_retrieval,
        verify_fn: Callable[..., VerifyResult] = judge_mod.verify_groundedness,
        on_step: Optional[OnStep] = None,
    ) -> None:
        self._retrieve = retrieve_fn
        self._rerank = rerank_fn
        self._final_answer = final_answer_fn
        self._cfg = config
        self._gen = generate_answer_fn
        self._decompose = decompose_fn
        self._judge = judge_fn
        self._verify = verify_fn
        self._on_step = on_step
        self._trace = HealingTrace()

    # ---- public entry ---------------------------------------------------- #
    def run(
        self,
        *,
        original_query: str,
        effective_query: str,
        chat_memory: object = None,
    ) -> LoopResult:
        cfg = self._cfg
        eq = effective_query
        budget = cfg.base_reranker_top_k

        subqueries = [eq]
        if cfg.decompose_enabled:
            self._step("decompose")
            subqueries = self._decompose(
                eq,
                generate_fn=self._gen,
                max_subqueries=cfg.max_subqueries,
                timeout=cfg.gate_timeout,
            )
            self._trace.llm_calls += 1

        best: Optional[_Candidate] = None

        for attempt in range(cfg.max_attempts):
            is_last = attempt == cfg.max_attempts - 1

            self._step("retrieval")
            parents, confidence = self._retrieve_union(subqueries, eq, budget)
            self._step("rerank")

            self._step("judge")
            verdict = self._judge(
                eq,
                parents,
                generate_fn=self._gen,
                snippet_count=cfg.snippet_count,
                snippet_chars=cfg.snippet_chars,
                timeout=cfg.gate_timeout,
            )
            self._trace.llm_calls += 1

            record = AttemptRecord(
                attempt=attempt,
                effective_query=eq,
                subqueries=list(subqueries),
                reranker_top_k=budget,
                verdict=verdict.verdict,
                reason=verdict.reason,
                action="generate",
                top_rerank_score=confidence,
            )

            # Repair BEFORE generating, only if attempts remain.
            if not is_last and verdict.verdict == VERDICT_IRRELEVANT:
                eq, subqueries, budget, record.action = self._repair_irrelevant(
                    verdict, eq, budget
                )
                self._record_attempt(record)
                self._step("retry")
                continue
            if not is_last and verdict.verdict == VERDICT_INSUFFICIENT:
                budget += cfg.widen_step
                record.action = "widen"
                self._record_attempt(record)
                self._step("retry")
                continue

            # verdict == good, or last attempt -> generate + check + verify.
            final, det, verify = self._generate_check_verify(
                original_query, eq, parents, chat_memory
            )
            unsupported = len(det.problems) + len(verify.unsupported_claims)
            grounded = det.ok and verify.grounded
            record.grounded = grounded
            record.unsupported_count = unsupported

            candidate = _Candidate(final, parents, eq, unsupported, confidence)
            if best is None or unsupported < best.unsupported:
                best = candidate

            if grounded:
                record.action = "success"
                self._record_attempt(record)
                self._trace.final_outcome = "success"
                record_step(
                    "final",
                    status="ok",
                    attempt=attempt,
                    summary=f"success on attempt {attempt}; llm_calls={self._trace.llm_calls}",
                    details={"path": "self_healing", "outcome": "success",
                             "llm_calls": self._trace.llm_calls},
                )
                return self._result(candidate, warn="")

            record.action = "retry" if not is_last else "exhausted"
            self._record_attempt(record)
            if not is_last:
                budget += cfg.widen_step
                self._step("retry")
                continue

        # Exhausted — best-effort with a low-confidence warning.
        self._trace.final_outcome = "best_effort"
        assert best is not None  # the last attempt always generates
        record_step(
            "final",
            status="warn",
            summary=(
                f"best_effort after {len(self._trace.attempts)} attempts; "
                f"unsupported={best.unsupported}; llm_calls={self._trace.llm_calls}"
            ),
            details={"path": "self_healing", "outcome": "best_effort",
                     "unsupported": best.unsupported, "llm_calls": self._trace.llm_calls},
        )
        return self._result(
            best,
            warn=(
                "Low confidence: the answer could not be fully verified against the "
                "knowledge base after retries — treat it with caution."
            ),
        )

    # ---- internals ------------------------------------------------------- #
    def _repair_irrelevant(
        self,
        verdict: JudgeVerdict,
        eq: str,
        budget: int,
    ) -> Tuple[str, List[str], int, str]:
        """Corrective rewrite using the judge's suggested_query. Falls back to
        widening if the judge offered no genuinely new query (avoids stalling)."""
        suggested = (verdict.suggested_query or "").strip()
        if suggested and suggested.lower() != eq.lower():
            new_subqueries = [suggested]
            if self._cfg.decompose_enabled:
                self._step("decompose")
                new_subqueries = self._decompose(
                    suggested,
                    generate_fn=self._gen,
                    max_subqueries=self._cfg.max_subqueries,
                    timeout=self._cfg.gate_timeout,
                )
                self._trace.llm_calls += 1
            return suggested, new_subqueries, budget, "rewrite"
        # No usable new query — widen instead so the next attempt differs.
        return eq, [eq], budget + self._cfg.widen_step, "widen"

    def _generate_check_verify(
        self,
        original_query: str,
        effective_query: str,
        parents: List[ParentContext],
        chat_memory: object,
    ) -> Tuple[FinalAnswer, judge_mod.DeterministicCheck, VerifyResult]:
        cfg = self._cfg

        def _gen(notes: str) -> FinalAnswer:
            self._step("generation")
            result = self._final_answer(
                original_query=original_query,
                effective_query=effective_query,
                parents=parents,
                chat_memory=chat_memory,
                correction_notes=notes,
            )
            self._trace.llm_calls += 1
            return result

        def _check(answer: FinalAnswer) -> judge_mod.DeterministicCheck:
            return judge_mod.run_deterministic_checks(
                answer.answer, answer.used_companies, parents
            )

        def _verify(answer: FinalAnswer) -> VerifyResult:
            if not cfg.verify_enabled:
                return VerifyResult(grounded=True)
            self._step("verify")
            result = self._verify(
                effective_query,
                answer.answer,
                parents,
                generate_fn=self._gen,
                snippet_count=cfg.snippet_count,
                snippet_chars=cfg.snippet_chars,
                timeout=cfg.gate_timeout,
            )
            self._trace.llm_calls += 1
            return result

        final = _gen("")
        det = _check(final)
        verify = _verify(final)

        regens = 0
        while (not det.ok or not verify.grounded) and regens < cfg.regen_max:
            notes = judge_mod.build_correction_notes(
                det, None if verify.grounded else verify
            )
            if not notes:
                break
            final = _gen(notes)
            det = _check(final)
            verify = _verify(final)
            regens += 1

        return final, det, verify

    def _retrieve_union(
        self,
        subqueries: List[str],
        effective_query: str,
        budget: int,
    ) -> Tuple[List[ParentContext], Optional[float]]:
        if len(subqueries) <= 1:
            query = subqueries[0] if subqueries else effective_query
            result = self._retrieve(query, reranker_top_k=budget)
            parents = list(getattr(result, "parent_contexts", []) or [])
            confidence = _top_score_from_trace(result)
            return parents, confidence

        # Multiple sub-queries: union (dedupe by record_id), re-rank vs the
        # original effective query so ordering is coherent.
        collected: dict[str, ParentContext] = {}
        for sub in subqueries:
            result = self._retrieve(sub, reranker_top_k=budget)
            for parent in getattr(result, "parent_contexts", []) or []:
                collected.setdefault(parent.record_id, parent)

        if not collected:
            return [], None
        scored = self._rerank(effective_query, list(collected.values()), budget)
        parents = [parent for parent, _score in scored]
        confidence = float(scored[0][1]) if scored else None
        return parents, confidence

    def _result(self, candidate: _Candidate, *, warn: str) -> LoopResult:
        return LoopResult(
            answer=candidate.final.answer,
            parent_contexts=candidate.final.filtered_parents,
            effective_query=candidate.effective_query,
            warn=warn,
            healing_trace=self._trace,
            confidence=candidate.confidence,
        )

    def _record_attempt(self, record: AttemptRecord) -> None:
        """Append the attempt to the healing trace and emit a detailed debug row."""
        self._trace.attempts.append(record)
        record_step(
            "attempt",
            status="ok",
            attempt=record.attempt,
            sub_step=record.action,
            summary=(
                f"verdict={record.verdict} action={record.action} "
                f"grounded={record.grounded} top_k={record.reranker_top_k} "
                f"unsupported={record.unsupported_count}"
            ),
            details={
                "attempt": record.attempt,
                "effective_query": record.effective_query,
                "subqueries": record.subqueries,
                "reranker_top_k": record.reranker_top_k,
                "verdict": record.verdict,
                "reason": record.reason,
                "action": record.action,
                "grounded": record.grounded,
                "unsupported_count": record.unsupported_count,
                "top_rerank_score": record.top_rerank_score,
            },
        )

    def _step(self, name: str) -> None:
        if self._on_step is None:
            return
        try:
            self._on_step(name)
        except Exception:
            pass


def _top_score_from_trace(result: object) -> Optional[float]:
    trace = getattr(result, "trace", None)
    if trace is None:
        return None
    return getattr(trace, "top_rerank_score", None)
