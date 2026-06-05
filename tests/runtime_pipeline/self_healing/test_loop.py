"""Use-case tests for SelfHealingLoop (UC1-UC10 from the plan).

The loop is fully dependency-injected, so these run with fakes — no Ollama,
no pipeline, no network.
"""
from __future__ import annotations

from typing import List, Optional

from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext
from georgia_ev_intelligence.runtime_pipeline.self_healing.loop import (
    SelfHealingConfig,
    SelfHealingLoop,
)
from georgia_ev_intelligence.runtime_pipeline.self_healing.models import (
    FinalAnswer,
    JudgeVerdict,
    VerifyResult,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
def _p(rid: str, text: str = "Company: Acme Corp\nRole: cells") -> ParentContext:
    return ParentContext(record_id=rid, source_row_id=1, parent_chunk_text=text)


class _FakeTrace:
    def __init__(self, top: Optional[float]) -> None:
        self.top_rerank_score = top


class _FakeRetrieval:
    def __init__(self, parents: List[ParentContext], top: Optional[float] = 0.9) -> None:
        self.parent_contexts = parents
        self.trace = _FakeTrace(top)


def _verdict(kind: str, suggested: str = "") -> JudgeVerdict:
    return JudgeVerdict(
        verdict=kind,
        relevant=kind != "irrelevant",
        sufficient=kind == "good",
        reason=kind,
        suggested_query=suggested,
    )


def _grounded_answer(text: str = "There are 1 supplier in Georgia.\nAcme Corp | Role: cells") -> FinalAnswer:
    return FinalAnswer(answer=text, used_companies=["Acme Corp"], parse_ok=True, filtered_parents=[_p("p1")])


def _actions(result) -> list[str]:
    return [a.action for a in result.healing_trace.attempts]


def _config(**overrides) -> SelfHealingConfig:
    base = dict(
        max_attempts=3,
        base_reranker_top_k=45,
        widen_step=20,
        decompose_enabled=False,
        max_subqueries=4,
        verify_enabled=True,
        snippet_count=10,
        snippet_chars=500,
        regen_max=1,
    )
    base.update(overrides)
    return SelfHealingConfig(**base)


class LoopHarness:
    """Records every injected call so tests can assert on them."""

    def __init__(
        self,
        *,
        judge_results: List[JudgeVerdict],
        verify_results: Optional[List[VerifyResult]] = None,
        final_answers: Optional[List[FinalAnswer]] = None,
        decompose_result: Optional[List[str]] = None,
        retrieval_parents: Optional[List[ParentContext]] = None,
        config: Optional[SelfHealingConfig] = None,
    ) -> None:
        self.retrieve_calls: list[tuple[str, int]] = []
        self.rerank_calls: list[tuple[str, int]] = []
        self.final_calls: list[dict] = []
        self.judge_calls: list[str] = []
        self.verify_calls: list[str] = []
        self.decompose_calls: list[str] = []

        self._judge_results = list(judge_results)
        self._verify_results = list(verify_results or [])
        self._final_answers = list(final_answers or [])
        self._decompose_result = decompose_result
        self._retrieval_parents = (
            retrieval_parents if retrieval_parents is not None else [_p("p1")]
        )
        self._config = config or _config()

    # injected callables -------------------------------------------------- #
    def retrieve_fn(self, query, reranker_top_k):
        self.retrieve_calls.append((query, reranker_top_k))
        return _FakeRetrieval(list(self._retrieval_parents), top=0.91)

    def rerank_fn(self, query, parents, top_k):
        self.rerank_calls.append((query, top_k))
        return [(p, 1.0 - i * 0.01) for i, p in enumerate(parents)]

    def final_answer_fn(self, **kwargs):
        self.final_calls.append(kwargs)
        if self._final_answers:
            return self._final_answers.pop(0)
        return _grounded_answer()

    def judge_fn(self, query, parents, **kwargs):
        self.judge_calls.append(query)
        if self._judge_results:
            return self._judge_results.pop(0)
        return _verdict("good")

    def verify_fn(self, query, answer, parents, **kwargs):
        self.verify_calls.append(query)
        if self._verify_results:
            return self._verify_results.pop(0)
        return VerifyResult(grounded=True)

    def decompose_fn(self, query, **kwargs):
        self.decompose_calls.append(query)
        return list(self._decompose_result) if self._decompose_result else [query]

    def build(self, on_step=None) -> SelfHealingLoop:
        return SelfHealingLoop(
            retrieve_fn=self.retrieve_fn,
            rerank_fn=self.rerank_fn,
            final_answer_fn=self.final_answer_fn,
            config=self._config,
            generate_answer_fn=lambda *a, **k: "",
            decompose_fn=self.decompose_fn,
            judge_fn=self.judge_fn,
            verify_fn=self.verify_fn,
            on_step=on_step,
        )

    def run(self, query="original q"):
        return self.build().run(
            original_query=query, effective_query=query, chat_memory=None
        )


# --------------------------------------------------------------------------- #
# UC1 — normal answerable query: one pass, no retry
# --------------------------------------------------------------------------- #
def test_uc1_happy_path_single_pass() -> None:
    h = LoopHarness(judge_results=[_verdict("good")], final_answers=[_grounded_answer()])
    result = h.run()

    assert result.warn == ""
    assert result.healing_trace.final_outcome == "success"
    assert len(result.healing_trace.attempts) == 1
    assert result.healing_trace.attempts[0].action == "success"
    assert len(h.retrieve_calls) == 1
    assert h.retrieve_calls[0][1] == 45  # base budget
    assert len(h.final_calls) == 1
    assert result.confidence == 0.91


# --------------------------------------------------------------------------- #
# UC2 — insufficient retrieval -> widen reranker_top_k and retry
# --------------------------------------------------------------------------- #
def test_uc2_insufficient_widens_budget() -> None:
    h = LoopHarness(
        judge_results=[_verdict("insufficient"), _verdict("good")],
        final_answers=[_grounded_answer()],
    )
    result = h.run()

    assert result.healing_trace.final_outcome == "success"
    assert [budget for _q, budget in h.retrieve_calls] == [45, 65]
    assert _actions(result) == ["widen", "success"]


# --------------------------------------------------------------------------- #
# UC3 — irrelevant retrieval -> corrective rewrite via suggested_query
# --------------------------------------------------------------------------- #
def test_uc3_irrelevant_rewrites_query() -> None:
    h = LoopHarness(
        judge_results=[_verdict("irrelevant", suggested="tier 1 in Cobb County"), _verdict("good")],
        final_answers=[_grounded_answer()],
    )
    result = h.run(query="suppliers near there")

    assert [q for q, _b in h.retrieve_calls] == ["suppliers near there", "tier 1 in Cobb County"]
    assert result.effective_query == "tier 1 in Cobb County"
    assert _actions(result)[0] == "rewrite"
    assert result.healing_trace.final_outcome == "success"


def test_uc3_irrelevant_without_new_query_widens_instead() -> None:
    # Judge says irrelevant but gives no usable suggested_query -> must not stall.
    h = LoopHarness(
        judge_results=[_verdict("irrelevant", suggested=""), _verdict("good")],
        final_answers=[_grounded_answer()],
    )
    result = h.run(query="q")
    assert [b for _q, b in h.retrieve_calls] == [45, 65]
    assert _actions(result)[0] == "widen"


# --------------------------------------------------------------------------- #
# UC4 — legitimately empty result is terminal (no retries, no fail-safe)
# --------------------------------------------------------------------------- #
def test_uc4_true_empty_is_terminal() -> None:
    empty_answer = FinalAnswer(
        answer="There are no lithium refineries in Georgia.\nBased on the provided evidence.",
        used_companies=[],
        parse_ok=True,
        filtered_parents=[],
    )
    h = LoopHarness(
        judge_results=[_verdict("good")],
        final_answers=[empty_answer],
        retrieval_parents=[_p("p1", "Company: Acme Corp\nRole: cells")],
    )
    result = h.run(query="lithium refineries in Georgia")

    assert result.warn == ""
    assert result.healing_trace.final_outcome == "success"
    assert len(result.healing_trace.attempts) == 1  # terminal on first pass
    assert len(h.retrieve_calls) == 1
    assert "no lithium refineries" in result.answer


# --------------------------------------------------------------------------- #
# UC5 — hallucinated company -> deterministic check fails -> regenerate
# --------------------------------------------------------------------------- #
def test_uc5_hallucinated_company_triggers_regenerate() -> None:
    parents = [_p("p1", "Company: Acme Corp\nRole: cells")]
    hallucinated = FinalAnswer(
        answer="There are 1 supplier in Georgia.\nGhost Inc | Role: cells",
        used_companies=["Ghost Inc"],
        parse_ok=True,
        filtered_parents=[],
    )
    fixed = _grounded_answer()
    h = LoopHarness(
        judge_results=[_verdict("good")],
        final_answers=[hallucinated, fixed],
        retrieval_parents=parents,
    )
    result = h.run()

    assert len(h.final_calls) == 2  # regenerated once
    assert h.final_calls[1]["correction_notes"]  # correction notes were passed
    assert "Ghost Inc" in h.final_calls[1]["correction_notes"]
    assert result.healing_trace.final_outcome == "success"
    assert len(result.healing_trace.attempts) == 1  # regen is within one attempt


# --------------------------------------------------------------------------- #
# UC6 — count/body mismatch -> regenerate
# --------------------------------------------------------------------------- #
def test_uc6_count_mismatch_triggers_regenerate() -> None:
    bad_count = FinalAnswer(
        answer="There are 3 suppliers in Georgia.\nAcme Corp | Role: cells",
        used_companies=["Acme Corp"],
        parse_ok=True,
        filtered_parents=[_p("p1")],
    )
    fixed = _grounded_answer()
    h = LoopHarness(judge_results=[_verdict("good")], final_answers=[bad_count, fixed])
    result = h.run()

    assert len(h.final_calls) == 2
    assert "3" in h.final_calls[1]["correction_notes"]
    assert result.healing_trace.final_outcome == "success"


# --------------------------------------------------------------------------- #
# UC8 — multi-part question -> decompose, retrieve each, union re-rank
# --------------------------------------------------------------------------- #
def test_uc8_decomposition_unions_and_reranks() -> None:
    h = LoopHarness(
        judge_results=[_verdict("good")],
        final_answers=[_grounded_answer()],
        decompose_result=["tier 1 in Fulton", "tier 1 in Cobb"],
        retrieval_parents=[_p("p1"), _p("p2")],
        config=_config(decompose_enabled=True),
    )
    result = h.run(query="compare Fulton vs Cobb tier 1")

    assert h.decompose_calls == ["compare Fulton vs Cobb tier 1"]
    # one retrieval per sub-query in the single attempt
    assert [q for q, _b in h.retrieve_calls] == ["tier 1 in Fulton", "tier 1 in Cobb"]
    # the union is re-ranked once against the original effective query
    assert h.rerank_calls and h.rerank_calls[0][0] == "compare Fulton vs Cobb tier 1"
    assert result.healing_trace.final_outcome == "success"


# --------------------------------------------------------------------------- #
# UC10 — exhaustion -> best-effort answer + low-confidence warning
# --------------------------------------------------------------------------- #
def test_uc10_exhaustion_returns_best_effort_with_warning() -> None:
    never_grounded = [VerifyResult(grounded=False, unsupported_claims=["claim"]) for _ in range(6)]
    h = LoopHarness(
        judge_results=[_verdict("good"), _verdict("good"), _verdict("good")],
        verify_results=never_grounded,
        final_answers=[_grounded_answer() for _ in range(6)],
        config=_config(max_attempts=3, regen_max=0),
    )
    result = h.run()

    assert result.healing_trace.final_outcome == "best_effort"
    assert result.warn != ""
    assert len(result.healing_trace.attempts) == 3
    # bounded: one retrieval per attempt (single sub-query, no regen)
    assert len(h.retrieve_calls) == 3
    assert _actions(result)[-1] == "exhausted"


def test_loop_never_exceeds_max_attempts_retrievals() -> None:
    # Always irrelevant -> every attempt would retry, but attempts are capped.
    h = LoopHarness(
        judge_results=[_verdict("irrelevant", suggested="x"), _verdict("irrelevant", suggested="y"), _verdict("irrelevant", suggested="z")],
        final_answers=[_grounded_answer()],
        config=_config(max_attempts=3),
    )
    result = h.run()
    assert len(h.retrieve_calls) == 3
    assert len(result.healing_trace.attempts) == 3
