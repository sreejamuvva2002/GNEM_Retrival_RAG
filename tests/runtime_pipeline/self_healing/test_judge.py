"""Unit tests for the self-healing gate functions and deterministic checks."""
from __future__ import annotations

from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext
from georgia_ev_intelligence.runtime_pipeline.self_healing import judge as J
from georgia_ev_intelligence.runtime_pipeline.self_healing.models import (
    VERDICT_GOOD,
    VERDICT_INSUFFICIENT,
    VERDICT_IRRELEVANT,
)


def _p(rid: str, text: str = "Company: Acme Corp\nRole: cells") -> ParentContext:
    return ParentContext(record_id=rid, source_row_id=1, parent_chunk_text=text)


# --------------------------------------------------------------------------- #
# JSON parsing
# --------------------------------------------------------------------------- #
def test_parse_json_object_handles_fences_embedded_and_invalid() -> None:
    assert J.parse_json_object('```json\n{"verdict":"good"}\n```') == {"verdict": "good"}
    assert J.parse_json_object('noise before {"a": 1} noise after') == {"a": 1}
    assert J.parse_json_object("not json at all") is None
    assert J.parse_json_object("[1, 2, 3]") is None  # arrays are not accepted


def test_format_snippets_numbers_truncates_and_handles_empty() -> None:
    assert J.format_snippets([], 10, 500) == "(no records were retrieved)"
    text = "x" * 1000
    out = J.format_snippets([_p("p1", text)], snippet_count=10, max_chars=20)
    assert out.startswith("[1]")
    assert "…" in out
    # Only the first `snippet_count` parents are shown.
    many = [_p(f"p{i}", f"rec {i}") for i in range(5)]
    out = J.format_snippets(many, snippet_count=2, max_chars=500)
    assert "[1]" in out and "[2]" in out and "[3]" not in out


# --------------------------------------------------------------------------- #
# Decompose
# --------------------------------------------------------------------------- #
def test_decompose_returns_subqueries_when_split() -> None:
    def gen(prompt, timeout=60):
        return '{"subqueries": ["q in Fulton", "q in Cobb"]}'

    assert J.decompose_query("compare", generate_fn=gen, max_subqueries=4) == [
        "q in Fulton",
        "q in Cobb",
    ]


def test_decompose_falls_back_to_original_on_atomic_or_bad_output() -> None:
    assert J.decompose_query("atomic q", generate_fn=lambda *a, **k: "not json") == ["atomic q"]
    assert J.decompose_query("atomic q", generate_fn=lambda *a, **k: '{"subqueries": []}') == [
        "atomic q"
    ]

    def boom(*a, **k):
        raise RuntimeError("model down")

    assert J.decompose_query("atomic q", generate_fn=boom) == ["atomic q"]


def test_decompose_dedupes_and_caps() -> None:
    def gen(prompt, timeout=60):
        return '{"subqueries": ["a", "A", "b", "c", "d", "e"]}'

    out = J.decompose_query("q", generate_fn=gen, max_subqueries=3)
    assert out == ["a", "b", "c"]  # "A" deduped, capped at 3


# --------------------------------------------------------------------------- #
# Gate A — retrieval judge
# --------------------------------------------------------------------------- #
def test_judge_parses_each_verdict() -> None:
    def gen_for(payload):
        return lambda *a, **k: payload

    good = J.judge_retrieval(
        "q", [_p("p1")], generate_fn=gen_for('{"verdict":"good","relevant":true,"sufficient":true,"reason":"ok"}')
    )
    assert good.verdict == VERDICT_GOOD

    insf = J.judge_retrieval(
        "q", [_p("p1")], generate_fn=gen_for('{"verdict":"insufficient","reason":"partial"}')
    )
    assert insf.verdict == VERDICT_INSUFFICIENT

    irr = J.judge_retrieval(
        "q",
        [_p("p1")],
        generate_fn=gen_for('{"verdict":"irrelevant","reason":"drift","suggested_query":"try this"}'),
    )
    assert irr.verdict == VERDICT_IRRELEVANT
    assert irr.suggested_query == "try this"


def test_judge_no_parents_forces_irrelevant_with_query_as_suggestion() -> None:
    verdict = J.judge_retrieval("the original q", [], generate_fn=lambda *a, **k: "unused")
    assert verdict.verdict == VERDICT_IRRELEVANT
    assert verdict.suggested_query == "the original q"


def test_judge_defaults_to_good_on_unparseable_or_invalid_or_error() -> None:
    assert J.judge_retrieval("q", [_p("p1")], generate_fn=lambda *a, **k: "garbage").verdict == VERDICT_GOOD
    assert (
        J.judge_retrieval("q", [_p("p1")], generate_fn=lambda *a, **k: '{"verdict":"weird"}').verdict
        == VERDICT_GOOD
    )

    def boom(*a, **k):
        raise RuntimeError("down")

    assert J.judge_retrieval("q", [_p("p1")], generate_fn=boom).verdict == VERDICT_GOOD


# --------------------------------------------------------------------------- #
# Gate B — groundedness verify
# --------------------------------------------------------------------------- #
def test_verify_grounded_and_ungrounded() -> None:
    grounded = J.verify_groundedness(
        "q", "answer", [_p("p1")], generate_fn=lambda *a, **k: '{"grounded": true, "unsupported_claims": []}'
    )
    assert grounded.grounded is True

    bad = J.verify_groundedness(
        "q",
        "answer",
        [_p("p1")],
        generate_fn=lambda *a, **k: '{"grounded": false, "unsupported_claims": ["X is tier 1"]}',
    )
    assert bad.grounded is False
    assert bad.unsupported_claims == ["X is tier 1"]


def test_verify_defaults_to_grounded_on_failure_or_empty_answer() -> None:
    assert J.verify_groundedness("q", "ans", [_p("p1")], generate_fn=lambda *a, **k: "garbage").grounded
    assert J.verify_groundedness("q", "", [_p("p1")], generate_fn=lambda *a, **k: "unused").grounded

    def boom(*a, **k):
        raise RuntimeError("down")

    assert J.verify_groundedness("q", "ans", [_p("p1")], generate_fn=boom).grounded


def test_verify_grounded_false_when_unsupported_present_even_if_flag_true() -> None:
    result = J.verify_groundedness(
        "q",
        "ans",
        [_p("p1")],
        generate_fn=lambda *a, **k: '{"grounded": true, "unsupported_claims": ["a claim"]}',
    )
    assert result.grounded is False  # unsupported list overrides a stray true flag


# --------------------------------------------------------------------------- #
# Deterministic checks
# --------------------------------------------------------------------------- #
def test_count_consistency_match_mismatch_empty_and_single_fact() -> None:
    ok, _ = J.check_count_consistency("There are 2 suppliers in Georgia.\nAcme | Role: x\nBeta | Role: y")
    assert ok
    ok, msg = J.check_count_consistency("There are 3 suppliers in Georgia.\nAcme | Role: x")
    assert not ok and "3" in msg
    ok, _ = J.check_count_consistency("There are no suppliers in Georgia.\nBased on the provided evidence.")
    assert ok
    ok, msg = J.check_count_consistency("There are no suppliers.\nAcme | Role: x")
    assert not ok  # says none but lists an item
    ok, _ = J.check_count_consistency("Acme Corp: a tier 1 supplier in Fulton County.")
    assert ok  # single-fact opening, no count -> skip


def test_company_grounding_detects_hallucinated_company() -> None:
    parents = [_p("p1", "Company: Acme Corp\nRole: cells")]
    ok, ung = J.check_company_grounding(["Acme Corp"], parents)
    assert ok and ung == []
    ok, ung = J.check_company_grounding(["Ghost Inc"], parents)
    assert not ok and ung == ["Ghost Inc"]
    ok, ung = J.check_company_grounding([], parents)
    assert ok and ung == []


def test_run_deterministic_checks_and_correction_notes() -> None:
    parents = [_p("p1", "Company: Acme Corp\nRole: cells")]
    det = J.run_deterministic_checks(
        "There are 2 suppliers.\nGhost Inc | Role: x", ["Ghost Inc"], parents
    )
    assert not det.ok
    assert det.ungrounded_companies == ["Ghost Inc"]
    # both a count problem and a grounding problem are reported
    assert len(det.problems) == 2

    notes = J.build_correction_notes(det)
    assert "Ghost Inc" in notes
    assert notes.startswith("- ")
