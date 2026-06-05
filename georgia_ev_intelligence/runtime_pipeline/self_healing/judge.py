"""Gate functions for the self-healing loop.

Two LLM gates (retrieval judge, groundedness verify) plus a query decomposer,
and two free deterministic checks (count consistency, company grounding).

This module imports only ``generate_answer`` + prompts + stdlib — never
``chat_service`` — so the loop can be imported from ChatService without a cycle.

Safe-default policy (so a malformed local-model response can't wedge the loop):
- unparseable judge   -> "good"      (avoid false retries)
- unparseable verify  -> grounded    (avoid loops on bad JSON)
"""
from __future__ import annotations

import json
import re
from typing import Callable, List, Optional, Tuple

from georgia_ev_intelligence.runtime_pipeline.debug_trace import record_step
from georgia_ev_intelligence.runtime_pipeline.generation.llm_client import generate_answer
from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext

from .models import (
    VALID_VERDICTS,
    VERDICT_GOOD,
    VERDICT_INSUFFICIENT,
    VERDICT_IRRELEVANT,
    DeterministicCheck,
    JudgeVerdict,
    VerifyResult,
)
from .prompts import (
    DECOMPOSE_PROMPT,
    GROUNDEDNESS_VERIFY_PROMPT,
    RETRIEVAL_JUDGE_PROMPT,
    render,
)

GenerateFn = Callable[..., str]

_JSON_FENCE_PREFIX = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_JSON_FENCE_SUFFIX = re.compile(r"\s*```$")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_COMPANY_LINE = re.compile(r"^\s*Company:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)

# Opening-line count patterns (see PROMPT_TEMPLATE opening-line rules).
_COUNT_THERE_ARE = re.compile(r"\bthere\s+(?:are|is)\s+(?:only\s+)?(\d+)\b", re.IGNORECASE)
_COUNT_NONE = re.compile(r"\bthere\s+(?:are|is)\s+(?:no|none)\b", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# JSON parsing (tolerant — mirrors chat_service._parse_json_response)
# --------------------------------------------------------------------------- #
def parse_json_object(raw: str) -> Optional[dict]:
    text = (raw or "").strip()
    text = _JSON_FENCE_PREFIX.sub("", text)
    text = _JSON_FENCE_SUFFIX.sub("", text)
    payload = _try_load(text)
    if payload is None:
        match = _JSON_OBJECT.search(text)
        if match is not None:
            payload = _try_load(match.group(0))
    return payload


def _try_load(candidate: str) -> Optional[dict]:
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _str_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


# --------------------------------------------------------------------------- #
# Snippet formatting
# --------------------------------------------------------------------------- #
def format_snippets(
    parents: List[ParentContext],
    snippet_count: int,
    max_chars: int,
) -> str:
    if not parents:
        return "(no records were retrieved)"
    lines: List[str] = []
    for index, parent in enumerate(parents[:snippet_count], start=1):
        text = (parent.parent_chunk_text or "").strip()
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars].rstrip() + " …"
        lines.append(f"[{index}]\n{text}")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# Decompose (input layer)
# --------------------------------------------------------------------------- #
def decompose_query(
    query: str,
    *,
    generate_fn: GenerateFn = generate_answer,
    max_subqueries: int = 4,
    timeout: int = 60,
) -> List[str]:
    """Split a multi-part question into atomic sub-queries.

    Returns ``[query]`` for atomic questions or on any failure."""
    prompt = render(DECOMPOSE_PROMPT, QUERY=query, MAX_SUBQUERIES=max_subqueries)
    try:
        raw = generate_fn(prompt, timeout=timeout)
    except Exception:
        return [query]

    payload = parse_json_object(raw or "")
    if payload is None:
        return [query]
    subs = _str_list(payload.get("subqueries"))

    # Dedupe (case-insensitive) preserving order; drop ones equal to the query.
    seen: set[str] = set()
    cleaned: List[str] = []
    for sub in subs:
        key = sub.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(sub)
    cleaned = cleaned[:max_subqueries]
    if not cleaned:
        return [query]
    return cleaned


# --------------------------------------------------------------------------- #
# Gate A — retrieval judge
# --------------------------------------------------------------------------- #
def judge_retrieval(
    query: str,
    parents: List[ParentContext],
    *,
    generate_fn: GenerateFn = generate_answer,
    snippet_count: int = 10,
    snippet_chars: int = 500,
    timeout: int = 60,
) -> JudgeVerdict:
    # Nothing retrieved at all -> drift; force a corrective rewrite/re-search.
    if not parents:
        return JudgeVerdict(
            verdict=VERDICT_IRRELEVANT,
            relevant=False,
            sufficient=False,
            reason="No records were retrieved.",
            suggested_query=query,
        )

    snippets = format_snippets(parents, snippet_count, snippet_chars)
    prompt = render(RETRIEVAL_JUDGE_PROMPT, QUERY=query, SNIPPETS=snippets)
    try:
        raw = generate_fn(prompt, timeout=timeout)
    except Exception:
        return _judge_default_good("judge call failed; defaulting to good")

    payload = parse_json_object(raw or "")
    if payload is None:
        return _judge_default_good("judge response unparseable; defaulting to good")

    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in VALID_VERDICTS:
        return _judge_default_good("judge verdict invalid; defaulting to good")

    suggested = str(payload.get("suggested_query", "") or "").strip()
    result = JudgeVerdict(
        verdict=verdict,
        relevant=bool(payload.get("relevant", verdict != VERDICT_IRRELEVANT)),
        sufficient=bool(payload.get("sufficient", verdict == VERDICT_GOOD)),
        reason=str(payload.get("reason", "") or "").strip(),
        suggested_query=suggested,
    )
    record_step(
        "judge",
        status="ok",
        summary=f"verdict={result.verdict}; {result.reason}",
        details={
            "query": query,
            "verdict": result.verdict,
            "relevant": result.relevant,
            "sufficient": result.sufficient,
            "reason": result.reason,
            "suggested_query": result.suggested_query,
            "num_parents_judged": len(parents),
        },
    )
    return result


def _judge_default_good(reason: str) -> JudgeVerdict:
    return JudgeVerdict(
        verdict=VERDICT_GOOD,
        relevant=True,
        sufficient=True,
        reason=reason,
        suggested_query="",
    )


# --------------------------------------------------------------------------- #
# Gate B — groundedness verify (semantic; count/company checked deterministically)
# --------------------------------------------------------------------------- #
def verify_groundedness(
    query: str,
    answer: str,
    parents: List[ParentContext],
    *,
    generate_fn: GenerateFn = generate_answer,
    snippet_count: int = 10,
    snippet_chars: int = 500,
    timeout: int = 60,
) -> VerifyResult:
    if not (answer or "").strip():
        return VerifyResult(grounded=True)

    snippets = format_snippets(parents, snippet_count, snippet_chars)
    prompt = render(
        GROUNDEDNESS_VERIFY_PROMPT, QUERY=query, SNIPPETS=snippets, ANSWER=answer
    )
    try:
        raw = generate_fn(prompt, timeout=timeout)
    except Exception:
        return VerifyResult(grounded=True)  # don't wedge the loop on a failed call

    payload = parse_json_object(raw or "")
    if payload is None:
        return VerifyResult(grounded=True)

    unsupported = _str_list(payload.get("unsupported_claims"))
    missing = _str_list(payload.get("missing_companies"))
    grounded = bool(payload.get("grounded", not unsupported)) and not unsupported
    record_step(
        "verify",
        status="ok" if grounded else "warn",
        summary=(
            f"grounded={grounded}; {len(unsupported)} unsupported claims, "
            f"{len(missing)} missing companies"
        ),
        details={
            "query": query,
            "grounded": grounded,
            "unsupported_claims": unsupported,
            "missing_companies": missing,
        },
    )
    return VerifyResult(
        grounded=grounded,
        unsupported_claims=unsupported,
        missing_companies=missing,
    )


# --------------------------------------------------------------------------- #
# Deterministic checks (free, no LLM)
# --------------------------------------------------------------------------- #
def _normalize_company(value: str) -> str:
    return _NON_ALNUM.sub("", (value or "").lower()).strip()


def check_company_grounding(
    used_companies: List[str],
    parents: List[ParentContext],
) -> Tuple[bool, List[str]]:
    """Every cited company must appear as a ``Company:`` line in some parent.

    Returns (ok, ungrounded_company_names). Mirrors
    chat_service._filter_parent_contexts_by_companies matching."""
    if not used_companies:
        return True, []

    haystacks: List[str] = []
    for parent in parents:
        match = _COMPANY_LINE.search(parent.parent_chunk_text or "")
        if match:
            normalized = _normalize_company(match.group(1))
            if normalized:
                haystacks.append(normalized)

    ungrounded: List[str] = []
    for company in used_companies:
        needle = _normalize_company(company)
        if not needle:
            continue
        if not any(needle in hay or hay in needle for hay in haystacks):
            ungrounded.append(company)
    return (not ungrounded), ungrounded


def _count_body_items(answer: str) -> int:
    """Count item lines in the body — the pipe-delimited '<Company> | ...' lines.
    Group-label lines (no pipe) are excluded, matching the answer format."""
    count = 0
    for line in (answer or "").splitlines()[1:]:  # skip opening line
        if " | " in line:
            count += 1
    return count


def check_count_consistency(answer: str) -> Tuple[bool, Optional[str]]:
    """If the opening line states an explicit count, the body must list that many
    item lines. Conservative: only fires when a numeric/'no' count is present."""
    text = (answer or "").strip()
    if not text:
        return True, None
    opening = text.splitlines()[0]

    stated: Optional[int] = None
    match = _COUNT_THERE_ARE.search(opening)
    if match:
        stated = int(match.group(1))
    elif _COUNT_NONE.search(opening):
        stated = 0
    if stated is None:
        return True, None  # single-fact / no explicit count -> nothing to check

    listed = _count_body_items(text)
    if stated == 0:
        if listed > 0:
            return False, (
                f"Opening line says there are none, but the body lists {listed} "
                "item line(s). Remove the item lines or correct the opening."
            )
        return True, None
    if listed != stated:
        return False, (
            f"Opening line states {stated} matching items but the body lists "
            f"{listed} item line(s). List exactly {stated} items, one per line."
        )
    return True, None


def run_deterministic_checks(
    answer: str,
    used_companies: List[str],
    parents: List[ParentContext],
) -> DeterministicCheck:
    problems: List[str] = []
    count_ok, count_problem = check_count_consistency(answer)
    if not count_ok and count_problem:
        problems.append(count_problem)

    companies_ok, ungrounded = check_company_grounding(used_companies, parents)
    if not companies_ok:
        problems.append(
            "These cited companies do not appear in the retrieved records: "
            + ", ".join(ungrounded)
            + ". Only cite companies present in the records."
        )

    check = DeterministicCheck(
        ok=(count_ok and companies_ok),
        problems=problems,
        ungrounded_companies=ungrounded,
    )
    record_step(
        "deterministic_check",
        status="ok" if check.ok else "warn",
        summary=(
            f"ok={check.ok}; count_ok={count_ok}; companies_ok={companies_ok}; "
            f"{len(problems)} problem(s)"
        ),
        details={
            "ok": check.ok,
            "count_ok": count_ok,
            "companies_ok": companies_ok,
            "problems": problems,
            "ungrounded_companies": ungrounded,
            "used_companies": used_companies,
        },
    )
    return check


def build_correction_notes(
    det: DeterministicCheck,
    verify: Optional[VerifyResult] = None,
) -> str:
    notes: List[str] = list(det.problems)
    if verify is not None:
        for claim in verify.unsupported_claims:
            notes.append(f"Unsupported claim: {claim}")
        if verify.missing_companies:
            notes.append(
                "These matching companies were present in the records but omitted: "
                + ", ".join(verify.missing_companies)
            )
    return "\n".join(f"- {note}" for note in notes)
