"""Chat service: orchestrate hybrid retrieval + Ollama generation.

The LLM is asked to return a strict JSON object so we can (a) extract the
formatted answer for display and (b) extract `used_companies`, which we use
to filter the retrieved `parent_contexts` down to just the parents that the
LLM actually cited. If the LLM ignores the JSON contract we fall back to
displaying the raw text and the full reranked top-K.
"""
from __future__ import annotations

import json
import re
from typing import Callable, List, Tuple

from georgia_ev_intelligence.runtime_pipeline.generation.llm_client import generate_answer
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_hybrid_rag import (
    _format_retrieved_context,
)
from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext

from .interfaces import ChatResult, IChatService


PROMPT_TEMPLATE = """You are an analyst answering questions about an EV supply chain knowledge base
for the state of Georgia. Use ONLY the retrieved context below. Do not use
outside knowledge. Do not invent companies, roles, products, OEMs, locations,
employment numbers, or counts that are not present in the context.

Retrieved context:
{retrieved_parent_chunks}

User question:
{user_question}

---

Respond with a SINGLE JSON object (no markdown code fences, no prose before or
after the JSON). The object must have exactly these two keys:

  "answer":         string  — the formatted answer following the style rules below
  "used_companies": array of strings — the exact company names you cited

Style rules for the "answer" string follow these rules exactly.

1. OPENING LINE
   - If the question asks for a list, count, or set of matching items, begin
     with a one-sentence count statement that names what was found.
     Examples of the pattern (do not copy the contents — only the shape):
       "There are N <short restatement of the filter> in Georgia."
       "There is only 1 <short restatement of the filter> in Georgia."
   - If the question asks for a single fact (which company / which county /
     which location / highest / largest / lowest), open with the direct
     answer in one sentence, including the relevant attribute value in
     parentheses or after a colon. Do not add a count statement.
   - If no item in the context matches the question's filter, open with a
     sentence like:
       "There are no <restated filter> in Georgia."
       or
       "No <restated filter> are explicitly identified in the provided
       context."
     Then add one short sentence explaining the conclusion is based on the
     provided evidence. Do not speculate beyond the context.

2. BODY (when listing matching items) — REQUIRED, NOT OPTIONAL
   - The body is mandatory whenever the question asks for a list, count, or set
     of matching items. NEVER return only the opening line. After the opening
     line, you MUST list every matching item, one per line.
   - If the opening line states a count of N items, the body MUST contain
     exactly N item lines — one for each item included in that count. A
     response that states a count but omits the item lines is INVALID.
   - One item per line. No bullets, no numbering, no markdown tables.
   - Format each line as:
       <Company Name> [<Tier>] | <FieldLabel>: <value> | <FieldLabel>: <value>
   - Include the tier in square brackets only when the question is about
     tiers or categories, or when the tier is informative for the answer.
   - Use a pipe character " | " to separate attributes on the same line.
   - The field labels should match the question's framing. Use these short
     labels when applicable: Role, Product, Produce, Employment, OEMs,
     Primary OEM, Primary OEMs, EV Supply Chain Role, Facility Type,
     EV Relevant, Industry Group, Updated Location, Address.
   - Only include attributes that the question asks for, or that the
     question's framing implies are relevant. Do not pad with extra fields.
   - Preserve company names, role values, location strings, and product
     descriptions exactly as they appear in the context (including
     capitalization, punctuation, ampersands, parentheses, and special
     characters).
   - Format employment as a plain integer. If a thousands separator helps
     readability for a single highlighted number in the opening line, use
     a comma; otherwise keep numbers bare.

3. GROUPING
   - If the question implies natural groups (for example, two different
     category values, or items split by tier), introduce each group with
     a short label line, then list its items beneath. Keep groups in the
     order the question presents them.

4. SCOPE AND GROUNDING
   - Every entity, attribute value, and count in your answer must be
     directly supported by the retrieved context. If a value is missing
     from the context for an item you would otherwise list, write "n/a"
     for that field rather than guessing.
   - If the question asks for a count, the count must equal the number of
     distinct items you actually list in the body.
   - If the context contains information that does not match the filter,
     ignore it. Do not mention non-matches.
   - Treat the same company appearing in multiple retrieved chunks as a
     single entity unless the question asks for per-location or per-site
     entries (in which case list each site as its own line).

5. TONE
   - Direct, factual, and concise. No preamble. No closing summary. No
     meta-commentary about the retrieval, the context, or your reasoning.
   - Do not mention "chunks", "retrieval", "the context", "the knowledge
     base", "the data", or how the answer was derived. The only acceptable
     reference is "based on the provided evidence" when explaining a
     no-result outcome.

Rules for the "used_companies" array:
   - Include each company you actually cited or filtered IN within the
     answer body, using the company name exactly as it appears in the
     retrieved context (preserve capitalization, punctuation, ampersands,
     "Inc.", "LLC", etc.).
   - Exclude any company you mention only to reject as a non-match.
   - If the answer concludes "no results" or otherwise cites no companies,
     return an empty list: [].
   - Do not invent companies that are not in the retrieved context.

Final check before you answer: if your "answer" states that there are N
matching items, confirm the body lists all N of them, each on its own line. If
it does not, add the missing item lines before returning the JSON.

Generate the JSON now."""


_JSON_FENCE_PREFIX = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_JSON_FENCE_SUFFIX = re.compile(r"\s*```$")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_COMPANY_LINE = re.compile(r"^\s*Company:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


class ChatService(IChatService):
    """Concrete chat service backed by HybridRetrievalOrchestrator + Ollama."""

    def __init__(
        self,
        retrieval_pipeline_factory: Callable,
        generate_answer_fn: Callable[..., str] = generate_answer,
        llm_timeout_seconds: int = 180,
    ) -> None:
        self._retrieval_pipeline_factory = retrieval_pipeline_factory
        self._generate_answer = generate_answer_fn
        self._llm_timeout_seconds = llm_timeout_seconds
        self._pipeline = None

    def _pipeline_lazy(self):
        if self._pipeline is None:
            self._pipeline = self._retrieval_pipeline_factory()
        return self._pipeline

    def answer(self, query: str, on_step: Callable[[str], None] | None = None) -> ChatResult:
        def _step(name: str) -> None:
            if on_step is not None:
                try:
                    on_step(name)
                except Exception:
                    pass

        query = (query or "").strip()
        if not query:
            return ChatResult(answer="", parent_contexts=[], trace={}, error="Empty question.")

        try:
            pipeline = self._pipeline_lazy()
            _step("retrieval")
            retrieval = pipeline.retrieve_with_sources(query)
            # Deduplication + reranking run inside retrieve_with_sources; surface
            # them as their own completed steps for the progress UI.
            _step("dedup")
            _step("rerank")
        except Exception as exc:
            return ChatResult(answer="", parent_contexts=[], trace={}, error=f"Retrieval failed: {exc}")

        try:
            _step("generation")
            prompt = PROMPT_TEMPLATE.format(
                retrieved_parent_chunks=_format_retrieved_context(retrieval.parent_contexts),
                user_question=query,
            )
            raw_response = self._generate_answer(prompt, timeout=self._llm_timeout_seconds)
        except Exception as exc:
            return ChatResult(
                answer="",
                parent_contexts=retrieval.parent_contexts,
                trace=_trace_to_dict(retrieval.trace),
                error=f"LLM generation failed: {exc}",
            )

        answer_text, used_companies, parse_ok = _parse_json_response(raw_response or "")
        if parse_ok:
            filtered = _filter_parent_contexts_by_companies(
                retrieval.parent_contexts, used_companies
            )
            return ChatResult(
                answer=answer_text,
                parent_contexts=filtered,
                trace=_trace_to_dict(retrieval.trace),
                error="",
                warn="" if used_companies else "The model returned no cited companies.",
            )

        # Fallback: show the raw text and keep the full reranked top-K so the
        # user still has some grounding info, even though we can't filter.
        return ChatResult(
            answer=answer_text,
            parent_contexts=retrieval.parent_contexts,
            trace=_trace_to_dict(retrieval.trace),
            error="",
            warn="Could not parse the model's JSON response — showing all top-K sources as a fallback.",
        )


def _parse_json_response(raw: str) -> Tuple[str, List[str], bool]:
    """Return (answer_text, used_companies, parse_succeeded)."""
    text = (raw or "").strip()
    text = _JSON_FENCE_PREFIX.sub("", text)
    text = _JSON_FENCE_SUFFIX.sub("", text)

    payload = _try_json_object(text)
    if payload is None:
        # Try to salvage a JSON object embedded inside a longer blob.
        match = _JSON_OBJECT.search(text)
        if match is not None:
            payload = _try_json_object(match.group(0))

    if payload is None:
        return raw, [], False

    answer_field = payload.get("answer")
    used_field = payload.get("used_companies", [])

    if not isinstance(answer_field, str):
        return raw, [], False

    if not isinstance(used_field, list):
        used_field = []

    cleaned_used: List[str] = []
    for item in used_field:
        if isinstance(item, str) and item.strip():
            cleaned_used.append(item.strip())

    return answer_field.strip(), cleaned_used, True


def _try_json_object(candidate: str):
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_company(value: str) -> str:
    return _NON_ALNUM.sub("", (value or "").lower()).strip()


def _filter_parent_contexts_by_companies(
    parents: List[ParentContext],
    used_companies: List[str],
) -> List[ParentContext]:
    if not used_companies:
        return []

    needles = {_normalize_company(name) for name in used_companies if name}
    needles.discard("")
    if not needles:
        return []

    kept: List[ParentContext] = []
    seen_keys: set = set()
    for parent in parents:
        match = _COMPANY_LINE.search(parent.parent_chunk_text or "")
        if not match:
            continue
        haystack = _normalize_company(match.group(1))
        if not haystack:
            continue
        if not any((needle and (needle in haystack or haystack in needle)) for needle in needles):
            continue
        # Dedupe by company so the panel doesn't list the same company twice
        # for two different chunks.
        if haystack in seen_keys:
            continue
        seen_keys.add(haystack)
        kept.append(parent)
    return kept


def extract_cited_company_names(parents: List[ParentContext]) -> set:
    """Normalized company names cited across the given parent contexts.

    Used by the map view to restrict markers to only the companies the answer
    actually drew on (the parents are already filtered to `used_companies`).
    """
    names: set = set()
    for parent in parents:
        match = _COMPANY_LINE.search(parent.parent_chunk_text or "")
        if not match:
            continue
        normalized = _normalize_company(match.group(1))
        if normalized:
            names.add(normalized)
    return names


def normalize_company_name(value: str) -> str:
    """Public wrapper around the internal normalizer (for the map filter)."""
    return _normalize_company(value)


def _trace_to_dict(trace) -> dict:
    if trace is None:
        return {}
    return {
        "sparse_child_count": trace.sparse_child_count,
        "dense_child_count": trace.dense_child_count,
        "merged_child_result_count": trace.merged_child_result_count,
        "unique_child_chunk_count": trace.unique_child_chunk_count,
        "unique_parent_id_count": trace.unique_parent_id_count,
        "parent_context_count_before_rerank": trace.parent_context_count_before_rerank,
        "parent_context_count_after_rerank": trace.parent_context_count_after_rerank,
    }
