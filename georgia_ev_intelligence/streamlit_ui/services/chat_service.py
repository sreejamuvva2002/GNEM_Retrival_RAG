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
from typing import Callable, Dict, List, Optional, Tuple

from georgia_ev_intelligence.runtime_pipeline.debug_trace import (
    record_step,
    set_effective_query,
)
from georgia_ev_intelligence.runtime_pipeline.generation.llm_client import generate_answer
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.config import RERANKER_TOP_K
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_hybrid_rag import (
    _format_retrieved_context,
)
from georgia_ev_intelligence.runtime_pipeline.schemas import ParentContext
from georgia_ev_intelligence.runtime_pipeline.self_healing.loop import (
    SelfHealingConfig,
    SelfHealingLoop,
)
from georgia_ev_intelligence.runtime_pipeline.self_healing.models import FinalAnswer
from georgia_ev_intelligence.runtime_pipeline.self_healing.prompts import (
    REGENERATION_SUFFIX,
    render,
)
from georgia_ev_intelligence.shared.config import settings

from ..models.chat import ChatMemory
from .interfaces import ChatResult, IChatService


PROMPT_TEMPLATE = """You are an analyst answering questions about an EV supply chain knowledge base
for the state of Georgia.

Use ONLY the retrieved context for factual claims. The recent conversation is
provided only to understand references in the user's latest question. Do not use
conversation history as evidence for facts unless the same fact appears in the
retrieved context. If conversation history conflicts with retrieved context, the
retrieved context wins. Do not use outside knowledge. Do not invent companies,
roles, products, OEMs, locations, employment numbers, or counts that are not
present in the context.

Conversation memory:
{chat_memory}

Original user question:
{user_question}

Standalone retrieval question:
{effective_query}

Retrieved context:
{retrieved_parent_chunks}

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


REWRITE_PROMPT_TEMPLATE = """You rewrite user questions into standalone retrieval queries for a Georgia EV supply chain RAG system.

Use conversation memory only when the current question depends on prior context.
Resolve references, filters, entities, locations, categories, tiers, and constraints
such as "those", "that county", "same category", "near there", "what about them",
or similar follow-up language.

If the current question is already standalone or introduces a new topic, return
the current question unchanged.

Do not answer the question.
Do not add facts that are not implied by the conversation.
Do not include markdown.
Return exactly one standalone search query as plain text.

Conversation summary:
{conversation_summary}

Recent conversation:
{recent_conversation}

Current user question:
{user_question}

Standalone retrieval query:"""


SUMMARY_PROMPT_TEMPLATE = """You maintain a compact conversation summary for a Georgia EV supply chain RAG system.

The summary is used only to resolve follow-up questions. It is not factual
evidence for final answers.

Update the existing summary with the new completed turns. Preserve durable
context useful for future follow-up resolution: companies, counties, cities,
tiers, categories, EV/battery relevance filters, OEMs, constraints, and unresolved
references. Do not add outside facts. Do not write a transcript.

Keep the updated summary under 1200 characters. Return plain text only.

Existing summary:
{existing_summary}

New completed turns:
{new_turns}

Updated summary:"""


_JSON_FENCE_PREFIX = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_JSON_FENCE_SUFFIX = re.compile(r"\s*```$")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_COMPANY_LINE = re.compile(r"^\s*Company:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_REWRITE_LABEL_PREFIX = re.compile(
    r"^(?:standalone\s+(?:retrieval\s+)?query|search\s+query)\s*:\s*",
    re.IGNORECASE,
)
_SUMMARY_LABEL_PREFIX = re.compile(r"^(?:updated\s+)?summary\s*:\s*", re.IGNORECASE)
_BAD_REWRITE_PREFIXES = ("the answer", "based on", "according to", "i found", "there are")
_MAX_SUMMARY_CHARS = 1200


class ChatService(IChatService):
    """Concrete chat service backed by HybridRetrievalOrchestrator + Ollama."""

    def __init__(
        self,
        retrieval_pipeline_factory: Callable,
        generate_answer_fn: Callable[..., str] = generate_answer,
        llm_timeout_seconds: int = 1800,
    ) -> None:
        self._retrieval_pipeline_factory = retrieval_pipeline_factory
        self._generate_answer = generate_answer_fn
        self._llm_timeout_seconds = llm_timeout_seconds
        self._pipeline = None

    def _pipeline_lazy(self):
        if self._pipeline is None:
            self._pipeline = self._retrieval_pipeline_factory()
        return self._pipeline

    def _rewrite_query_for_retrieval(
        self,
        query: str,
        chat_memory: ChatMemory | list[dict[str, str]] | None,
    ) -> tuple[str, bool]:
        memory = _coerce_chat_memory(chat_memory)
        conversation_summary = _clean_memory_summary(memory.summary)
        recent_conversation = _format_chat_messages(memory.recent_messages)
        if not conversation_summary and not recent_conversation:
            return query, False

        try:
            rewrite_prompt = REWRITE_PROMPT_TEMPLATE.format(
                conversation_summary=conversation_summary or "None",
                recent_conversation=recent_conversation or "None",
                user_question=query,
            )
            raw = self._generate_answer(
                rewrite_prompt,
                timeout=min(self._llm_timeout_seconds, 60),
            )
        except Exception:
            return query, False

        rewritten = _clean_rewrite_response(raw or "")
        if not rewritten or _is_bad_rewrite(rewritten):
            return query, False

        if _normalize_query_for_compare(rewritten) == _normalize_query_for_compare(query):
            return query, False

        return rewritten, True

    def answer(
        self,
        query: str,
        chat_memory: Optional[ChatMemory] = None,
        on_step: Callable[[str], None] | None = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> ChatResult:
        def _step(name: str) -> None:
            if on_step is not None:
                try:
                    on_step(name)
                except Exception:
                    pass

        query = (query or "").strip()
        if not query:
            record_step(
                "error",
                status="error",
                summary="Empty question — nothing to answer.",
                error="Empty question.",
            )
            return ChatResult(
                answer="",
                parent_contexts=[],
                trace={},
                error="Empty question.",
                effective_query=query,
                history_used=False,
            )

        memory = _coerce_chat_memory(chat_memory, chat_history=chat_history)
        effective_query, history_used = self._rewrite_query_for_retrieval(query, memory)
        set_effective_query(effective_query)
        record_step(
            "rewrite",
            status="ok",
            summary=(
                f"history_used={history_used}; "
                + ("rewritten" if effective_query != query else "unchanged")
            ),
            details={
                "original_query": query,
                "effective_query": effective_query,
                "history_used": history_used,
                "self_healing_enabled": settings.SELF_HEALING_ENABLED,
                "memory_summary": _clean_memory_summary(memory.summary),
                "recent_messages": _format_chat_messages(memory.recent_messages),
            },
        )

        if settings.SELF_HEALING_ENABLED:
            return self._answer_self_healing(
                original_query=query,
                effective_query=effective_query,
                memory=memory,
                history_used=history_used,
                on_step=on_step,
            )

        return self._answer_open_loop(
            original_query=query,
            effective_query=effective_query,
            memory=memory,
            history_used=history_used,
            step=_step,
        )

    def _answer_open_loop(
        self,
        *,
        original_query: str,
        effective_query: str,
        memory: ChatMemory,
        history_used: bool,
        step: Callable[[str], None],
    ) -> ChatResult:
        """The original single-pass path (SELF_HEALING_ENABLED=false)."""
        try:
            pipeline = self._pipeline_lazy()
            step("retrieval")
            retrieval = pipeline.retrieve_with_sources(effective_query)
            # Deduplication + reranking run inside retrieve_with_sources; surface
            # them as their own completed steps for the progress UI.
            step("dedup")
            step("rerank")
        except Exception as exc:
            record_step(
                "error",
                status="error",
                summary=f"Retrieval failed: {exc}",
                error=str(exc),
            )
            return ChatResult(
                answer="",
                parent_contexts=[],
                trace={},
                error=f"Retrieval failed: {exc}",
                effective_query=effective_query,
                history_used=history_used,
            )

        try:
            step("generation")
            final = self._generate_and_filter(
                original_query=original_query,
                effective_query=effective_query,
                parents=retrieval.parent_contexts,
                chat_memory=memory,
            )
        except Exception as exc:
            record_step(
                "error",
                status="error",
                summary=f"LLM generation failed: {exc}",
                error=str(exc),
            )
            return ChatResult(
                answer="",
                parent_contexts=retrieval.parent_contexts,
                trace=_trace_to_dict(retrieval.trace),
                error=f"LLM generation failed: {exc}",
                effective_query=effective_query,
                history_used=history_used,
            )

        if final.parse_ok:
            warn = "" if final.used_companies else "The model returned no cited companies."
        else:
            warn = (
                "Could not parse the model's JSON response — showing all top-K "
                "sources as a fallback."
            )
        record_step(
            "final",
            status="warn" if warn else "ok",
            summary=f"open_loop done; {len(final.filtered_parents)} parents cited",
            details={
                "path": "open_loop",
                "warn": warn,
                "parse_ok": final.parse_ok,
                "used_companies": final.used_companies,
                "num_parents_cited": len(final.filtered_parents),
                "answer_chars": len(final.answer or ""),
            },
        )
        return ChatResult(
            answer=final.answer,
            parent_contexts=final.filtered_parents,
            trace=_trace_to_dict(retrieval.trace),
            error="",
            warn=warn,
            effective_query=effective_query,
            history_used=history_used,
        )

    def _answer_self_healing(
        self,
        *,
        original_query: str,
        effective_query: str,
        memory: ChatMemory,
        history_used: bool,
        on_step: Callable[[str], None] | None,
    ) -> ChatResult:
        """Closed-loop path: retrieve → judge → generate → verify → bounded retry."""
        try:
            pipeline = self._pipeline_lazy()
        except Exception as exc:
            record_step(
                "error",
                status="error",
                summary=f"Retrieval failed: {exc}",
                error=str(exc),
            )
            return ChatResult(
                answer="",
                parent_contexts=[],
                trace={},
                error=f"Retrieval failed: {exc}",
                effective_query=effective_query,
                history_used=history_used,
            )

        loop = SelfHealingLoop(
            retrieve_fn=lambda q, reranker_top_k: pipeline.retrieve_with_sources(
                q, reranker_top_k=reranker_top_k
            ),
            rerank_fn=lambda q, parents, top_k: pipeline.reranker.score_parents(
                q, parents, top_k
            ),
            final_answer_fn=self._generate_and_filter,
            config=SelfHealingConfig.from_settings(base_reranker_top_k=RERANKER_TOP_K),
            generate_answer_fn=self._generate_answer,
            on_step=on_step,
        )

        try:
            result = loop.run(
                original_query=original_query,
                effective_query=effective_query,
                chat_memory=memory,
            )
        except Exception as exc:
            record_step(
                "error",
                status="error",
                summary=f"Self-healing loop failed: {exc}",
                error=str(exc),
            )
            return ChatResult(
                answer="",
                parent_contexts=[],
                trace={},
                error=f"Self-healing loop failed: {exc}",
                effective_query=effective_query,
                history_used=history_used,
            )

        healing_dict = result.healing_trace.as_dict()
        return ChatResult(
            answer=result.answer,
            parent_contexts=result.parent_contexts,
            trace=healing_dict,
            error="",
            warn=result.warn,
            effective_query=result.effective_query,
            history_used=history_used,
            healing_trace=healing_dict.get("attempts", []),
            confidence=result.confidence,
        )

    def _generate_and_filter(
        self,
        *,
        original_query: str,
        effective_query: str,
        parents: List[ParentContext],
        chat_memory: ChatMemory | list[dict[str, str]] | None = None,
        correction_notes: str = "",
    ) -> FinalAnswer:
        """Build the answer prompt, generate, parse JSON, filter parents by the
        cited companies. Shared by both the open-loop and self-healing paths;
        emits no progress steps (the caller owns step reporting)."""
        formatted_memory = _format_chat_memory(_coerce_chat_memory(chat_memory))
        prompt = PROMPT_TEMPLATE.format(
            chat_memory=formatted_memory or "None",
            retrieved_parent_chunks=_format_retrieved_context(parents),
            user_question=original_query,
            effective_query=effective_query,
        )
        if correction_notes:
            prompt = prompt + render(REGENERATION_SUFFIX, CORRECTION_NOTES=correction_notes)

        raw_response = self._generate_answer(prompt, timeout=self._llm_timeout_seconds)
        answer_text, used_companies, parse_ok = _parse_json_response(raw_response or "")
        if parse_ok:
            filtered = _filter_parent_contexts_by_companies(parents, used_companies)
        else:
            # Keep the full reranked top-K so the user still has grounding info.
            filtered = list(parents)
        record_step(
            "generation",
            status="ok" if parse_ok else "warn",
            sub_step="regenerate" if correction_notes else "",
            summary=(
                f"parse_ok={parse_ok}; {len(used_companies)} companies cited; "
                f"parents {len(parents)}→{len(filtered)}"
            ),
            details={
                "parse_ok": parse_ok,
                "used_companies": used_companies,
                "parents_in": len(parents),
                "parents_after_company_filter": len(filtered),
                "correction_notes": correction_notes,
                "answer_text": answer_text,
                "prompt": prompt,
                "raw_response": raw_response,
            },
        )
        return FinalAnswer(
            answer=answer_text,
            used_companies=used_companies,
            parse_ok=parse_ok,
            filtered_parents=filtered,
        )

    def summarize_memory(
        self,
        existing_summary: str,
        messages: list[dict[str, str]],
    ) -> str:
        formatted_messages = _format_chat_messages(messages, total_char_limit=7000)
        existing_summary = _clean_memory_summary(existing_summary)
        if not formatted_messages:
            return existing_summary

        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            existing_summary=existing_summary or "None",
            new_turns=formatted_messages,
        )
        raw = self._generate_answer(
            prompt,
            timeout=min(self._llm_timeout_seconds, 60),
        )

        summary = _clean_summary_response(raw or "")
        return summary or existing_summary


def _coerce_chat_memory(
    chat_memory: ChatMemory | list[dict[str, str]] | None,
    chat_history: list[dict[str, str]] | None = None,
) -> ChatMemory:
    if isinstance(chat_memory, ChatMemory):
        return chat_memory
    if isinstance(chat_memory, list):
        return ChatMemory(recent_messages=chat_memory)
    if chat_history:
        return ChatMemory(recent_messages=chat_history)
    return ChatMemory()


def _clean_memory_summary(value: str | None) -> str:
    return str(value or "").strip()[:_MAX_SUMMARY_CHARS].rstrip()


def _format_chat_memory(memory: ChatMemory) -> str:
    parts: list[str] = []
    summary = _clean_memory_summary(memory.summary)
    recent = _format_chat_messages(memory.recent_messages)
    if summary:
        parts.append(f"Conversation summary:\n{summary}")
    if recent:
        parts.append(f"Recent conversation:\n{recent}")
    return "\n\n".join(parts)


def _format_chat_messages(
    messages: list[dict[str, str]] | None,
    max_chars_per_message: int = 900,
    total_char_limit: int = 5000,
) -> str:
    if not messages:
        return ""

    lines: list[str] = []
    total_chars = 0
    for item in messages:
        role = str(item.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content", "") or "").strip()
        if not content:
            continue
        if max_chars_per_message > 0 and len(content) > max_chars_per_message:
            content = content[:max_chars_per_message].rstrip()
        label = "User" if role == "user" else "Assistant"
        line = f"{label}: {content}"
        total_chars += len(line)
        if total_chars > total_char_limit:
            break
        lines.append(line)

    return "\n".join(lines)


def _format_chat_history(chat_history: list[dict[str, str]] | None) -> str:
    """Backward-compatible formatter for older tests/callers."""
    return _format_chat_messages(chat_history)


def _clean_rewrite_response(raw: str) -> str:
    text = (raw or "").strip()
    text = _JSON_FENCE_PREFIX.sub("", text)
    text = _JSON_FENCE_SUFFIX.sub("", text).strip()
    for line in text.splitlines():
        line = line.strip().lstrip("-*").strip().strip('"').strip("'").strip()
        line = _REWRITE_LABEL_PREFIX.sub("", line).strip()
        if line:
            return line
    return ""


def _clean_summary_response(raw: str) -> str:
    text = (raw or "").strip()
    text = _JSON_FENCE_PREFIX.sub("", text)
    text = _JSON_FENCE_SUFFIX.sub("", text).strip()
    text = _SUMMARY_LABEL_PREFIX.sub("", text).strip()
    if len(text) > _MAX_SUMMARY_CHARS:
        text = text[:_MAX_SUMMARY_CHARS].rstrip()
    return text


def _is_bad_rewrite(value: str) -> bool:
    text = (value or "").strip()
    lowered = text.lower()
    if not text:
        return True
    if len(text) > 500:
        return True
    if "{" in text or "}" in text:
        return True
    if "\n\n" in text:
        return True
    if lowered.startswith(_BAD_REWRITE_PREFIXES):
        return True
    if len(re.split(r"[.!?]\s+", text)) > 3:
        return True
    return False


def _normalize_query_for_compare(value: str) -> str:
    return _NON_ALNUM.sub(" ", (value or "").lower()).strip()


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
