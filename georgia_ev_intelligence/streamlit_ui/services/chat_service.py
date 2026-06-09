"""Chat service: orchestrate hybrid retrieval + Ollama generation.

The LLM is asked to return a strict JSON object so we can (a) extract the
formatted answer for display and (b) extract `used_companies`, which we use
to filter the retrieved `parent_contexts` down to just the parents that the
LLM actually cited. Common malformed JSON is salvaged so JSON scaffolding is
never presented as the user-facing answer.
"""
from __future__ import annotations

import json
import inspect
import math
import re
from itertools import combinations
from typing import Any, Callable, Dict, List, Tuple

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

Conversation history:
{conversation_history}

Current user question:
{user_question}

---

The current question may refer to entities from the conversation history using
words such as "these", "those", "them", or "the same companies". Resolve those
references from the history and answer only the current question.

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
_ANAPHORIC_FOLLOWUP = re.compile(
    r"\b(these|those|them|they|their|same|above|previous|that list|this list)\b",
    re.IGNORECASE,
)
_ANSWER_FIELD = re.compile(r"""["']answer["']\s*:\s*(?P<quote>["'])?""", re.IGNORECASE)
_USED_FIELD = re.compile(r"""["']used_companies["']\s*:""", re.IGNORECASE)
_COMPANY_COUNT = re.compile(
    r"(\b(?:There (?:are|is)|Found)\s+)(\d+)(?=[^\n.]*\bcompan(?:y|ies)\b)",
    re.IGNORECASE,
)
_NEAR_PLACE = re.compile(
    r"\b(?:near(?:est)?(?:\s+to)?|closest\s+to|around)\s+"
    r"([A-Za-z][A-Za-z\s\-']+?)(?:[?.!,]|$)",
    re.IGNORECASE,
)
_EXPLICIT_RADIUS = re.compile(
    r"(\d+(?:\.\d+)?)\s*(km|kilometers?|miles?|mi)\b",
    re.IGNORECASE,
)
_PARENT_FIELD = re.compile(r"^\s*([^:\n]+):\s*(.*?)\s*$", re.MULTILINE)
_PARENT_FIELD_KEYS = {
    "company": "company",
    "category": "category",
    "updated location": "location",
    "latitude": "latitude",
    "longitude": "longitude",
    "ev supply chain role": "ev_supply_chain_role",
}
_DEFAULT_NEAR_RADIUS_KM = 100.0


class ChatService(IChatService):
    """Concrete chat service backed by HybridRetrievalOrchestrator + Ollama."""

    def __init__(
        self,
        retrieval_pipeline_factory: Callable,
        generate_answer_fn: Callable[..., str] = generate_answer,
        llm_timeout_seconds: int = 180,
        structured_lookup_fn: Callable[[], Dict[int, Dict[str, Any]]] | None = None,
    ) -> None:
        self._retrieval_pipeline_factory = retrieval_pipeline_factory
        self._generate_answer = generate_answer_fn
        self._llm_timeout_seconds = llm_timeout_seconds
        self._structured_lookup_fn = structured_lookup_fn
        self._pipeline = None

    def _pipeline_lazy(self):
        if self._pipeline is None:
            self._pipeline = self._retrieval_pipeline_factory()
        return self._pipeline

    def answer(
        self,
        query: str,
        history: list[tuple[str, str]] | None = None,
        previous_contexts: List[ParentContext] | None = None,
        on_step: Callable[[str], None] | None = None,
    ) -> ChatResult:
        def _step(name: str) -> None:
            if on_step is not None:
                try:
                    on_step(name)
                except Exception:
                    pass

        query = (query or "").strip()
        if not query:
            return ChatResult(answer="", parent_contexts=[], trace={}, error="Empty question.")

        spatial_result = self._answer_contextual_spatial_query(
            query,
            previous_contexts or [],
            _step,
        )
        if spatial_result is not None:
            return spatial_result

        structured_result = self._answer_exact_role_query(query, _step)
        if structured_result is not None:
            return structured_result

        reuse_previous = bool(previous_contexts) and _is_contextual_followup(query, history)
        if reuse_previous:
            parent_contexts = list(previous_contexts or [])
            retrieval_trace = None
            _step("retrieval")
            _step("dedup")
            _step("rerank")
        else:
            try:
                pipeline = self._pipeline_lazy()
                _step("retrieval")
                retrieval_query = _contextualize_retrieval_query(query, history)
                retrieval = pipeline.retrieve_with_sources(retrieval_query)
                parent_contexts = retrieval.parent_contexts
                retrieval_trace = retrieval.trace
                # Deduplication + reranking run inside retrieve_with_sources; surface
                # them as their own completed steps for the progress UI.
                _step("dedup")
                _step("rerank")
            except Exception as exc:
                return ChatResult(
                    answer="", parent_contexts=[], trace={}, error=f"Retrieval failed: {exc}"
                )

        try:
            _step("generation")
            prompt = PROMPT_TEMPLATE.format(
                retrieved_parent_chunks=_format_retrieved_context(parent_contexts),
                conversation_history=_format_conversation_history(history),
                user_question=query,
            )
            raw_response = _generate_json_response(
                self._generate_answer,
                prompt,
                timeout=self._llm_timeout_seconds,
            )
        except Exception as exc:
            return ChatResult(
                answer="",
                parent_contexts=parent_contexts,
                trace=_trace_to_dict(retrieval_trace),
                error=f"LLM generation failed: {exc}",
            )

        answer_text, used_companies, parse_ok = _parse_json_response(raw_response or "")
        if parse_ok:
            filtered = _filter_parent_contexts_by_companies(
                parent_contexts, used_companies
            )
            return ChatResult(
                answer=answer_text,
                parent_contexts=filtered,
                trace=_trace_to_dict(retrieval_trace),
                error="",
                warn="" if used_companies else "The model returned no cited companies.",
            )

        # Plain-text fallback: keep the full reranked top-K. Structured-looking
        # responses are cleaned so raw JSON scaffolding is never shown in chat.
        return ChatResult(
            answer=_clean_structured_fallback(answer_text),
            parent_contexts=parent_contexts,
            trace=_trace_to_dict(retrieval_trace),
            error="",
            warn="Could not parse the model's JSON response — showing all top-K sources as a fallback.",
        )

    def _answer_exact_role_query(
        self,
        query: str,
        on_step: Callable[[str], None],
    ) -> ChatResult | None:
        """Answer explicit EV supply-chain role filters from structured source rows."""
        if self._structured_lookup_fn is None or "role" not in query.lower():
            return None

        try:
            lookup = self._structured_lookup_fn()
        except Exception:
            return None

        requested_roles = _requested_roles(query, lookup)
        if not requested_roles:
            return None

        requested_normalized = {_normalize_field_value(role) for role in requested_roles}
        matches: List[Tuple[int, Dict[str, Any]]] = []
        seen_companies: set[str] = set()
        for row_id, row in lookup.items():
            role = _clean_cell(row.get("ev_supply_chain_role"))
            company = _clean_cell(row.get("company"))
            company_key = _normalize_company(company)
            if (
                not company_key
                or _normalize_field_value(role) not in requested_normalized
                or company_key in seen_companies
            ):
                continue
            seen_companies.add(company_key)
            matches.append((int(row_id), row))

        matches.sort(key=lambda item: _clean_cell(item[1].get("company")).lower())
        parents = [_structured_parent(row_id, row) for row_id, row in matches]
        answer = _format_role_answer(matches, requested_roles)

        for step in ("retrieval", "dedup", "rerank", "generation"):
            on_step(step)

        return ChatResult(
            answer=answer,
            parent_contexts=parents,
            trace={
                "source": "structured_company_lookup",
                "requested_roles": requested_roles,
                "matched_company_count": len(matches),
            },
        )

    def _answer_contextual_spatial_query(
        self,
        query: str,
        previous_contexts: List[ParentContext],
        on_step: Callable[[str], None],
    ) -> ChatResult | None:
        """Calculate proximity and pairwise distance follow-ups from coordinates."""
        if not previous_contexts or not _ANAPHORIC_FOLLOWUP.search(query):
            return None

        lower = query.lower()
        is_distance = "distance" in lower or "far apart" in lower
        place_match = _NEAR_PLACE.search(query)
        if not is_distance and place_match is None:
            return None

        try:
            lookup = self._structured_lookup_fn() if self._structured_lookup_fn else {}
        except Exception:
            lookup = {}
        records = _context_records(previous_contexts, lookup)

        if is_distance:
            result = _distance_result(records)
        else:
            place = re.sub(
                r"\s+in\s+georgia$",
                "",
                place_match.group(1).strip(),
                flags=re.IGNORECASE,
            )
            place_coords = _resolve_place_from_lookup(place, lookup)
            result = (
                _proximity_result(records, place, place_coords, _query_radius_km(query))
                if place_coords
                else None
            )

        if result is None:
            return None
        for step in ("retrieval", "dedup", "rerank", "generation"):
            on_step(step)
        return result


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
        answer_text, used_companies = _salvage_structured_response(text)
        if answer_text:
            return _reconcile_company_count(answer_text, used_companies), used_companies, True
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

    return _reconcile_company_count(answer_field.strip(), cleaned_used), cleaned_used, True


def _generate_json_response(generate_fn: Callable[..., str], prompt: str, timeout: int) -> str:
    """Use Ollama JSON mode when the injected generator supports it."""
    try:
        signature = inspect.signature(generate_fn)
        supports_json_mode = "json_mode" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        supports_json_mode = False

    if supports_json_mode:
        return generate_fn(prompt, timeout=timeout, json_mode=True)
    return generate_fn(prompt, timeout=timeout)


def _format_conversation_history(history: list[tuple[str, str]] | None) -> str:
    if not history:
        return "(none)"
    lines: List[str] = []
    for role, content in history[-6:]:
        clean = str(content or "").strip()
        if not clean:
            continue
        lines.append(f"{str(role).strip().title()}: {clean[:4000]}")
    return "\n\n".join(lines) or "(none)"


def _is_contextual_followup(
    query: str, history: list[tuple[str, str]] | None
) -> bool:
    return bool(history and _ANAPHORIC_FOLLOWUP.search(query or ""))


def _contextualize_retrieval_query(
    query: str, history: list[tuple[str, str]] | None
) -> str:
    """Give retrieval the missing subject for an anaphoric follow-up."""
    if not _is_contextual_followup(query, history):
        return query
    prior_user = next(
        (
            str(content).strip()
            for role, content in reversed(history or [])
            if str(role).lower() == "user" and str(content).strip()
        ),
        "",
    )
    if not prior_user:
        return query
    return f"{prior_user}\nFollow-up: {query}"


def _salvage_structured_response(text: str) -> Tuple[str, List[str]]:
    """Recover common model output where a multiline answer breaks JSON syntax."""
    answer_match = _ANSWER_FIELD.search(text)
    used_match = _USED_FIELD.search(text)
    if answer_match is None:
        return "", []

    answer_end = (
        used_match.start()
        if used_match is not None and used_match.start() > answer_match.end()
        else len(text)
    )
    answer = text[answer_match.end() : answer_end]
    answer = re.sub(r",?\s*$", "", answer)
    answer = answer.strip().rstrip("}").strip().strip("\"'").strip()
    answer = answer.replace("\\n", "\n").replace('\\"', '"')
    lines = answer.splitlines()
    if lines:
        lines[0] = re.sub(r"""["']\s*$""", "", lines[0]).rstrip()
        answer = "\n".join(lines).strip()

    used_companies = (
        _extract_loose_string_array(text[used_match.end() :]) if used_match is not None else []
    )
    return answer, used_companies


def _extract_loose_string_array(text: str) -> List[str]:
    start = text.find("[")
    if start < 0:
        return []

    depth = 0
    in_string = False
    escaped = False
    end = -1
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end = index + 1
                break

    candidate = text[start:end] if end > start else text[start:]
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        parsed = re.findall(r"""["']([^"']+)["']""", candidate)
    return [item.strip() for item in parsed if isinstance(item, str) and item.strip()]


def _reconcile_company_count(answer: str, used_companies: List[str]) -> str:
    """Make a company-count opening agree with the companies actually cited."""
    unique = list(dict.fromkeys(name.strip() for name in used_companies if name.strip()))
    if not unique:
        return answer
    match = _COMPANY_COUNT.search(answer)
    if match is None or int(match.group(2)) == len(unique):
        return answer
    return answer[: match.start(2)] + str(len(unique)) + answer[match.end(2) :]


def _clean_structured_fallback(text: str) -> str:
    answer, _ = _salvage_structured_response((text or "").strip())
    return answer or text


def _try_json_object(candidate: str):
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _requested_roles(
    query: str,
    lookup: Dict[int, Dict[str, Any]],
) -> List[str]:
    query_normalized = _normalize_field_value(query)
    roles = {
        _clean_cell(row.get("ev_supply_chain_role"))
        for row in lookup.values()
        if _clean_cell(row.get("ev_supply_chain_role"))
    }
    return sorted(
        (
            role
            for role in roles
            if _normalize_field_value(role) in query_normalized
        ),
        key=lambda role: query_normalized.index(_normalize_field_value(role)),
    )


def _context_records(
    parents: List[ParentContext],
    lookup: Dict[int, Dict[str, Any]],
) -> List[Tuple[ParentContext, Dict[str, Any]]]:
    records: List[Tuple[ParentContext, Dict[str, Any]]] = []
    for parent in parents:
        row = dict(lookup.get(int(parent.source_row_id), {}) or {})
        for label, value in _PARENT_FIELD.findall(parent.parent_chunk_text or ""):
            key = _PARENT_FIELD_KEYS.get(label.strip().lower())
            if key and not _clean_cell(row.get(key)):
                row[key] = value.strip()
        if _clean_cell(row.get("company")):
            records.append((parent, row))
    return records


def _resolve_place_from_lookup(
    place: str,
    lookup: Dict[int, Dict[str, Any]],
) -> Tuple[float, float] | None:
    wanted = _normalize_field_value(place).removesuffix(" county").strip()
    coordinates: List[Tuple[float, float]] = []
    for row in lookup.values():
        city = _normalize_field_value(_clean_cell(row.get("city")))
        county = _normalize_field_value(_clean_cell(row.get("county"))).removesuffix(
            " county"
        ).strip()
        if wanted not in {city, county}:
            continue
        coords = _row_coordinates(row)
        if coords is not None:
            coordinates.append(coords)
    if not coordinates:
        return None
    return (
        sum(lat for lat, _ in coordinates) / len(coordinates),
        sum(lon for _, lon in coordinates) / len(coordinates),
    )


def _query_radius_km(query: str) -> float:
    match = _EXPLICIT_RADIUS.search(query)
    if match is None:
        return _DEFAULT_NEAR_RADIUS_KM
    radius = float(match.group(1))
    return radius * 1.60934 if match.group(2).lower().startswith(("mi", "mile")) else radius


def _proximity_result(
    records: List[Tuple[ParentContext, Dict[str, Any]]],
    place: str,
    place_coords: Tuple[float, float],
    radius_km: float,
) -> ChatResult:
    matches: List[Tuple[float, ParentContext, Dict[str, Any]]] = []
    for parent, row in records:
        coords = _row_coordinates(row)
        if coords is None:
            continue
        distance_km = _haversine_scalar_km(place_coords, coords)
        if distance_km <= radius_km:
            matches.append((distance_km, parent, row))
    matches.sort(key=lambda item: (item[0], _clean_cell(item[2].get("company")).lower()))

    count = len(matches)
    lines = [
        f"There {'is' if count == 1 else 'are'} {count} of these "
        f"{'company' if count == 1 else 'companies'} within {radius_km:g} km of {place}."
    ]
    for distance_km, _, row in matches:
        company = _clean_cell(row.get("company")) or "n/a"
        location = _clean_cell(row.get("location")) or "n/a"
        lines.append(
            f"{company} | Distance from {place}: {distance_km:.1f} km "
            f"({distance_km * 0.621371:.1f} mi) | Updated Location: {location}"
        )
    return ChatResult(
        answer="\n".join(lines),
        parent_contexts=[parent for _, parent, _ in matches],
        trace={
            "source": "coordinate_calculation",
            "calculation": "proximity",
            "place": place,
            "radius_km": radius_km,
            "matched_company_count": count,
        },
    )


def _distance_result(
    records: List[Tuple[ParentContext, Dict[str, Any]]],
) -> ChatResult | None:
    located = [
        (parent, row, coords)
        for parent, row in records
        if (coords := _row_coordinates(row)) is not None
    ]
    if len(located) < 2:
        return None

    pairs = list(combinations(located, 2))
    if len(pairs) == 1:
        (parent_a, row_a, coords_a), (parent_b, row_b, coords_b) = pairs[0]
        company_a = _clean_cell(row_a.get("company")) or "n/a"
        company_b = _clean_cell(row_b.get("company")) or "n/a"
        distance_km = _haversine_scalar_km(coords_a, coords_b)
        answer = (
            f"The straight-line distance between {company_a} and {company_b} is "
            f"{distance_km:.1f} km ({distance_km * 0.621371:.1f} mi), calculated "
            "from their coordinates."
        )
        parents = [parent_a, parent_b]
    else:
        lines = [
            f"These are the {len(pairs)} pairwise straight-line distances, "
            "calculated from the companies' coordinates."
        ]
        for (_, row_a, coords_a), (_, row_b, coords_b) in pairs:
            distance_km = _haversine_scalar_km(coords_a, coords_b)
            lines.append(
                f"{_clean_cell(row_a.get('company'))} to {_clean_cell(row_b.get('company'))} "
                f"| Distance: {distance_km:.1f} km ({distance_km * 0.621371:.1f} mi)"
            )
        answer = "\n".join(lines)
        parents = [parent for parent, _, _ in located]

    return ChatResult(
        answer=answer,
        parent_contexts=parents,
        trace={
            "source": "coordinate_calculation",
            "calculation": "pairwise_distance",
            "company_count": len(located),
            "pair_count": len(pairs),
        },
    )


def _row_coordinates(row: Dict[str, Any]) -> Tuple[float, float] | None:
    lat = _safe_float(row.get("latitude"))
    lon = _safe_float(row.get("longitude"))
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def _haversine_scalar_km(
    first: Tuple[float, float],
    second: Tuple[float, float],
) -> float:
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1.0, value)))


def _structured_parent(row_id: int, row: Dict[str, Any]) -> ParentContext:
    fields = (
        ("Company", row.get("company")),
        ("Category", row.get("category")),
        ("Industry Group", row.get("industry_group")),
        ("Updated Location", row.get("location")),
        ("Address", row.get("address")),
        ("Latitude", row.get("latitude")),
        ("Longitude", row.get("longitude")),
        ("Primary Facility Type", row.get("facility_type")),
        ("EV Supply Chain Role", row.get("ev_supply_chain_role")),
        ("Primary OEMs", row.get("primary_oems")),
        ("Supplier or Affiliation Type", row.get("supplier_type")),
        ("Employment", row.get("employment")),
        ("Product / Service", row.get("product_service")),
        ("EV / Battery Relevant", row.get("ev_battery_relevant")),
    )
    parent_text = "\n".join(
        f"{label}: {_clean_cell(value)}"
        for label, value in fields
        if _clean_cell(value)
    )
    return ParentContext(
        record_id=f"KB_ROW_{row_id}",
        source_row_id=row_id,
        parent_chunk_text=parent_text,
    )


def _format_role_answer(
    matches: List[Tuple[int, Dict[str, Any]]],
    requested_roles: List[str],
) -> str:
    role_label = " or ".join(requested_roles)
    count = len(matches)
    noun = "company" if count == 1 else "companies"
    lines = [
        f"There {'is' if count == 1 else 'are'} {count} Georgia {noun} "
        f"classified under {role_label} roles."
    ]
    for _, row in matches:
        company = _clean_cell(row.get("company")) or "n/a"
        category = _clean_cell(row.get("category")) or "n/a"
        role = _clean_cell(row.get("ev_supply_chain_role")) or "n/a"
        lines.append(f"{company} [{category}] | EV Supply Chain Role: {role}")
    return "\n".join(lines)


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value).strip()


def _normalize_field_value(value: str) -> str:
    return " ".join(_NON_ALNUM.sub(" ", (value or "").lower()).split())


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
        if haystack not in needles:
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
