"""Static mappings between user/LLM language and the control vocabulary.

Everything here is schema-level / control-level — never KB data values. Shared
by the pre-router (signal detection) and the validator (route correction,
operation normalization, required-field checks, needs flags).
"""
from __future__ import annotations

import re
from functools import lru_cache

# Re-export the field-alias map (owned by the metadata layer, README §22).
from ..metadata.provider import DEFAULT_FIELD_ALIASES as FIELD_ALIASES  # noqa: F401
from ..schemas import Operation, RouteName

# ---------------------------------------------------------------------------
# Stable schema registry — field NAME -> meaning. Schema-level only: these are
# descriptions of what each column holds, never KB data values. Single source of
# truth shared by the router prompt and the validator's field reasoning.
# ---------------------------------------------------------------------------
SCHEMA_FIELDS: dict[str, str] = {
    "category": "company classification, tier, or OEM classification",
    "ev_supply_chain_role": "role or capability in the EV supply chain",
    "product_service": "products, services, materials, components, or manufacturing output",
    "primary_oems": "OEM customers, OEM links, contracts, or relationships",
    "employment": "employee count",
    "updated_location": "general location text",
    "county": "county field",
    "city": "city field",
    "state": "state field",
    "primary_facility_type": "facility type",
    "industry_group": "industry or sector grouping",
}

# Field NAMES (not KB values) to try, in order, for a value that reads like a
# tier/OEM classification but was dropped on the wrong field. ``category`` first.
TIER_FIELD_CANDIDATES = ["category", "ev_supply_chain_role", "product_service"]

# ---------------------------------------------------------------------------
# Field groups used by required-field checks (real KB columns)
# ---------------------------------------------------------------------------
# Columns that count as a "structured filter" for structured_sql.
STRUCTURED_FILTER_FIELDS = {
    "category",
    "state",
    "industry_group",
    "ev_supply_chain_role",
    "primary_facility_type",
    "ev_battery_relevant",
    "employment",
}
# Numeric-by-name: employment is object dtype in the index (is_numeric=False)
# because the loader injects the string "Unknown"; treat it as numeric here.
NUMERIC_FIELDS = {"employment"}
# Geo anchor comes from the location column's components (city / "X County").
GEO_ANCHOR_FIELDS = {"updated_location"}
# Entity anchors for exact_lookup / disruption_analysis.
ENTITY_FIELDS = {"company", "primary_oems"}

# ---------------------------------------------------------------------------
# Keyword signal sets for deterministic route correction / pre-routing.
# Matched with word boundaries (see ``matches_any``) so substrings like "most"
# inside "almost" do not trigger.
# ---------------------------------------------------------------------------
AGGREGATE_SIGNALS = {
    "how many", "count", "number of", "total", "average", "minimum",
    "maximum", "highest", "lowest", "top", "group by", "per county",
    "per category",
}
LIST_SIGNALS = {
    "list", "show me", "show all", "which companies", "what companies", "all companies",
}
GEO_SIGNALS = {
    "near", "nearby", "closest", "distance", "map",
    "radius", "km", "miles", "county", "counties", "coordinates",
}
DISRUPTION_SIGNALS = {
    "shutdown", "shut down", "disruption", "disrupt", "disrupted",
    "replacement", "alternative", "alternatives", "backup",
    "substitute", "reroute", "resilience", "goes offline",
}
GREETING_SIGNALS = {
    "hi", "hello", "hey", "thanks", "thank you", "good morning",
    "good afternoon", "good evening",
}
META_SIGNALS = {"what can you do", "who are you", "what is this"}
# Obvious non-domain markers for the pre-router's out_of_domain shortcut.
OUT_OF_DOMAIN_SIGNALS = {
    "weather", "stock price", "recipe", "football", "nba", "horoscope",
    "capital of", "translate", "poem", "tell me a joke",
}
# Numeric-comparison / superlative wording. Any of these means the question is
# structured (filter / rank / threshold), never a single-entity exact lookup.
COMPARISON_SIGNALS = {
    "highest", "largest", "greatest", "biggest", "lowest", "smallest",
    "more than", "greater than", "fewer than", "less than", "over", "under",
    "at least", "at most", "top", "maximum", "minimum",
}
# Multi-entity question openers. A question with any of these is asking to
# filter/list MANY records, so it must not be classified as exact_lookup.
MULTI_ENTITY_SIGNALS = {
    "which companies", "which suppliers", "which georgia companies",
    "which firms", "list all", "identify all", "find companies",
    "find georgia companies", "find georgia-based", "show every",
    "show all", "all companies",
}
# True proximity/distance wording. Map and county-containment signals are handled
# separately by the validator because they are spatial without requiring a radius.
PROXIMITY_SIGNALS = {
    "near", "nearby", "closest", "within", "radius", "km", "kilometer",
    "kilometers", "miles", "drive", "distance", "coordinates", "proximity",
    "close to", "around",
}
# Web/document evidence wording. With structured filters present these upgrade a
# route to hybrid_search (structured DB + document chunks), NOT plain vector_search.
WEB_EVIDENCE_SIGNALS = {
    "web evidence", "source support", "documents", "document evidence",
    "from web", "from web data", "latest evidence", "what evidence",
    "mentioned in documents", "based on web", "web sources", "web data",
}


@lru_cache(maxsize=None)
def _compiled(signals: frozenset) -> "re.Pattern | None":
    parts: list[str] = []
    for sig in signals:
        sig = sig.strip()
        if not sig:
            continue
        # Phrases match literally; single tokens get word boundaries.
        parts.append(re.escape(sig) if " " in sig else rf"\b{re.escape(sig)}\b")
    return re.compile("|".join(parts), re.IGNORECASE) if parts else None


def matches_any(text: str, signals) -> bool:
    """True if any signal occurs in ``text`` (word-boundary aware)."""
    pattern = _compiled(frozenset(signals))
    return bool(pattern and pattern.search(text or ""))

# ---------------------------------------------------------------------------
# Per-route "needs" flags (README §35): (needs_kb_access, needs_document_retrieval)
# ---------------------------------------------------------------------------
ROUTE_NEEDS_FLAGS: dict[str, tuple[bool, bool]] = {
    RouteName.no_retrieval.value: (False, False),
    RouteName.out_of_domain.value: (False, False),
    RouteName.clarification_needed.value: (False, False),
    RouteName.exact_lookup.value: (True, False),
    RouteName.structured_sql.value: (True, False),
    RouteName.geo_search.value: (True, False),
    RouteName.keyword_search.value: (True, True),
    RouteName.vector_search.value: (True, True),
    RouteName.hybrid_search.value: (True, True),
    RouteName.disruption_analysis.value: (True, True),
}


def route_needs_flags(route: str) -> tuple[bool, bool]:
    """Return (needs_kb_access, needs_document_retrieval) for a route."""
    return ROUTE_NEEDS_FLAGS.get(str(route), (True, True))


def route_retrieval_sources(needs_kb: bool, needs_doc: bool) -> list[str]:
    """Derive the retrieval source labels from a route's needs flags.

    ``structured_db`` when the route reads structured KB columns; ``document_chunks``
    when it reads document/web chunks. Routes that need neither (no_retrieval,
    out_of_domain, clarification_needed) return an empty list.
    """
    sources: list[str] = []
    if needs_kb:
        sources.append("structured_db")
    if needs_doc:
        sources.append("document_chunks")
    return sources


# ---------------------------------------------------------------------------
# Operation normalization (README §34)
# ---------------------------------------------------------------------------
ALLOWED_OPERATIONS = {op.value for op in Operation}

DEFAULT_OPERATION: dict[str, str] = {
    RouteName.no_retrieval.value: Operation.direct_response.value,
    RouteName.exact_lookup.value: Operation.lookup_entity.value,
    RouteName.keyword_search.value: Operation.keyword_search.value,
    RouteName.structured_sql.value: Operation.list_records.value,
    RouteName.geo_search.value: Operation.nearby_search.value,
    RouteName.vector_search.value: Operation.semantic_search.value,
    RouteName.hybrid_search.value: Operation.hybrid_search.value,
    RouteName.disruption_analysis.value: Operation.find_alternatives.value,
    RouteName.clarification_needed.value: Operation.ask_clarification.value,
    RouteName.out_of_domain.value: Operation.reject_out_of_domain.value,
}

# Descriptive LLM operation phrasing -> canonical Operation label.
OPERATION_ALIASES: dict[str, str] = {
    "direct": Operation.direct_response.value,
    "respond": Operation.direct_response.value,
    "lookup": Operation.lookup_entity.value,
    "lookup_entity": Operation.lookup_entity.value,
    "get_attribute": Operation.lookup_entity.value,
    "list": Operation.list_records.value,
    "list_records": Operation.list_records.value,
    "filter": Operation.list_records.value,
    "count": Operation.count_records.value,
    "count_records": Operation.count_records.value,
    "aggregate": Operation.aggregate_records.value,
    "aggregate_records": Operation.aggregate_records.value,
    "sum": Operation.aggregate_records.value,
    "average": Operation.aggregate_records.value,
    "group": Operation.group_records.value,
    "group_records": Operation.group_records.value,
    "group_by": Operation.group_records.value,
    "keyword": Operation.keyword_search.value,
    "keyword_search": Operation.keyword_search.value,
    "semantic": Operation.semantic_search.value,
    "semantic_search": Operation.semantic_search.value,
    "vector_search": Operation.semantic_search.value,
    "hybrid": Operation.hybrid_search.value,
    "hybrid_search": Operation.hybrid_search.value,
    "nearby": Operation.nearby_search.value,
    "nearby_search": Operation.nearby_search.value,
    "nearby_or_distance_search": Operation.nearby_search.value,
    "distance": Operation.distance_search.value,
    "distance_search": Operation.distance_search.value,
    "alternatives": Operation.find_alternatives.value,
    "find_alternatives": Operation.find_alternatives.value,
    "find_alternatives_or_risk": Operation.find_alternatives.value,
    "risk": Operation.risk_analysis.value,
    "risk_analysis": Operation.risk_analysis.value,
    "clarify": Operation.ask_clarification.value,
    "ask_clarification": Operation.ask_clarification.value,
    "reject": Operation.reject_out_of_domain.value,
    "reject_out_of_domain": Operation.reject_out_of_domain.value,
}


def normalize_operation(raw_op: str | None, route: str) -> str:
    """Map an LLM-supplied operation to a safe control label.

    Falls back to the route's default operation when the operation is missing or
    unrecognized — arbitrary LLM strings never reach the execution branch.
    """
    if raw_op:
        key = str(raw_op).strip().lower().replace(" ", "_")
        if key in ALLOWED_OPERATIONS:
            return key
        if key in OPERATION_ALIASES:
            return OPERATION_ALIASES[key]
    return DEFAULT_OPERATION.get(str(route), Operation.direct_response.value)


# Human-readable required slots per route (for docs / clarification text).
ROUTE_REQUIRED_FIELDS: dict[str, str] = {
    RouteName.no_retrieval.value: "none",
    RouteName.out_of_domain.value: "none",
    RouteName.clarification_needed.value: "missing_fields + question_to_user",
    RouteName.exact_lookup.value: "a company / entity to look up",
    RouteName.keyword_search.value: "a keyword or query focus",
    RouteName.structured_sql.value: "at least one resolvable filter or an aggregate target",
    RouteName.geo_search.value: "a spatial anchor or map/filter request over geocoded records",
    RouteName.vector_search.value: "a query focus (semantic topic)",
    RouteName.hybrid_search.value: "a query focus plus a structured/geo signal",
    RouteName.disruption_analysis.value: "an anchor entity (company or OEM)",
}


# ---------------------------------------------------------------------------
# Numeric employment parsing (README §34) — deterministic, KB-free.
# We read comparison wording from the QUESTION text (not from KB values) so an
# "employment" comparison becomes a real numeric operator instead of a CONTAINS.
# ---------------------------------------------------------------------------
# Word numbers used for limits like "three companies" / "top five".
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
# Anything that means "number of people working at the company".
_EMPLOY_CONTEXT = re.compile(
    r"employ\w*|\bworkers?\b|workforce|head\s?count|\bstaff\b|\bjobs?\b|personnel",
    re.IGNORECASE,
)
# Comparison phrase -> operator. Longer phrases first so "no more than" beats
# "more than" and "no fewer than" beats "fewer than" during alternation.
_PRE_COMPARATORS: list[tuple[str, str]] = [
    ("no fewer than", "GTE"), ("no less than", "GTE"),
    ("no more than", "LTE"),
    ("greater than", "GT"), ("more than", "GT"), ("larger than", "GT"),
    ("fewer than", "LT"), ("less than", "LT"), ("smaller than", "LT"),
    ("at least", "GTE"), ("minimum of", "GTE"), ("a minimum of", "GTE"),
    ("at most", "LTE"), ("maximum of", "LTE"), ("up to", "LTE"),
    ("over", "GT"), ("above", "GT"), ("exceeding", "GT"), ("exceeds", "GT"),
    ("under", "LT"), ("below", "LT"),
]
# Phrases that follow the number, e.g. "300 or more".
_POST_COMPARATORS: list[tuple[str, str]] = [
    ("or more", "GTE"), ("or higher", "GTE"), ("or greater", "GTE"),
    ("and above", "GTE"), ("or fewer", "LTE"), ("or less", "LTE"),
    ("or lower", "LTE"), ("and below", "LTE"),
]
_NUM = r"(\d[\d,]*)"
_PRE_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p, _ in
                       sorted(_PRE_COMPARATORS, key=lambda x: -len(x[0])))
    + r")\s+" + _NUM,
    re.IGNORECASE,
)
_POST_RE = re.compile(
    _NUM + r"\s+(" + "|".join(re.escape(p) for p, _ in _POST_COMPARATORS) + r")",
    re.IGNORECASE,
)
_PRE_OP = {p.lower(): op for p, op in _PRE_COMPARATORS}
_POST_OP = {p.lower(): op for p, op in _POST_COMPARATORS}


def _has_employment_context(text: str, start: int, end: int, window: int = 55) -> bool:
    """True when an employment keyword sits near the matched comparison."""
    return bool(_EMPLOY_CONTEXT.search(text[max(0, start - window): end + window]))


def parse_employment_comparisons(lower: str) -> list[tuple[str, int]]:
    """Extract ``(operator, value)`` employment thresholds from the question.

    Returns e.g. ``[("GT", 300)]`` for "over 300 employees". Empty when there is
    no employment-scoped numeric comparison. Never consults KB values.
    """
    out: list[tuple[str, int]] = []
    for m in _PRE_RE.finditer(lower):
        if _has_employment_context(lower, m.start(), m.end()):
            op = _PRE_OP[m.group(1).lower()]
            out.append((op, int(m.group(2).replace(",", ""))))
    for m in _POST_RE.finditer(lower):
        if _has_employment_context(lower, m.start(), m.end()):
            op = _POST_OP[m.group(2).lower()]
            out.append((op, int(m.group(1).replace(",", ""))))
    # De-dup while preserving order.
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Ranking parsing — "top 10 by employment", "highest employment", etc.
# ---------------------------------------------------------------------------
_DESC_WORDS = {"highest", "largest", "greatest", "biggest", "most", "maximum", "top", "leading"}
_ASC_WORDS = {"lowest", "smallest", "fewest", "least", "minimum", "bottom"}
_TOP_N_RE = re.compile(r"\b(?:top|first|leading)\s+(\d+|" + "|".join(_WORD_NUMBERS) + r")\b", re.IGNORECASE)
# A leading count before a plural entity noun ("three Georgia companies"). The
# lookbehind/lookahead stop a slash fragment like "2/3" from matching its "3".
_COUNT_ENTITY_RE = re.compile(
    r"(?<![\w/])(\d+|" + "|".join(_WORD_NUMBERS) + r")(?![\d/])\s+(?:[a-z-]+\s+){0,3}?"
    r"(?:companies|suppliers|firms|facilities|manufacturers|areas|counties|sites|plants)\b",
    re.IGNORECASE,
)


def _as_int(token: str) -> int | None:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return _WORD_NUMBERS.get(token)


def parse_ranking(lower: str) -> tuple[list[str], int | None]:
    """Return ``(sort_by, limit)`` for a ranking question (KB-free).

    ``sort_by`` is only populated with ``employment`` when the question is
    employment-scoped; otherwise it stays empty (we never guess a sort column).
    """
    descending = matches_any(lower, _DESC_WORDS)
    ascending = matches_any(lower, _ASC_WORDS)

    limit: int | None = None
    m = _TOP_N_RE.search(lower)
    if m:
        limit = _as_int(m.group(1))
    if limit is None:
        m = _COUNT_ENTITY_RE.search(lower)
        if m:
            limit = _as_int(m.group(1))
    # "highest"/"greatest" with no explicit count implies a single best row.
    if limit is None and re.search(r"\b(highest|greatest)\b", lower):
        limit = 1

    sort_by: list[str] = []
    if (descending or ascending) and _EMPLOY_CONTEXT.search(lower):
        direction = "DESC" if descending or not ascending else "ASC"
        sort_by = [f"employment {direction}"]
    return sort_by, limit


# ---------------------------------------------------------------------------
# Output-column detection — separate "show / list / what is its X" from filters.
# Uses ONLY schema-level field aliases + English output cues, never KB values.
# ---------------------------------------------------------------------------
_OUTPUT_CUE_RE = re.compile(
    r"\b(show|list|display|include|report|provide|return|give|"
    r"what(?:'s| is| are)?(?: its| their| the)?|their|its|"
    r"along with|as well as|and the|and its|and their)\b",
    re.IGNORECASE,
)
# Aliases longest-first so "supply chain role" wins over "role".
_ALIASES_BY_LEN = sorted(FIELD_ALIASES.items(), key=lambda kv: -len(kv[0]))


def _word_bounded(text: str, idx: int, phrase: str) -> bool:
    before = text[idx - 1] if idx > 0 else " "
    after_i = idx + len(phrase)
    after = text[after_i] if after_i < len(text) else " "
    return not before.isalnum() and not after.isalnum()


def detect_output_columns(lower: str, max_gap: int = 30) -> list[str]:
    """Real columns the question asks to SHOW (output), not filter on.

    A field alias counts as an output column when it appears shortly after an
    output cue ("show X", "what is its X", "and their X"). Returns real column
    names (mapped via the alias table); callers intersect with allowed fields.
    """
    cue_ends = [m.end() for m in _OUTPUT_CUE_RE.finditer(lower)]
    if not cue_ends:
        return []
    cols: list[str] = []
    for phrase, column in _ALIASES_BY_LEN:
        start = 0
        while True:
            idx = lower.find(phrase, start)
            if idx == -1:
                break
            if _word_bounded(lower, idx, phrase) and any(
                0 <= idx - end <= max_gap for end in cue_ends
            ):
                cols.append(column)
                break
            start = idx + len(phrase)
    return list(dict.fromkeys(cols))


# ---------------------------------------------------------------------------
# Cross-field rescue — move a value off a likely-wrong field using question
# wording + field semantics (never KB candidate values).
# ---------------------------------------------------------------------------
# Wording that marks a value as a tier/OEM classification (belongs in category).
_CATEGORY_LIKE_RE = re.compile(r"\btier\b|\boem\b|footprint|supply chain", re.IGNORECASE)
_LOCATION_RE = re.compile(
    r"\b(county|counties|city|cities|state|region|regions|area|areas|located|location)\b",
    re.IGNORECASE,
)
_INDUSTRY_RE = re.compile(r"\b(industry|industries|sector|sectors)\b", re.IGNORECASE)
_PRODUCT_RE = re.compile(
    r"\b(produce[sd]?|producing|manufactur\w*|material\w*|component\w*|product\w*|"
    r"part[s]?|make[s]?|making|equipment|enclosure\w*)\b",
    re.IGNORECASE,
)
_ROLE_RE = re.compile(
    r"\b(role[s]?|capab\w*|supplier\w*|supplies|services?|managing|management)\b",
    re.IGNORECASE,
)


def looks_like_category(value: str) -> bool:
    """True when the value itself reads like a tier/OEM classification token."""
    return bool(_CATEGORY_LIKE_RE.search(str(value or "")))


def is_field_name_echo(field: str, value) -> bool:
    """True when ``value`` is just the column header echoed back (not a real value).

    Output questions ("what is its EV Supply Chain Role?") make the LLM emit the
    field NAME as a filter value. We detect that by exact match against the real
    column name (underscores -> spaces) or any alias pointing at the field — never
    a substring test, so genuine values like "Tier 1" are kept.
    """
    folded = str(value or "").strip().casefold()
    if not folded:
        return True
    names = {field.casefold(), field.replace("_", " ").casefold()}
    names.update(alias.casefold() for alias, col in FIELD_ALIASES.items() if col == field)
    return folded in names


def rescue_text_field(value, lower: str) -> str:
    """Pick a text-compatible column for a value mis-assigned to ``category``.

    Decision uses only the value text and question wording — no KB lookup. The
    default is ``ev_supply_chain_role`` (a searchable capability field) so an
    unresolved value is preserved as a CONTAINS filter rather than clarified.
    """
    blob = f"{lower} {str(value or '').lower()}"
    if _LOCATION_RE.search(blob):
        return "updated_location"
    if _INDUSTRY_RE.search(blob):
        return "industry_group"
    if _PRODUCT_RE.search(blob):
        return "product_service"
    if _ROLE_RE.search(blob):
        return "ev_supply_chain_role"
    return "ev_supply_chain_role"
