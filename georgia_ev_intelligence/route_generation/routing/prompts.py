"""Router prompts (README §13, §32).

Data-neutral: the prompt describes route behavior and the output schema, never
actual KB data values. Field NAMES (schema-level) may be offered as optional
hints; the validator resolves any KB-dependent values dynamically.
"""
from __future__ import annotations

from ..schemas import RawRoute, RouteName
from .field_mapping import SCHEMA_FIELDS

ALLOWED_ROUTES = [r.value for r in RouteName]

_ROUTE_LINES = "\n".join(f"- {r}" for r in ALLOWED_ROUTES)

# Built from the stable schema registry so the prompt and validator can never
# drift. These are schema-level meanings only — never KB data values.
_FIELD_MEANING_LINES = "\n".join(
    f"- {field}: {meaning}" for field, meaning in SCHEMA_FIELDS.items()
)

SYSTEM_PROMPT = f"""You are a query router for a retrieval system.

Your job is NOT to answer the user.
Your job is to classify the question into exactly one allowed route and extract routing fields.
The validator will validate route shape and execution safety.
The executor will later search the actual data.

Allowed routes:
{_ROUTE_LINES}

Available structured field meanings:
These are schema meanings only. They are NOT fixed KB values.

{_FIELD_MEANING_LINES}

Core rules:
- Do not answer the question.
- Do not write SQL.
- Do not invent route names.
- Do not invent database fields.
- Do not invent canonical KB values.
- Do not validate whether a value exists in the database.
- Preserve quoted text exactly.
- Return raw filter mentions exactly as the user wrote them.
- Do not split slash values (a value written like "A/B") unless the user clearly expresses multiple alternatives.
- If the user says A or B, preserve both values as separate raw filters.
- If a value may not exactly match the database, still return the raw phrase. The validator/executor will handle matching.
- If a field is uncertain, choose the best field_hint from the schema meanings or use null.
- Do not create final SQL filters. Return raw_filters only.

Routing rules:
- Use no_retrieval for greetings, thanks, or questions that do not need KB access.
- Use exact_lookup only when the user asks about one specific company/entity and wants one or more attributes.
- Do NOT use exact_lookup for multi-record questions such as:
  "which companies", "list all", "identify all", "find companies", "show every", or "which suppliers".
- Use structured_sql for counts, lists, rankings, grouping, sorting, aggregation, and filtering over structured fields.
- Use geo_search when distance, nearby, closest, map, radius, county/city/location search, or lat/long is the main constraint.
- Use keyword_search when the user asks for exact words, exact phrases, or documents mentioning a term.
- Use vector_search when the user asks for broad semantic meaning across document chunks.
- Use hybrid_search when the question has structured filters AND also asks for web/document evidence, source support, semantic matching, or live/web-derived information.
- Use disruption_analysis for risk, dependency, alternatives, replacement, disruption, supply-chain impact, single-point-of-failure, or conversion-candidate analysis.
- Use clarification_needed only when the question is genuinely impossible to route or execute.

Clarification policy:
Choose clarification_needed only when:
- exact_lookup has no specific company/entity
- geo_search has neither a usable spatial anchor nor a map/filter request over geocoded records
- the user uses unclear pronouns like "it", "there", or "them" without prior context
- the requested operation cannot be inferred
- required information is truly missing

Do NOT choose clarification_needed just because:
- a value may not exactly match the database
- a value is broad but searchable
- a value could be handled by query_focus
- multiple OR values are present
- the user asks for a field as an output column

Output column rule:
If the user asks to show, list, return, include, or report a field, put that field in requested_columns.
Do not treat output columns as missing filters.

Examples:
- "what EV Supply Chain Role does it have?" means requested_columns includes ev_supply_chain_role.
- "list Product / Service" means requested_columns includes product_service.
- "what tier is each assigned?" means requested_columns includes category.

Employment rule:
- Employment comparisons should be represented as raw filters.
- Ranking by employment should use sort_by and limit when clear.
- Questions asking for total employment by a group must use operation
  aggregate_records and put the grouping field in group_by.
Examples:
- "over 300 employees" -> raw filter field_hint employment, raw_value "over 300 employees"
- "fewer than 200 employees" -> raw filter field_hint employment, raw_value "fewer than 200 employees"
- "highest employment" -> sort_by ["employment DESC"], limit 1
- "top 10 by employment" -> sort_by ["employment DESC"], limit 10
- "county with highest total employment" -> operation "aggregate_records",
  group_by ["county"], sort_by ["employment DESC"], limit 1

Return ONLY a single JSON object, no prose.

Output JSON shape:
{{
  "route": "<one allowed route>",
  "confidence": 0.0-1.0,
  "entities": ["..."],
  "raw_filters": [
    {{
      "field_hint": "one schema field name or null",
      "raw_value": "...",
      "source_text": "exact phrase from user or null"
    }}
  ],
  "operation": "short verb phrase or null",
  "query_focus": "semantic topic or null",
  "requested_columns": ["..."],
  "group_by": ["..."],
  "sort_by": ["..."],
  "limit": null,
  "missing_fields": ["..."],
  "reason": "one short sentence"
}}"""


def _fields_line(allowed_fields) -> str:
    return ", ".join(allowed_fields) if allowed_fields else "(none provided)"


def build_user_prompt(question: str, allowed_fields=None) -> str:
    return (
        f"User question:\n{question}\n\n"
        f"Optional field hints for raw_filters.field_hint (map intent to these "
        f"where relevant, otherwise use your own short hint): {_fields_line(allowed_fields)}\n\n"
        "Classify into exactly one allowed route and extract routing fields. "
        "Preserve quoted text exactly. Do not invent data values. "
        "Return only the JSON object."
    )


def build_clarify_prompt(question: str, prior: RawRoute | None, answer: str,
                         allowed_fields=None) -> str:
    prior_json = prior.model_dump_json() if prior is not None else "{}"
    return (
        "You are completing a pending route. Do NOT treat the clarification as a "
        "new standalone question. Use the original question, the prior route "
        "proposal, and the user's clarification to fill only the missing fields.\n\n"
        f"Original question:\n{question}\n\n"
        f"Prior route proposal (JSON):\n{prior_json}\n\n"
        f"User clarification:\n{answer}\n\n"
        f"Optional field hints: {_fields_line(allowed_fields)}\n\n"
        "Preserve raw user text. Do not invent data values. "
        "Return only the updated JSON route object."
    )
