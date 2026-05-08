"""
Phase 4 — KB-grounded soft filter interpreter.

Takes the already extracted hard filters plus the current GNEM candidate set
and asks a small local model to choose the narrowest additional structured
filters that are justified by the question and by the values that actually
exist in the KB.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import httpx

from filters_and_validation.query_entity_extractor import Entities
from shared.config import Config
from shared.logger import get_logger

logger = get_logger("phase4.filter_interpreter")

_MAX_SAMPLE_ROWS = 12
_MAX_FIELD_VALUES = 10


@dataclass
class SoftFilterPlan:
    require_ev_battery_relevant: str | None = None
    include_roles: list[str] = field(default_factory=list)
    exclude_roles: list[str] = field(default_factory=list)
    include_supplier_affiliation_types: list[str] = field(default_factory=list)
    include_classification_methods: list[str] = field(default_factory=list)
    include_facility_types: list[str] = field(default_factory=list)
    include_industry_groups: list[str] = field(default_factory=list)
    explanation: str = ""

    @property
    def active(self) -> bool:
        return any([
            self.require_ev_battery_relevant,
            self.include_roles,
            self.exclude_roles,
            self.include_supplier_affiliation_types,
            self.include_classification_methods,
            self.include_facility_types,
            self.include_industry_groups,
        ])


def _normalize(value: object) -> str:
    return str(value or "").strip().lower()


def _top_values(companies: list[dict], field: str) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for company in companies:
        value = str(company.get(field) or "").strip()
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:_MAX_FIELD_VALUES]


def _hard_filter_summary(entities: Entities) -> str:
    parts = []
    for label, value in [
        ("tier", entities.tier),
        ("county", entities.county),
        ("company_name", entities.company_name),
        ("oem", entities.oem),
        ("ev_role", entities.ev_role),
        ("ev_role_list", entities.ev_role_list),
        ("industry_group", entities.industry_group),
        ("facility_type", entities.facility_type),
        ("classification_method", entities.classification_method),
        ("supplier_affiliation_type", entities.supplier_affiliation_type),
        ("min_employment", entities.min_employment),
        ("max_employment", entities.max_employment),
        ("ev_relevant_filter", entities.ev_relevant_filter),
        ("ev_relevance_value", getattr(entities, "ev_relevance_value", None)),
    ]:
        if value not in (None, "", [], False):
            parts.append(f"{label}={value}")
    if not parts:
        return "none"
    return ", ".join(parts)


def _sample_rows(companies: list[dict]) -> str:
    rows = []
    for company in companies[:_MAX_SAMPLE_ROWS]:
        rows.append(
            " | ".join([
                str(company.get("company_name") or ""),
                str(company.get("tier") or ""),
                str(company.get("ev_battery_relevant") or ""),
                str(company.get("ev_supply_chain_role") or ""),
                str(company.get("supplier_affiliation_type") or ""),
                str(company.get("classification_method") or ""),
                str(company.get("industry_group") or ""),
                str(company.get("facility_type") or ""),
                str(company.get("products_services") or "")[:120],
            ])
        )
    return "\n".join(rows)


def _build_prompt(question: str, entities: Entities, companies: list[dict]) -> str:
    ev_values = _top_values(companies, "ev_battery_relevant")
    role_values = _top_values(companies, "ev_supply_chain_role")
    affiliation_values = _top_values(companies, "supplier_affiliation_type")
    classification_values = _top_values(companies, "classification_method")
    industry_values = _top_values(companies, "industry_group")
    facility_values = _top_values(companies, "facility_type")

    def render(values: list[tuple[str, int]]) -> str:
        if not values:
            return "(none)"
        return "\n".join(f"- {value} ({count})" for value, count in values)

    return f"""You are choosing additional retrieval filters for a Georgia EV knowledge base.

Question:
{question}

Hard filters already applied:
{_hard_filter_summary(entities)}

Residual keywords from the question:
{entities.product_keywords}

Current candidate count:
{len(companies)}

Available KB values in the current candidate set:
EV relevance:
{render(ev_values)}

Roles:
{render(role_values)}

Supplier affiliation types:
{render(affiliation_values)}

Classification methods:
{render(classification_values)}

Industry groups:
{render(industry_values)}

Facility types:
{render(facility_values)}

Sample candidate rows:
Company | Tier | EV_Relevant | Role | Supplier_Affiliation | Classification | Industry | Facility | Products
{_sample_rows(companies)}

Task:
Choose ONLY the extra structured filters that are clearly supported by the question.
Use only values shown above. Do not invent values.

Rules:
1. Direct KB matches are already handled as hard filters. You are only interpreting the ambiguous remainder.
2. Prefer the narrowest faithful filter when the question says things like "primary", "direct", "specific", or clearly EV/battery-focused.
3. If the question does NOT clearly justify a filter, leave it empty.
4. If EV relevance should be explicit, set require_ev_battery_relevant to exactly one of: "Yes", "Indirect", "No".
5. Return STRICT JSON only.

JSON schema:
{{
  "require_ev_battery_relevant": "Yes|Indirect|No|",
  "include_roles": ["..."],
  "exclude_roles": ["..."],
  "include_supplier_affiliation_types": ["..."],
  "include_classification_methods": ["..."],
  "include_facility_types": ["..."],
  "include_industry_groups": ["..."],
  "explanation": "short explanation"
}}
"""


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
        raw = raw.rstrip("`").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    return json.loads(match.group(0))


def _clean_list(values: object, allowed: list[tuple[str, int]]) -> list[str]:
    allowed_map = {_normalize(value): value for value, _ in allowed}
    result: list[str] = []
    for value in values if isinstance(values, list) else []:
        normalized = _normalize(value)
        if normalized in allowed_map and allowed_map[normalized] not in result:
            result.append(allowed_map[normalized])
    return result


def interpret_soft_filters(question: str, entities: Entities, companies: list[dict]) -> SoftFilterPlan:
    if not companies:
        return SoftFilterPlan()

    cfg = Config.get()
    prompt = _build_prompt(question, entities, companies)
    payload = {
        "model": cfg.ollama_cypher_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 300,
            "num_ctx": 4096,
        },
    }

    try:
        url = f"{cfg.ollama_base_url}/api/generate"
        with httpx.Client(timeout=90.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = _extract_json(str(response.json().get("response", "")))

        ev_values = _top_values(companies, "ev_battery_relevant")
        role_values = _top_values(companies, "ev_supply_chain_role")
        affiliation_values = _top_values(companies, "supplier_affiliation_type")
        classification_values = _top_values(companies, "classification_method")
        facility_values = _top_values(companies, "facility_type")
        industry_values = _top_values(companies, "industry_group")
        allowed_ev = {_normalize(value): value for value, _ in ev_values}

        plan = SoftFilterPlan(
            require_ev_battery_relevant=allowed_ev.get(_normalize(data.get("require_ev_battery_relevant"))),
            include_roles=_clean_list(data.get("include_roles"), role_values),
            exclude_roles=_clean_list(data.get("exclude_roles"), role_values),
            include_supplier_affiliation_types=_clean_list(
                data.get("include_supplier_affiliation_types"), affiliation_values
            ),
            include_classification_methods=_clean_list(
                data.get("include_classification_methods"), classification_values
            ),
            include_facility_types=_clean_list(data.get("include_facility_types"), facility_values),
            include_industry_groups=_clean_list(data.get("include_industry_groups"), industry_values),
            explanation=str(data.get("explanation") or "").strip(),
        )
        if plan.active:
            logger.info("Soft filter plan: %s", plan)
        else:
            logger.info("Soft filter interpreter returned no extra filters")
        return plan
    except Exception as exc:
        logger.warning("Soft filter interpretation failed: %s", exc)
        return SoftFilterPlan()


# ── New retrieval pipeline: demoted suggester ────────────────────────────────
# In the V4 retrieval architecture this module is no longer in the main path.
# It is invoked only by synonym_expander as a last-resort *suggester* when
# no rule and no schema match was found for a residual abstract term.
# Suggestions are recorded with confidence capped at 0.50 so the validator
# always has the final say.

def suggest_mapping(question: str, entities: Entities, companies: list[dict] | None = None):
    """
    Wrap the existing LLM soft-filter call but return its output as a list
    of CandidateMapping objects (suitable for synonym_expander to merge).

    Returns an empty list on any failure or when the LLM output is empty.
    Confidence is fixed at 0.50 so the orchestrator never picks an LLM
    suggestion over a KB-grounded heuristic or a human-approved rule.
    """
    from core_agent.retrieval_types import CandidateMapping

    if not companies:
        return []

    plan = interpret_soft_filters(question, entities, companies)
    if not plan.active:
        return []

    suggestions: list[CandidateMapping] = []
    if plan.require_ev_battery_relevant:
        suggestions.append(CandidateMapping(
            mapping=f"ev_battery_relevant = '{plan.require_ev_battery_relevant}'",
            mapped_column="ev_battery_relevant",
            mapped_value=plan.require_ev_battery_relevant,
            support_basis="llm_suggestion",
            confidence=0.50,
        ))
    for role in plan.include_roles:
        suggestions.append(CandidateMapping(
            mapping=f"ev_supply_chain_role contains '{role}'",
            mapped_column="ev_supply_chain_role",
            mapped_value=role,
            support_basis="llm_suggestion",
            confidence=0.50,
        ))
    for industry in plan.include_industry_groups:
        suggestions.append(CandidateMapping(
            mapping=f"industry_group = '{industry}'",
            mapped_column="industry_group",
            mapped_value=industry,
            support_basis="llm_suggestion",
            confidence=0.50,
        ))
    for facility in plan.include_facility_types:
        suggestions.append(CandidateMapping(
            mapping=f"facility_type = '{facility}'",
            mapped_column="facility_type",
            mapped_value=facility,
            support_basis="llm_suggestion",
            confidence=0.50,
        ))
    return suggestions
