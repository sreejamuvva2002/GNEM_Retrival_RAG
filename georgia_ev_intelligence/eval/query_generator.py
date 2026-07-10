"""Synthetic test-question generator for routing/answer evaluation.

Questions are grounded in real KB entity values pulled from
``georgia_ev_intelligence/outputs/Normalized_kb.xlsx`` (the same file the
route-execution DB is built from) so ``exact_lookup``/``structured_sql``/
``geo_search``/etc. questions reference real companies, tiers, industries,
roles, OEMs, and locations instead of hallucinated ones.

Each generated question's ``expected_route`` is known by construction — the
template that produced it names the route it is meant to trigger — so
``scripts/eval_routing.py`` can score routing accuracy without any manual
labeling pass.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..route_generation.schemas import RouteName

DEFAULT_KB_PATH = Path(__file__).resolve().parents[1] / "outputs" / "Normalized_kb.xlsx"
KB_SHEET = "Normalized_kb"

_KB_COLUMNS = (
    "company",
    "category",
    "industry_group",
    "updated_location",
    "ev_supply_chain_role",
    "primary_oems",
    "product_service",
    "primary_facility_type",
    "supplier_or_affiliation_type",
    "employment",
)


@dataclass(frozen=True)
class GeneratedQuestion:
    """One synthetic question with a known expected route."""

    question: str
    expected_route: str
    template_id: str
    entities_used: dict[str, Any] = field(default_factory=dict)


def load_kb_values(kb_path: Path = DEFAULT_KB_PATH) -> dict[str, list[Any]]:
    """Return distinct non-null, non-"Unknown" values per relevant KB column."""
    df = pd.read_excel(kb_path, sheet_name=KB_SHEET)
    values: dict[str, list[Any]] = {}
    for col in _KB_COLUMNS:
        if col not in df.columns:
            values[col] = []
            continue
        series = df[col].dropna()
        if col == "employment":
            nums = pd.to_numeric(series, errors="coerce").dropna()
            values[col] = sorted({int(v) for v in nums if v > 0})
        else:
            vals = {
                str(v).strip()
                for v in series
                if str(v).strip() and str(v).strip().lower() != "unknown"
            }
            values[col] = sorted(vals)
    return values


def _pick(rng: random.Random, pool: list[Any]) -> Any | None:
    return rng.choice(pool) if pool else None


def _nearest_employment_threshold(rng: random.Random, pool: list[int]) -> int | None:
    """Pick a real employment value and jitter it into a plausible threshold."""
    base = _pick(rng, pool)
    if base is None:
        return None
    return max(1, base - rng.randint(-50, 50))


# ---------------------------------------------------------------------------
# Templates: (route, template_id, fn(rng, kb) -> (question, entities_used) | None)
# ---------------------------------------------------------------------------
TemplateFn = Callable[[random.Random, dict[str, list[Any]]], tuple[str, dict[str, Any]] | None]


def _tmpl_exact_role(rng, kb):
    company = _pick(rng, kb["company"])
    if not company:
        return None
    return f'What EV supply chain role does "{company}" have?', {"company": company}


def _tmpl_exact_location(rng, kb):
    company = _pick(rng, kb["company"])
    if not company:
        return None
    return f'Where is "{company}" located?', {"company": company}


def _tmpl_exact_products(rng, kb):
    company = _pick(rng, kb["company"])
    if not company:
        return None
    return f'What products or services does "{company}" provide?', {"company": company}


def _tmpl_exact_employment(rng, kb):
    company = _pick(rng, kb["company"])
    if not company:
        return None
    return f'How many employees does "{company}" have?', {"company": company}


def _tmpl_sql_category_count(rng, kb):
    category = _pick(rng, kb["category"])
    if not category:
        return None
    return f'How many companies are classified as "{category}"?', {"category": category}


def _tmpl_sql_industry_list(rng, kb):
    industry = _pick(rng, kb["industry_group"])
    if not industry:
        return None
    return f'List all companies in the "{industry}" industry group.', {"industry_group": industry}


def _tmpl_sql_employment_threshold(rng, kb):
    threshold = _nearest_employment_threshold(rng, kb["employment"])
    if threshold is None:
        return None
    return (
        f"Which companies have more than {threshold} employees?",
        {"employment_threshold": threshold},
    )


def _tmpl_sql_oem_filter(rng, kb):
    oem = _pick(rng, kb["primary_oems"])
    if not oem:
        return None
    return f'Show all suppliers with primary OEM "{oem}".', {"primary_oems": oem}


def _tmpl_sql_role_group_employment(rng, kb):
    role = _pick(rng, kb["ev_supply_chain_role"])
    if not role:
        return None
    return (
        f'What is the total employment for companies with an EV supply chain role of "{role}"?',
        {"ev_supply_chain_role": role},
    )


def _tmpl_geo_near(rng, kb):
    location = _pick(rng, kb["updated_location"])
    if not location:
        return None
    return f'What EV supply chain companies are near "{location}"?', {"updated_location": location}


def _tmpl_geo_radius(rng, kb):
    location = _pick(rng, kb["updated_location"])
    if not location:
        return None
    return (
        f'Show companies within 25 miles of "{location}".',
        {"updated_location": location},
    )


def _tmpl_geo_closest(rng, kb):
    location = _pick(rng, kb["updated_location"])
    if not location:
        return None
    return (
        f'Which suppliers are located closest to "{location}"?',
        {"updated_location": location},
    )


def _tmpl_keyword_phrase(rng, kb):
    phrase = _pick(rng, kb["product_service"])
    if not phrase:
        return None
    return f'Find documents that mention the exact phrase "{phrase}".', {"product_service": phrase}


def _tmpl_keyword_role(rng, kb):
    phrase = _pick(rng, kb["ev_supply_chain_role"])
    if not phrase:
        return None
    return (
        f'Search for the exact term "{phrase}" in supplier documents.',
        {"ev_supply_chain_role": phrase},
    )


def _tmpl_vector_risk(rng, kb):
    return (
        "What are the general risks facing EV battery supply chains in Georgia?",
        {},
    )


def _tmpl_vector_role_topic(rng, kb):
    role = _pick(rng, kb["ev_supply_chain_role"])
    if not role:
        return None
    return (
        f'Describe how companies with a "{role}" role fit into EV manufacturing.',
        {"ev_supply_chain_role": role},
    )


def _tmpl_hybrid_category_evidence(rng, kb):
    category = _pick(rng, kb["category"])
    role = _pick(rng, kb["ev_supply_chain_role"])
    if not category or not role:
        return None
    return (
        f'Which "{category}" suppliers with a "{role}" role have web evidence of recent expansion?',
        {"category": category, "ev_supply_chain_role": role},
    )


def _tmpl_hybrid_location_evidence(rng, kb):
    industry = _pick(rng, kb["industry_group"])
    location = _pick(rng, kb["updated_location"])
    if not industry or not location:
        return None
    return (
        f'Show "{industry}" companies near "{location}" along with supporting document evidence.',
        {"industry_group": industry, "updated_location": location},
    )


def _tmpl_disruption_alternatives(rng, kb):
    company = _pick(rng, kb["company"])
    if not company:
        return None
    return f'What are the alternatives if "{company}" shuts down?', {"company": company}


def _tmpl_disruption_risk(rng, kb):
    company = _pick(rng, kb["company"])
    if not company:
        return None
    return (
        f'What is the supply chain risk if "{company}" experiences a disruption?',
        {"company": company},
    )


_ROUTE_TEMPLATES: dict[str, list[tuple[str, TemplateFn]]] = {
    RouteName.exact_lookup.value: [
        ("exact_role", _tmpl_exact_role),
        ("exact_location", _tmpl_exact_location),
        ("exact_products", _tmpl_exact_products),
        ("exact_employment", _tmpl_exact_employment),
    ],
    RouteName.structured_sql.value: [
        ("sql_category_count", _tmpl_sql_category_count),
        ("sql_industry_list", _tmpl_sql_industry_list),
        ("sql_employment_threshold", _tmpl_sql_employment_threshold),
        ("sql_oem_filter", _tmpl_sql_oem_filter),
        ("sql_role_group_employment", _tmpl_sql_role_group_employment),
    ],
    RouteName.geo_search.value: [
        ("geo_near", _tmpl_geo_near),
        ("geo_radius", _tmpl_geo_radius),
        ("geo_closest", _tmpl_geo_closest),
    ],
    RouteName.keyword_search.value: [
        ("keyword_phrase", _tmpl_keyword_phrase),
        ("keyword_role", _tmpl_keyword_role),
    ],
    RouteName.vector_search.value: [
        ("vector_risk", _tmpl_vector_risk),
        ("vector_role_topic", _tmpl_vector_role_topic),
    ],
    RouteName.hybrid_search.value: [
        ("hybrid_category_evidence", _tmpl_hybrid_category_evidence),
        ("hybrid_location_evidence", _tmpl_hybrid_location_evidence),
    ],
    RouteName.disruption_analysis.value: [
        ("disruption_alternatives", _tmpl_disruption_alternatives),
        ("disruption_risk", _tmpl_disruption_risk),
    ],
}

# Edge routes need no KB entities — fixed phrasing pools, sampled without
# replacement up to the pool size (duplicates would test nothing new).
_EDGE_ROUTE_QUESTIONS: dict[str, list[str]] = {
    RouteName.no_retrieval.value: [
        "Hi there!",
        "Hello, how are you?",
        "Thanks for your help.",
        "Thank you so much.",
        "Good morning.",
        "Good afternoon.",
        "What can you do?",
        "Who are you?",
    ],
    RouteName.out_of_domain.value: [
        "What's the weather like today?",
        "What's the current stock price of Apple?",
        "Give me a recipe for chocolate chip cookies.",
        "Who won the football game last night?",
        "What's my horoscope for today?",
        "What is the capital of France?",
        "Can you translate 'hello' into Spanish?",
        "Tell me a joke.",
    ],
    RouteName.clarification_needed.value: [
        "What about it?",
        "Can you tell me more about them?",
        "Is that close by?",
        "How many of those are there?",
        "What's its status?",
        "Does it have that?",
    ],
}


def _generate_for_real_route(
    rng: random.Random,
    kb: dict[str, list[Any]],
    route: str,
    templates: list[tuple[str, TemplateFn]],
    count: int,
) -> list[GeneratedQuestion]:
    out: list[GeneratedQuestion] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = count * 20  # generous cap so a sparse KB column can't loop forever
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        template_id, fn = rng.choice(templates)
        result = fn(rng, kb)
        if result is None:
            continue
        question, entities_used = result
        if question in seen:
            continue
        seen.add(question)
        out.append(
            GeneratedQuestion(
                question=question,
                expected_route=route,
                template_id=template_id,
                entities_used=entities_used,
            )
        )
    return out


def _generate_for_edge_route(
    rng: random.Random, route: str, count: int
) -> list[GeneratedQuestion]:
    pool = list(_EDGE_ROUTE_QUESTIONS[route])
    rng.shuffle(pool)
    chosen = pool[: min(count, len(pool))]
    return [
        GeneratedQuestion(
            question=q,
            expected_route=route,
            template_id=f"{route}_fixed",
            entities_used={},
        )
        for q in chosen
    ]


def generate_questions(
    count_per_route: int = 15,
    seed: int = 42,
    kb_path: Path = DEFAULT_KB_PATH,
) -> list[GeneratedQuestion]:
    """Generate ``count_per_route`` questions for each route (fewer for edge
    routes, which draw from a small fixed phrasing pool without duplicates).
    """
    rng = random.Random(seed)
    kb = load_kb_values(kb_path)

    questions: list[GeneratedQuestion] = []
    for route, templates in _ROUTE_TEMPLATES.items():
        questions.extend(_generate_for_real_route(rng, kb, route, templates, count_per_route))
    for route in _EDGE_ROUTE_QUESTIONS:
        questions.extend(_generate_for_edge_route(rng, route, count_per_route))

    rng.shuffle(questions)
    return questions
