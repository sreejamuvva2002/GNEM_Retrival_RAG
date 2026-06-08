"""Disruption analysis executor (composite, execution README §15).

Not a single retrieval call. Steps:
  1. exact-lookup the disrupted company and read its role/category/location;
  2. find geographically nearby companies (PostGIS);
  3. find functionally similar companies (same supply-chain role / category);
  4. rank the union of candidates by role/category match and proximity;
  5. return the ranked alternatives as evidence.
"""
from __future__ import annotations

import logging
from typing import Any

from ..db import get_connection
from ..schemas import STATUS_SUCCESS, ExecutionResult
from . import geo_search

logger = logging.getLogger(__name__)

NEARBY_RADIUS_MILES = 100.0
NEARBY_LIMIT = 100
SIMILAR_LIMIT = 100
RESULT_LIMIT = 25

_PROFILE_SQL = """
SELECT company, ev_supply_chain_role, category, product_service, updated_location,
       latitude, longitude
FROM parent_chunks
WHERE lower(company) = lower(%s)
LIMIT 1;
"""

_SIMILAR_SQL = """
SELECT DISTINCT ON (company)
    company, ev_supply_chain_role, category, product_service, updated_location
FROM parent_chunks
WHERE lower(company) <> lower(%(name)s)
  AND (
    (%(role)s <> '' AND ev_supply_chain_role = %(role)s)
    OR (%(category)s <> '' AND category = %(category)s)
  )
ORDER BY company
LIMIT %(limit)s;
"""


def _fetch_profile(name: str) -> dict[str, Any] | None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_PROFILE_SQL, (name,))
            row = cur.fetchone()
            if row is None:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row))
    finally:
        conn.close()


def _fetch_similar(name: str, role: str, category: str) -> list[dict[str, Any]]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                _SIMILAR_SQL,
                {
                    "name": name,
                    "role": role or "",
                    "category": category or "",
                    "limit": SIMILAR_LIMIT,
                },
            )
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]
    finally:
        conn.close()
    return [dict(zip(cols, row)) for row in rows]


def _rank_candidates(
    profile: dict[str, Any],
    nearby: list[dict[str, Any]],
    similar: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    role = profile.get("ev_supply_chain_role") or ""
    category = profile.get("category") or ""

    distances: dict[str, float] = {}
    for row in nearby:
        company = row.get("company")
        if company is not None:
            distances[company] = row.get("distance_miles")

    candidates: dict[str, dict[str, Any]] = {}

    def _ensure(company: str) -> dict[str, Any]:
        return candidates.setdefault(company, {"company": company})

    for row in similar:
        company = row.get("company")
        if not company:
            continue
        cand = _ensure(company)
        cand.update({
            "ev_supply_chain_role": row.get("ev_supply_chain_role"),
            "category": row.get("category"),
            "product_service": row.get("product_service"),
            "updated_location": row.get("updated_location"),
        })

    for row in nearby:
        company = row.get("company")
        if not company:
            continue
        cand = _ensure(company)
        cand.setdefault("ev_supply_chain_role", row.get("ev_supply_chain_role"))
        cand.setdefault("category", row.get("category"))
        cand.setdefault("updated_location", row.get("updated_location"))

    ranked: list[dict[str, Any]] = []
    for company, cand in candidates.items():
        role_match = bool(role) and cand.get("ev_supply_chain_role") == role
        category_match = bool(category) and cand.get("category") == category
        distance = distances.get(company)

        score = 0.0
        if role_match:
            score += 2.0
        if category_match:
            score += 1.0
        if distance is not None:
            # Closer is better; bounded proximity bonus in [0, 1].
            score += max(0.0, 1.0 - distance / NEARBY_RADIUS_MILES)

        cand["role_match"] = role_match
        cand["category_match"] = category_match
        cand["distance_miles"] = distance
        cand["score"] = round(score, 4)
        ranked.append(cand)

    ranked.sort(
        key=lambda c: (
            c["score"],
            -(c["distance_miles"] if c["distance_miles"] is not None else 1e9),
        ),
        reverse=True,
    )
    return ranked[:RESULT_LIMIT]


def _format_alternatives(company: str, ranked: list[dict[str, Any]]) -> str:
    if not ranked:
        return f"No alternative suppliers found for {company}."
    lines = [f"Ranked alternatives if {company} is disrupted:"]
    for idx, cand in enumerate(ranked, start=1):
        bits = []
        if cand.get("role_match"):
            bits.append("same role")
        if cand.get("category_match"):
            bits.append("same category")
        dist = cand.get("distance_miles")
        if dist is not None:
            bits.append(f"{dist:.1f} mi")
        detail = ", ".join(bits) if bits else "related"
        lines.append(f"{idx}. {cand['company']} ({detail})")
    return "\n".join(lines)


def execute_disruption_analysis(final_route: dict[str, Any]) -> ExecutionResult:
    entities = [str(e).strip() for e in (final_route.get("entities") or []) if str(e).strip()]
    name = entities[0] if entities else ""

    if not name:
        return ExecutionResult(
            route="disruption_analysis",
            status=STATUS_SUCCESS,
            answer="I need the name of the disrupted company to analyse alternatives.",
            evidence={"type": "clarification", "reason": "no entity provided"},
        )

    profile = _fetch_profile(name)
    if profile is None:
        return ExecutionResult(
            route="disruption_analysis",
            status=STATUS_SUCCESS,
            answer=f"I could not find '{name}' in the knowledge base to analyse disruption.",
            evidence={"type": "clarification", "reason": "company not resolved", "query": name},
        )

    # Nearby (best-effort: needs PostGIS geo columns; tolerate absence).
    try:
        nearby = geo_search.nearby_by_company(name, NEARBY_RADIUS_MILES, NEARBY_LIMIT)
    except Exception as exc:
        logger.warning("disruption: nearby lookup failed for %s: %s", name, exc)
        nearby = []

    similar = _fetch_similar(
        name,
        profile.get("ev_supply_chain_role") or "",
        profile.get("category") or "",
    )

    ranked = _rank_candidates(profile, nearby, similar)

    return ExecutionResult(
        route="disruption_analysis",
        status=STATUS_SUCCESS,
        answer=_format_alternatives(profile.get("company") or name, ranked),
        evidence={
            "type": "ranked_alternatives",
            "disrupted_company": profile,
            "alternatives": ranked,
        },
    )
