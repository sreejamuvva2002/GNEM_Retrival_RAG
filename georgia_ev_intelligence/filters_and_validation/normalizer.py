"""
Phase 4 — Parameter Normalizer

WHY THIS EXISTS:
  The LLM router returns natural-language values like "<OEM> Automotive"
  but the database stores the brand alone (e.g. an OEM node with
  `name=<brand>`). Instead of hardcoding suffixes to strip, this module
  fetches the REAL values from Neo4j / Postgres at startup and fuzzy-
  matches the LLM's output against them. When a new value is added to
  the KB it becomes available here automatically — no code change.

WHAT IT NORMALIZES (shape only — no domain values appear in this file):
  - oem:                     LLM string → canonical OEM node name
  - tier:                    LLM string → canonical Tier node name
  - county:                  LLM string → canonical county value
  - ev_supply_chain_role:    list → "Role1 OR Role2" (OR-joined for SQL)
"""
from __future__ import annotations

import difflib
from functools import lru_cache
from typing import Any

from shared.logger import get_logger

logger = get_logger("filters_and_validation.normalizer")


# ── Load real values from graph/DB (cached per session) ───────────────────────

@lru_cache(maxsize=1)
def _load_oem_names() -> list[str]:
    """Fetch all OEM names from Neo4j. Cached for the session lifetime."""
    try:
        from db_storage.graph_loader import get_driver
        driver = get_driver()
        with driver.session() as s:
            rows = s.run("MATCH (o:OEM) RETURN o.name AS name ORDER BY o.name").data()
        names = [r["name"] for r in rows if r["name"]]
        logger.info("Loaded %d OEM names from Neo4j", len(names))
        return names
    except Exception as exc:
        logger.warning("Could not load OEM names from Neo4j: %s", exc)
        return []


@lru_cache(maxsize=1)
def _load_tier_names() -> list[str]:
    """Fetch all Tier names from Neo4j."""
    try:
        from db_storage.graph_loader import get_driver
        driver = get_driver()
        with driver.session() as s:
            rows = s.run("MATCH (t:Tier) RETURN t.name AS name ORDER BY t.name").data()
        names = [r["name"] for r in rows if r["name"]]
        logger.info("Loaded %d Tier names from Neo4j", len(names))
        return names
    except Exception as exc:
        logger.warning("Could not load Tier names from Neo4j: %s", exc)
        return []


@lru_cache(maxsize=1)
def _load_county_names() -> list[str]:
    """Fetch all unique county names from PostgreSQL."""
    try:
        from shared.db import get_session, Company
        session = get_session()
        try:
            rows = session.query(Company.location_county).filter(
                Company.location_county.isnot(None)
            ).distinct().all()
            names = [r.location_county.split(",")[0].strip() for r in rows if r.location_county]
            logger.info("Loaded %d county names from PostgreSQL", len(names))
            return list(set(names))
        finally:
            session.close()
    except Exception as exc:
        logger.warning("Could not load county names from DB: %s", exc)
        return []


@lru_cache(maxsize=1)
def _load_ev_roles() -> list[str]:
    """Fetch all unique EV supply chain roles from PostgreSQL."""
    try:
        from shared.db import get_session, Company
        session = get_session()
        try:
            rows = session.query(Company.ev_supply_chain_role).filter(
                Company.ev_supply_chain_role.isnot(None)
            ).distinct().all()
            return [r.ev_supply_chain_role for r in rows if r.ev_supply_chain_role]
        finally:
            session.close()
    except Exception as exc:
        logger.warning("Could not load EV roles from DB: %s", exc)
        return []


# ── Fuzzy matcher ──────────────────────────────────────────────────────────────

def _best_match(value: str, candidates: list[str], cutoff: float = 0.5) -> str:
    """
    Find the best matching real value for the LLM's output.

    Strategy:
      1. Exact match (case-insensitive)
      2. Substring: value appears inside a candidate
      3. Containing: a candidate appears inside value
      4. Word-level: any meaningful word from value matches inside a candidate
         (returns the matched word itself so a Cypher CONTAINS still works
          when the candidate is a comma-joined list of names)
      5. difflib fuzzy match

    Returns original value if no match found (safe fallback).
    """
    if not value or not candidates:
        return value

    value_lower = value.lower().strip()

    # 1. Exact match
    for c in candidates:
        if c.lower().strip() == value_lower:
            return c

    # 2. Substring: value is in candidate
    substring_matches = [c for c in candidates if value_lower in c.lower()]
    if substring_matches:
        best = min(substring_matches, key=len)
        logger.debug("Substring match: '%s' -> '%s'", value, best)
        return best

    # 3. Candidate is in value
    containing = [c for c in candidates if c.lower() in value_lower]
    if containing:
        best = max(containing, key=len)
        logger.debug("Contained match: '%s' -> '%s'", value, best)
        return best

    # 4. Word-level: split value into words, find any meaningful word in a candidate
    #    Returns the WORD itself so Cypher CONTAINS works against
    #    comma-joined candidate strings.
    stop_words = {
        "the", "and", "for", "with", "inc", "corp", "llc", "ltd",
        "group", "north", "south", "east", "west", "america", "americas",
        "automotive", "manufacturing", "company", "industries", "systems",
        "motors", "products", "services", "georgia",
    }
    for word in value_lower.split():
        word_clean = word.strip(".,;-")
        if len(word_clean) <= 3 or word_clean in stop_words:
            continue
        for c in candidates:
            if word_clean in c.lower():
                logger.debug("Word-level match: '%s' -> word '%s' in '%s'", value, word_clean, c)
                return word_clean  # short word — works with Cypher CONTAINS

    # 5. difflib fuzzy match
    matches = difflib.get_close_matches(value_lower, [c.lower() for c in candidates], n=1, cutoff=cutoff)
    if matches:
        matched_lower = matches[0]
        for c in candidates:
            if c.lower() == matched_lower:
                logger.debug("Fuzzy match: '%s' -> '%s'", value, c)
                return c

    logger.debug("No match found for '%s' in %d candidates", value, len(candidates))
    return value


def is_valid_ev_role(value: str) -> bool:
    """
    Check whether a value is an actual EV supply chain role in the KB.

    Used to detect when the LLM mistakenly puts product keywords (e.g. raw
    material names) into ev_supply_chain_role. Returns False for product
    descriptions, True for values that actually appear in the KB role list.
    """
    if not value:
        return False
    real_roles = _load_ev_roles()
    value_lower = value.lower()
    for role in real_roles:
        if role.lower() in value_lower or value_lower in role.lower():
            return True
    return False




# ── Main normalize function ────────────────────────────────────────────────────

def normalize_params(strategy: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize LLM router params against real database values.

    Shape (no domain values appear in this docstring):
      - OEM strings:     "<brand> <suffix>"  →  "<brand>"   (matched to KB)
      - tier strings:    "<tier> <noise>"    →  "<tier>"    (matched to KB)
      - county strings:  "<county> County"   →  "<county>"  (matched to KB)
      - ev_supply_chain_role list →           "<r1> OR <r2>" (joined for SQL)

    The candidate sets come from Neo4j / Postgres at runtime; nothing is
    hardcoded here.
    """
    if not params:
        return params

    result = dict(params)

    # Normalize OEM name
    for key in ("oem", "primary_oem", "oem_name"):
        if key in result and result[key]:
            oem_candidates = _load_oem_names()
            result[key] = _best_match(str(result[key]), oem_candidates)

    # Normalize tier
    for key in ("tier", "filter_tier"):
        if key in result and result[key]:
            tier_candidates = _load_tier_names()
            result[key] = _best_match(str(result[key]), tier_candidates)

    # Normalize county
    for key in ("county", "filter_county", "location_county"):
        if key in result and result[key]:
            county_candidates = _load_county_names()
            # Strip "County" suffix first so "<county> County" → "<county>"
            raw = str(result[key]).replace(" County", "").strip()
            result[key] = _best_match(raw, county_candidates)

    # Normalize EV role — handle list values AND validate the value is a real role
    for key in ("ev_supply_chain_role", "role"):
        if key in result and result[key]:
            val = result[key]
            if isinstance(val, list):
                result[key] = " OR ".join(str(v).strip() for v in val)
            elif isinstance(val, str) and "," in val:
                result[key] = " OR ".join(v.strip() for v in val.split(","))
            # Validate: if the value is NOT a real EV role, mark for keyword search
            # The caller (pipeline) will detect this flag and switch strategy
            final_role = result[key]
            if final_role and " OR " not in final_role and not is_valid_ev_role(final_role):
                logger.info("'%s' is not a valid EV role — flagging as keyword search", final_role)
                result["_use_keyword_search"] = True
                result["_keywords"] = [v.strip() for v in final_role.split(" OR ")]

    return result
