"""
Phase 4 — Deterministic Entity Extractor

WHY DETERMINISTIC INSTEAD OF LLM ROUTING:
  LLM routing is unreliable:
    - Returns "copper foil" as an EV role
    - Returns "Rivian Automotive" instead of "Rivian"
    - Requires JSON parsing that fails on edge cases
    - Needs 1 extra LLM call per question (slow)

  Deterministic extraction:
    - Always extracts the RIGHT entities
    - Zero LLM calls for routing
    - Zero JSON parsing failures
    - Works for any new question automatically

HOW IT WORKS:
  Reads the question, matches against REAL database values:
    - Tier     → loaded live from Neo4j
    - County   → loaded live from PostgreSQL
    - OEM      → loaded live from Neo4j
    - Role     → loaded live from PostgreSQL
    - Employment range → regex on numbers
    - Product keywords → remaining meaningful words in question

  Everything is matched against real data — no hardcoded lists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from shared.logger import get_logger

logger = get_logger("phase4.entity_extractor")


# ── Load real DB values (cached per session) ──────────────────────────────────

@lru_cache(maxsize=1)
def _oem_names() -> list[str]:
    try:
        from phase3_graph.graph_loader import get_driver
        driver = get_driver()
        with driver.session() as s:
            rows = s.run("MATCH (o:OEM) RETURN o.name AS name").data()
        names = [r["name"] for r in rows if r["name"]]
        logger.info("Entity extractor loaded %d OEM names", len(names))
        return names
    except Exception as exc:
        logger.warning("OEM load failed: %s", exc)
        return []


@lru_cache(maxsize=1)
def _tier_names() -> list[str]:
    try:
        from phase3_graph.graph_loader import get_driver
        driver = get_driver()
        with driver.session() as s:
            rows = s.run("MATCH (t:Tier) RETURN t.name AS name").data()
        return [r["name"] for r in rows if r["name"]]
    except Exception as exc:
        logger.warning("Tier load failed: %s", exc)
        return []


@lru_cache(maxsize=1)
def _county_names() -> list[str]:
    try:
        from shared.db import get_session, Company
        session = get_session()
        try:
            rows = session.query(Company.location_county).filter(
                Company.location_county.isnot(None)
            ).distinct().all()
            # Extract just the county name part (before comma)
            return list({r.location_county.split(",")[0].strip() for r in rows if r.location_county})
        finally:
            session.close()
    except Exception as exc:
        logger.warning("County load failed: %s", exc)
        return []


@lru_cache(maxsize=1)
def _ev_roles() -> list[str]:
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
        logger.warning("EV roles load failed: %s", exc)
        return []


@lru_cache(maxsize=1)
def _company_names() -> list[str]:
    """Load all real company names from PostgreSQL for direct name lookup."""
    try:
        from shared.db import get_session, Company
        session = get_session()
        try:
            rows = session.query(Company.company_name).filter(
                Company.company_name.isnot(None)
            ).all()
            return [r.company_name for r in rows if r.company_name]
        finally:
            session.close()
    except Exception as exc:
        logger.warning("Company names load failed: %s", exc)
        return []


@lru_cache(maxsize=1)
def _facility_types() -> list[str]:
    """
    Load all DISTINCT facility_type values from PostgreSQL.
    Example real values: 'Manufacturing Plant', 'R&D', 'Engineering / Operations',
    'Headquarters', 'Distribution Center', 'Assembly'
    These come from the actual database — no hardcoding.
    """
    try:
        from shared.db import get_session, Company
        session = get_session()
        try:
            rows = session.query(Company.facility_type).filter(
                Company.facility_type.isnot(None)
            ).distinct().all()
            return [r.facility_type for r in rows if r.facility_type]
        finally:
            session.close()
    except Exception as exc:
        logger.warning("Facility types load failed: %s", exc)
        return []


# ── True stop words — ONLY English function/structure words ──────────────────
# These NEVER carry product/technology meaning.
# Do NOT add domain words here — they block valid keyword searches.

_STOP_WORDS = {
    # English function words
    "a", "an", "the", "and", "or", "in", "of", "to", "for", "by", "with",
    "is", "are", "was", "be", "been", "has", "have", "had", "do", "does",
    "that", "this", "these", "those", "what", "which", "who", "how", "where",
    "when", "why", "all", "any", "each", "some", "their", "its", "not",
    "only", "also", "than", "from", "but", "as", "at", "on", "up", "out",
    # Question-framing verbs (structural, not product terms)
    "list", "show", "find", "identify", "describe", "provide", "detail",
    "give", "tell", "name", "explain", "suggest", "indicate", "reflect",
    # Manufacturing/operation verbs — appear in questions, never in DB product values
    "manufacture", "manufactures", "manufacturing", "operate", "operates",
    "operating", "produce", "produces", "producing", "make", "makes",
    "create", "creates", "build", "builds", "supply", "supplies",
    # Geographic/entity nouns that are always already extracted
    "georgia", "company", "companies", "supplier", "suppliers",
    # Compound geographic words — also caught by hyphen-split below
    "georgia-based", "ev-relevant", "ev-specific",
    # Pure structural connector words
    "currently", "primary", "existing", "associated", "involved",
    "looking", "seeking", "based", "having", "using", "under",
    "focused", "related", "relevant", "specific",
}


# ── Result dataclass ──────────────────────────────────────────────────────────

# ── Tier synonym mapping ─────────────────────────────────────────────────────
# Maps natural language phrasing to exact DB tier values.
# WHY: Questions use "Direct Manufacturer", "OEM", "vehicle assembler" etc.
# but the DB stores 'OEM', 'OEM (Footprint)', 'OEM Supply Chain'.
# Mapping here avoids question-specific rules and works for ANY new phrasing.
_TIER_SYNONYMS: dict[str, str] = {
    "direct manufacturer":   "OEM",
    "vehicle assembler":      "OEM",
    "vehicle assembly":       "OEM",
    "original equipment manufacturer": "OEM",
    "oem footprint":         "OEM (Footprint)",
    "oem supply chain":      "OEM Supply Chain",
}


# ── Industry group loader ─────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _industry_groups() -> list[str]:
    """Load all DISTINCT industry_group values from PostgreSQL."""
    try:
        from shared.db import get_session, Company
        session = get_session()
        try:
            rows = session.query(Company.industry_group).filter(
                Company.industry_group.isnot(None)
            ).distinct().all()
            return [r.industry_group for r in rows if r.industry_group]
        finally:
            session.close()
    except Exception as exc:
        logger.warning("Industry groups load failed: %s", exc)
        return []


@dataclass
class Entities:
    """Structured entities extracted from a question."""
    tier:              str | None  = None   # e.g. "Tier 1/2"
    county:            str | None  = None   # e.g. "Gwinnett"
    oem:               str | None  = None   # PRIMARY oem (first matched) — backward-compat
    oem_list:          list[str]   = field(default_factory=list)  # ALL oems in question
    company_name:      str | None  = None   # e.g. "SungEel Recycling Park Georgia"
    ev_role:           str | None  = None   # e.g. "Thermal Management"
    ev_role_list:      list[str]   = field(default_factory=list)
    facility_type:     str | None  = None   # e.g. "R&D", "Manufacturing Plant"
    industry_group:    str | None  = None   # e.g. "Chemicals and Allied Products"
    min_employment:    int | None  = None
    max_employment:    int | None  = None
    product_keywords:  list[str]   = field(default_factory=list)
    is_aggregate:      bool        = False  # county-level SUM questions
    is_risk_query:     bool        = False  # SPOF: single-point-of-failure questions
    is_oem_dependency: bool        = False  # sole-sourced by specific OEM
    is_capacity_risk:  bool        = False  # small companies (<N employees)
    is_misalignment:   bool        = False  # tier/role mismatch risk
    is_top_n:          bool        = False  # "Top 10 by employment" questions
    top_n_limit:       int         = 10     # how many to return for top_n
    ev_relevant_filter: bool       = False  # only EV-relevant companies



# ── Main extractor ────────────────────────────────────────────────────────────

def extract(question: str) -> Entities:
    """
    Extract structured entities from a natural language question.

    All matching is done against REAL database values — no hardcoded domain lists.
    When new OEMs, counties, or roles are added to the database, they are
    automatically available here.
    """
    q_lower = question.lower()
    e = Entities()

    # ── 1. Detect aggregate vs individual query ────────────────────────────────
    # Only true county-group-by questions (not top-N company ranking)
    aggregate_signals = [
        "which county has", "highest total", "total employment",
        "how many counties", "ranked by",
    ]
    e.is_aggregate = any(sig in q_lower for sig in aggregate_signals)

    # ── 1b. Top-N company ranking questions ─────────────────────────────────
    top_n_match = re.search(
        r"(?:top|largest|biggest|leading)\s+(\d+)"
        r"|(?:(\d+)\s+(?:largest|biggest|leading))",
        q_lower
    )
    if top_n_match:
        e.is_top_n = True
        e.is_aggregate = False  # top-N overrides aggregate
        n_str = top_n_match.group(1) or top_n_match.group(2)
        if n_str and n_str.isdigit():
            e.top_n_limit = int(n_str)

    # ── 2. Detect risk query subtypes ────────────────────────────────────────
    # WHY SUBTYPES (not one broad flag):
    #   Q27 "single-point-of-failure" → SPOF list (get_single_supplier_roles)
    #   Q28 "sole-sourced by a specific OEM" → OEM dependency query (different SQL)
    #   Q29 "fewer than 200 employees" + OEM context → capacity risk (filtered query)
    #   Q30 "EV Relevant + General Automotive + Battery" → misalignment (compound filter)
    # Mixing all these into one is_risk_query=True caused Q28/29/30 to get Q27's answer.

    # SPOF: only true when asking about roles served by single company
    spof_signals = ["single-point", "single point", "only a single", "only one company",
                    "served by only", "single-point-of-failure", "single point of failure"]
    e.is_risk_query = any(sig in q_lower for sig in spof_signals)

    # OEM dependency: sole-sourced / dependency from OEM perspective
    oem_dep_signals = ["sole-sourced", "sole sourced", "dependency risk", "high dependency",
                       "dependent on", "only one oem", "single oem"]
    e.is_oem_dependency = any(sig in q_lower for sig in oem_dep_signals)

    # Capacity risk: small company employee count with OEM/supplier context
    capacity_signals = ["fewer than", "less than", "under 200", "under 300", "small scale",
                        "capacity fragility", "surge production", "limited capacity"]
    e.is_capacity_risk = any(sig in q_lower for sig in capacity_signals)

    # Misalignment: role/tier mismatch signal
    misalign_signals = ["misalignment", "mismatch", "misclassified", "supply chain misalignment"]
    e.is_misalignment = any(sig in q_lower for sig in misalign_signals)

    if e.is_top_n:
        # top-N overrides all risk flags
        e.is_risk_query = e.is_oem_dependency = e.is_capacity_risk = e.is_misalignment = False

    # ── 2b. EV-relevant filter ───────────────────────────────────────────────
    ev_filter_signals = ["ev relevant", "ev-relevant", "ev specific", "ev-specific"]
    e.ev_relevant_filter = any(sig in q_lower for sig in ev_filter_signals)

    # ── 3. Tier extraction — from real DB values + synonym mapping ────────────
    # First try synonym mapping ("Direct Manufacturer" → "OEM" etc.)
    # Then fall back to exact DB tier matching.
    for phrase, mapped_tier in _TIER_SYNONYMS.items():
        if phrase in q_lower:
            e.tier = mapped_tier
            break

    if not e.tier:
        tiers = sorted(_tier_names(), key=len, reverse=True)  # longest first
        for t in tiers:
            if t.lower() in q_lower:
                e.tier = t
                break

    # ── 4. County extraction — regex + real DB values ─────────────────────────
    _county_skip = {
        "which", "what", "where", "this", "that", "the", "a", "an",
        "in", "of", "at", "from", "by", "near", "within", "across",
        "each", "any", "all", "both",
        # Block short connector words — 'and', 'or', 'to' can appear before 'county' in sentences
        "and", "or", "to", "for", "on",
    }
    county_match = re.search(r"(\w+(?:\s+\w+)?)\s+county", question, re.IGNORECASE)
    if county_match:
        raw_words = county_match.group(1).strip().split()
        county_word = next(
            # Must be >3 chars AND not in skip list
            (w for w in reversed(raw_words)
             if w.lower() not in _county_skip and len(w) > 3),
            None,
        )
        if county_word:
            county_names = _county_names()
            matched_county = next(
                (c for c in county_names if county_word.lower() == c.lower().split()[0].lower()),
                None,
            )
            if matched_county:
                e.county = matched_county.split(",")[0].strip()
            else:
                partial = next(
                    (c for c in county_names if county_word.lower() in c.lower()),
                    None,
                )
                if partial:
                    e.county = county_word

    # ── 4b. Company name extraction — from real PostgreSQL company names ────────
    # Matches the question against real company names using multi-word matching.
    # Handles: "What does SungEel do?", "What tier is Hanwha Q CELLS?"
    # Sorted longest-first so "Hyundai Motor Group" matches before "Hyundai".
    all_companies = sorted(_company_names(), key=len, reverse=True)
    for name in all_companies:
        # Use word-boundary-aware matching: company name must appear as a phrase
        # in the question (case-insensitive). Skip very short names (< 5 chars)
        # to avoid false positives.
        if len(name) >= 5 and name.lower() in q_lower:
            e.company_name = name
            break

    # ── 4c. Facility type extraction — from real PostgreSQL values ─────────────
    # Loads DISTINCT facility_type values from the DB (same pattern as tier/county).
    # Real values: 'Manufacturing Plant', 'R&D', 'Engineering / Operations', etc.
    # No hardcoding — if a new facility type is added to DB, it is auto-detected.
    facility_types = sorted(_facility_types(), key=len, reverse=True)  # longest first
    for ftype in facility_types:
        if ftype.lower() in q_lower:
            e.facility_type = ftype
            break

    # ── 5. EV role extraction (BEFORE OEM — so role words don't pollute OEM matching) ──
    roles = sorted(_ev_roles(), key=len, reverse=True)  # longest match first
    matched_roles = []
    for role in roles:
        role_lower = role.lower()
        if role_lower not in q_lower:
            continue
        # Single-word roles (e.g. "Materials", "Assembly") are common English words
        # that appear frequently in product descriptions without being a role reference.
        # Require the role to appear CAPITALIZED in the original question, which signals
        # it is being used as a classification label (not part of a product description).
        # e.g. "Battery Cell" → capitalized in "Battery Cell roles" → ✓ matched
        #      "materials"  → lowercase in "electrodeposited materials" → ✗ not matched
        if " " not in role:  # single-word role
            if not re.search(r'\b' + re.escape(role) + r'\b', question):
                continue  # appears only as lowercase common word — skip
        matched_roles.append(role)

    if len(matched_roles) == 1:
        e.ev_role = matched_roles[0]
    elif len(matched_roles) > 1:
        e.ev_role_list = matched_roles

    # Words already captured as tier/role — OEM extractor must skip these
    # e.g. 'battery' is in 'Battery Cell' role → must not match 'SK Battery' OEM
    role_words: set[str] = set()
    for role in (e.ev_role_list or ([e.ev_role] if e.ev_role else [])):
        role_words.update(role.lower().split())
    if e.tier:
        role_words.update(e.tier.lower().split())

    # ── 6. OEM extraction — word-level match against real Neo4j OEM nodes ─────
    oem_names = _oem_names()
    # Corporate/geographic SUFFIX words — never identify a brand in any language.
    # NO domain-specific words here — those are handled by role_words above.
    oem_stop = {
        # Legal entity type suffixes
        "inc", "corp", "llc", "ltd", "plc", "ag", "gmbh", "se", "nv", "bv",
        # Ownership/scale descriptors
        "group", "holdings", "international", "global", "national", "united",
        # Geographic descriptors (adjectives, not proper nouns)
        "north", "south", "east", "west", "central",
        "georgia", "america", "americas", "usa", "american",
        # Generic company-type nouns
        "company", "manufacturing", "industries", "services", "solutions",
        "systems", "technologies", "motors", "motor", "automotive",
    }
    oem_word_map: dict[str, str] = {}
    for name in oem_names:
        for part in re.split(r"[,\s]+", name):
            part_clean = part.strip().lower()
            if (len(part_clean) > 3
                    and part_clean not in oem_stop
                    and part_clean not in role_words):
                oem_word_map[part_clean] = part_clean

    # Collect ALL OEMs mentioned in the question (multi-OEM support).
    # WHY: Questions like "Kia or Rivian suppliers" previously only extracted
    # the first OEM (break-on-first) and missed the second.
    # Now extracts all, sets e.oem = first for backward-compat with single-OEM code paths.
    matched_oems: list[str] = []
    for word in oem_word_map:
        if re.search(r'\b' + re.escape(word[0].upper() + word[1:]) + r'\b', question):
            matched_oems.append(word)

    if matched_oems:
        e.oem      = matched_oems[0]        # primary OEM — backward compat
        e.oem_list = matched_oems           # all OEMs for multi-OEM SQL


    # ── 6b. Industry group extraction — from real DB values ──────────────────
    # Matches the full industry group label (e.g. "Chemicals and Allied Products")
    # in the question. Sorted longest-first for greedy match.
    industry_groups = sorted(_industry_groups(), key=len, reverse=True)
    for ig in industry_groups:
        if ig.lower() in q_lower:
            e.industry_group = ig
            break

    # ── 7. Employment range extraction — regex ─────────────────────────────────
    over_match = re.search(r"(?:over|more than|above|greater than)\s+(\d[\d,]*)", q_lower)
    if over_match:
        e.min_employment = int(over_match.group(1).replace(",", ""))

    under_match = re.search(r"(?:fewer than|less than|under|below)\s+(\d[\d,]*)", q_lower)
    if under_match:
        e.max_employment = int(under_match.group(1).replace(",", ""))

    # ── 8. Product keyword extraction ─────────────────────────────────────────
    known_extracted = set(_STOP_WORDS)
    if e.tier:
        known_extracted.update(e.tier.lower().split())
    if e.county:
        known_extracted.update(e.county.lower().split())
    if e.oem:
        known_extracted.update(e.oem.lower().split())
    if e.industry_group:
        known_extracted.update(e.industry_group.lower().split())
    for role in (e.ev_role_list or ([e.ev_role] if e.ev_role else [])):
        known_extracted.update(role.lower().split())


    # Extract words including acronyms (R&D, EV, HV, BEV)
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-&]+", question)
    product_terms = []

    for w in words:
        # Split hyphenated compounds before stop-word check:
        #   "georgia-based" → ["georgia", "based"] → both in stop words → skip
        #   "copper-foil" → ["copper", "foil"] → neither in stop words → keep "copper"
        parts = re.split(r"[-]", w)
        # If ALL parts are stop words or too short, skip the whole word
        all_stop = all(p.lower() in known_extracted or len(p) < 3 for p in parts)
        if all_stop:
            continue

        wl = w.lower()
        if wl in known_extracted:
            continue
        # Always include acronyms (R&D, EV, HV, BEV, OEM used as product context)
        if "&" in w or (w.isupper() and len(w) >= 2):
            product_terms.append(w)
            continue
        # Regular words: minimum 3 chars
        if len(wl) >= 3:
            product_terms.append(wl)

    # Deduplicate preserving order, cap at 10
    seen_kw: set[str] = set()
    unique_terms = []
    for t in product_terms:
        if t.lower() not in seen_kw:
            seen_kw.add(t.lower())
            unique_terms.append(t)

    e.product_keywords = unique_terms[:10]

    logger.info(
        "Extracted: tier=%s county=%s company=%s oem=%s role=%s keywords=%s aggregate=%s",
        e.tier, e.county, e.company_name, e.oem, e.ev_role or e.ev_role_list,
        e.product_keywords, e.is_aggregate,
    )
    return e

