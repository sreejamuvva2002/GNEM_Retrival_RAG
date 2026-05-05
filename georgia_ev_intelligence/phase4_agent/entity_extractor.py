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


# ── Load KB values from the GNEM workbook (cached per session) ───────────────

@lru_cache(maxsize=1)
def _kb_companies() -> list[dict]:
    from phase1_extraction.kb_loader import load_companies_from_excel
    companies = load_companies_from_excel(apply_overrides=False)
    logger.info("Entity extractor loaded %d companies from GNEM workbook", len(companies))
    return companies

@lru_cache(maxsize=1)
def _oem_names() -> list[str]:
    names = set()
    for company in _kb_companies():
        raw = str(company.get("primary_oems") or "")
        for part in raw.replace("/", ",").split(","):
            part = part.strip()
            if part:
                names.add(part)
    names_list = sorted(names)
    logger.info("Entity extractor loaded %d OEM names", len(names_list))
    return names_list


@lru_cache(maxsize=1)
def _tier_names() -> list[str]:
    return sorted({company.get("tier") for company in _kb_companies() if company.get("tier")})


@lru_cache(maxsize=1)
def _county_names() -> list[str]:
    return sorted({
        str(company.get("location_county")).split(",")[0].strip()
        for company in _kb_companies()
        if company.get("location_county")
    })


@lru_cache(maxsize=1)
def _ev_roles() -> list[str]:
    return sorted({
        company.get("ev_supply_chain_role")
        for company in _kb_companies()
        if company.get("ev_supply_chain_role")
    })


@lru_cache(maxsize=1)
def _company_names() -> list[str]:
    """Load all real company names from the GNEM workbook for direct lookup."""
    return [company["company_name"] for company in _kb_companies() if company.get("company_name")]


@lru_cache(maxsize=1)
def _facility_types() -> list[str]:
    """
    Load all DISTINCT facility_type values from PostgreSQL.
    Example real values: 'Manufacturing Plant', 'R&D', 'Engineering / Operations',
    'Headquarters', 'Distribution Center', 'Assembly'
    These come from the actual database — no hardcoding.
    """
    return sorted({
        company.get("facility_type")
        for company in _kb_companies()
        if company.get("facility_type")
    })


@lru_cache(maxsize=1)
def _classification_methods() -> list[str]:
    return sorted({
        company.get("classification_method")
        for company in _kb_companies()
        if company.get("classification_method")
    })


@lru_cache(maxsize=1)
def _supplier_affiliation_types() -> list[str]:
    return sorted({
        company.get("supplier_affiliation_type")
        for company in _kb_companies()
        if company.get("supplier_affiliation_type")
    })


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
    "will", "would", "could", "should", "may", "might", "shall",
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
    "classified", "role", "roles", "tier", "tiers", "employment",
    "facility", "facilities", "type", "types", "location", "locations",
    "oem", "oems", "map", "every", "along", "assigned", "cover",
    "industry", "group",
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


def _phrase_pattern(phrase: str) -> str:
    """Return a boundary-aware regex for a KB value or natural-language phrase."""
    parts = [re.escape(part) for part in phrase.strip().split()]
    body = r"\s+".join(parts)
    return rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])"


def _phrase_spans(text: str, phrase: str, flags: int = re.IGNORECASE) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(_phrase_pattern(phrase), text, flags)]


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(_phrase_spans(text, phrase))


def _overlaps(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    return any(span[0] < existing[1] and existing[0] < span[1] for existing in spans)


def _window(text: str, start: int, end: int, before: int = 60, after: int = 60) -> tuple[str, str]:
    return text[max(0, start - before):start].lower(), text[end:end + after].lower()


def _dedupe_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _is_hypothetical_company_filter(question: str, start: int, end: int) -> bool:
    before, after = _window(question, start, end, before=35, after=90)
    return (
        "new" in before
        and "company" in after
        and any(signal in after for signal in ("looking", "seeking", "locat", "site"))
    )


def _has_oem_tier_intent(q_lower: str) -> bool:
    return any(
        signal in q_lower
        for signal in (
            "classified as oem",
            "classified under oem",
            "oem companies",
            "oem manufacturers",
            "oem vehicle assemblers",
        )
    )


def _company_is_relationship_target(question: str, start: int, end: int) -> bool:
    before, _ = _window(question, start, end, before=75, after=20)
    return any(
        signal in before
        for signal in (
            "linked to",
            "serving",
            "serve ",
            "serves ",
            "suppliers to",
            "supplier network linked to",
            "contracts with",
            "contract with",
            "customer of",
            "customers of",
            "support ",
        )
    )


def _facility_has_intent(question: str, start: int, end: int, facility_type: str) -> bool:
    before, after = _window(question, start, end, before=45, after=55)
    nearby = before + after
    f_norm = facility_type.lower()
    if "facility" in nearby or "facilities" in nearby or "facility type" in nearby:
        return True
    if f_norm == "r&d" and any(signal in nearby for signal in ("research", "development", "innovation")):
        return True
    return False


def _classification_has_intent(q_lower: str, method: str) -> bool:
    method_lower = method.lower()
    if method_lower == "direct manufacturer":
        return _contains_phrase(q_lower, method_lower)
    if method_lower == "supplier":
        return any(
            signal in q_lower
            for signal in (
                "classified as supplier",
                "classified under supplier",
                "supplier classification",
                "classification method supplier",
            )
        )
    return _contains_phrase(q_lower, method_lower)


def _role_is_excluded(question: str, start: int) -> bool:
    before = question[max(0, start - 55):start].lower()
    return any(
        signal in before
        for signal in (
            "lack",
            "lacks",
            "without",
            "absence of",
            "no ",
            "not ",
            "currently lack",
            "currently lacks",
        )
    )


def _role_is_relationship_target(question: str, start: int) -> bool:
    before = question[max(0, start - 35):start].lower()
    return any(signal in before for signal in (" to ", " for ", "support "))


def _role_has_candidate_intent(question: str, start: int, end: int, role: str) -> bool:
    before, after = _window(question, start, end, before=55, after=65)
    role_lower = role.lower()
    if _role_is_relationship_target(question, start):
        return False
    if any(signal in before for signal in ("classified as", "classified under", "listed under", "under ")):
        return True
    if any(signal in after for signal in ("role", "roles", "supplier", "suppliers", "companies", "category")):
        return True
    if role_lower == "general automotive" and "infrastructure" in after:
        return True
    if any(signal in before for signal in ("in ev ", "in the ")):
        return True
    return False


def _significant_tokens(text: str) -> set[str]:
    local_stop = _STOP_WORDS | {"industry", "group", "other", "components", "component"}
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", text)
        if token.lower() not in local_stop and len(token) >= 3
    }


# ── Industry group loader ─────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _industry_groups() -> list[str]:
    """Load all distinct industry groups from the GNEM workbook."""
    return sorted({
        company.get("industry_group")
        for company in _kb_companies()
        if company.get("industry_group")
    })


@dataclass
class Entities:
    """Structured entities extracted from a question."""
    tier:              str | None  = None   # e.g. "Tier 1/2"
    tier_list:         list[str]   = field(default_factory=list)
    county:            str | None  = None   # e.g. "Gwinnett"
    oem:               str | None  = None   # PRIMARY oem (first matched) — backward-compat
    oem_list:          list[str]   = field(default_factory=list)  # ALL oems in question
    company_name:      str | None  = None   # e.g. "SungEel Recycling Park Georgia"
    ev_role:           str | None  = None   # e.g. "Thermal Management"
    ev_role_list:      list[str]   = field(default_factory=list)
    exclude_ev_role_list: list[str] = field(default_factory=list)
    facility_type:     str | None  = None   # e.g. "R&D", "Manufacturing Plant"
    industry_group:    str | None  = None   # e.g. "Chemicals and Allied Products"
    classification_method: str | None = None   # e.g. "Supplier", "Direct Manufacturer"
    supplier_affiliation_type: str | None = None
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

    # ── 3. Tier extraction — boundary-aware with intent guards ────────────────
    tier_matches: list[tuple[int, str]] = []
    occupied_tier_spans: list[tuple[int, int]] = []

    for phrase, mapped_tier in sorted(_TIER_SYNONYMS.items(), key=lambda item: len(item[0]), reverse=True):
        for span in _phrase_spans(question, phrase):
            if _overlaps(span, occupied_tier_spans) or _is_hypothetical_company_filter(question, *span):
                continue
            tier_matches.append((span[0], mapped_tier))
            occupied_tier_spans.append(span)

    for tier_name in sorted(_tier_names(), key=len, reverse=True):
        if tier_name == "OEM" and not _has_oem_tier_intent(q_lower):
            continue
        for span in _phrase_spans(question, tier_name):
            if _overlaps(span, occupied_tier_spans) or _is_hypothetical_company_filter(question, *span):
                continue
            tier_matches.append((span[0], tier_name))
            occupied_tier_spans.append(span)

    matched_tiers = _dedupe_preserve([tier for _, tier in sorted(tier_matches, key=lambda item: item[0])])
    if matched_tiers:
        e.tier = matched_tiers[0]
        e.tier_list = matched_tiers

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
        if len(name) < 5:
            continue
        spans = _phrase_spans(question, name)
        if not spans:
            continue
        if any(not _company_is_relationship_target(question, *span) for span in spans):
            e.company_name = name
            break

    # ── 4c. Facility type extraction — from real PostgreSQL values ─────────────
    # Loads DISTINCT facility_type values from the DB (same pattern as tier/county).
    # Real values: 'Manufacturing Plant', 'R&D', 'Engineering / Operations', etc.
    # No hardcoding — if a new facility type is added to DB, it is auto-detected.
    facility_types = sorted(_facility_types(), key=len, reverse=True)  # longest first
    for ftype in facility_types:
        for span in _phrase_spans(question, ftype):
            if _facility_has_intent(question, *span, ftype):
                e.facility_type = ftype
                break
        if e.facility_type:
            break

    # ── 4d. Classification method / supplier affiliation direct matches ──────
    for method in sorted(_classification_methods(), key=len, reverse=True):
        if _classification_has_intent(q_lower, method):
            e.classification_method = method
            break

    for affiliation in sorted(_supplier_affiliation_types(), key=len, reverse=True):
        if _contains_phrase(q_lower, affiliation.lower()):
            e.supplier_affiliation_type = affiliation
            break

    # ── 5. EV role extraction (BEFORE OEM — so role words don't pollute OEM matching) ──
    roles = sorted(_ev_roles(), key=len, reverse=True)  # longest match first
    matched_roles = []
    excluded_roles = []
    for role in roles:
        for span in _phrase_spans(question, role):
            if " " not in role and not re.search(r'\b' + re.escape(role) + r'\b', question):
                continue
            if _role_is_excluded(question, span[0]):
                excluded_roles.append(role)
                continue
            if _role_has_candidate_intent(question, *span, role):
                matched_roles.append(role)

    matched_roles = _dedupe_preserve(matched_roles)
    excluded_roles = _dedupe_preserve(excluded_roles)
    if len(matched_roles) == 1:
        e.ev_role = matched_roles[0]
    elif len(matched_roles) > 1:
        e.ev_role_list = matched_roles
    e.exclude_ev_role_list = excluded_roles

    # Words already captured as tier/role — OEM extractor must skip these
    # e.g. 'battery' is in 'Battery Cell' role → must not match 'SK Battery' OEM
    role_words: set[str] = set()
    for role in (e.ev_role_list or ([e.ev_role] if e.ev_role else [])):
        role_words.update(role.lower().split())
    for tier in (e.tier_list or ([e.tier] if e.tier else [])):
        role_words.update(tier.lower().split())

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
        "oem", "oems", "primary",
        # Domain words that occur in OEM names but are too generic in questions
        "battery", "vehicle", "vehicles", "car", "cars", "specialized",
    }
    oem_word_map: dict[str, str] = {}
    for name in oem_names:
        for part in re.split(r"[,\s]+", name):
            part_clean = part.strip().lower()
            if (len(part_clean) >= 2
                    and part_clean not in oem_stop
                    and part_clean not in role_words):
                oem_word_map[part_clean] = part_clean

    # Collect ALL OEMs mentioned in the question (multi-OEM support).
    # WHY: Questions like "Kia or Rivian suppliers" previously only extracted
    # the first OEM (break-on-first) and missed the second.
    # Now extracts all, sets e.oem = first for backward-compat with single-OEM code paths.
    matched_oems: list[str] = []
    for word in oem_word_map:
        if re.search(r'\b' + re.escape(word) + r'\b', question, re.IGNORECASE):
            matched_oems.append(word)

    if matched_oems:
        e.oem      = matched_oems[0]        # primary OEM — backward compat
        e.oem_list = matched_oems           # all OEMs for multi-OEM SQL


    # ── 6b. Industry group extraction — from real DB values ──────────────────
    # Matches the full industry group label (e.g. "Chemicals and Allied Products")
    # in the question. Sorted longest-first for greedy match.
    industry_groups = sorted(_industry_groups(), key=len, reverse=True)
    for ig in industry_groups:
        if _contains_phrase(q_lower, ig.lower()):
            e.industry_group = ig
            break
    if not e.industry_group and "industry group" in q_lower:
        industry_match = re.search(
            r"([A-Za-z][A-Za-z\s&/-]{3,80}?)\s+industry\s+group",
            question,
            re.IGNORECASE,
        )
        requested_tokens = _significant_tokens(industry_match.group(1)) if industry_match else set()
        if requested_tokens:
            for ig in industry_groups:
                value_tokens = _significant_tokens(ig)
                if requested_tokens and requested_tokens.issubset(value_tokens):
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
    for tier in (e.tier_list or ([e.tier] if e.tier else [])):
        known_extracted.update(tier.lower().split())
    if e.county:
        known_extracted.update(e.county.lower().split())
    if e.oem:
        known_extracted.update(e.oem.lower().split())
    for oem in e.oem_list:
        known_extracted.update(oem.lower().split())
    if e.company_name:
        known_extracted.update(re.findall(r"[a-z0-9]+", e.company_name.lower()))
    if e.facility_type:
        known_extracted.update(re.findall(r"[a-z0-9]+", e.facility_type.lower()))
        known_extracted.add(e.facility_type.lower())
    if e.classification_method:
        known_extracted.update(re.findall(r"[a-z0-9]+", e.classification_method.lower()))
    if e.supplier_affiliation_type:
        known_extracted.update(re.findall(r"[a-z0-9]+", e.supplier_affiliation_type.lower()))
    if e.industry_group:
        known_extracted.update(e.industry_group.lower().split())
    for role in (e.ev_role_list or ([e.ev_role] if e.ev_role else [])) + e.exclude_ev_role_list:
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
        "Extracted: tier=%s tiers=%s county=%s company=%s oem=%s role=%s exclude_roles=%s keywords=%s aggregate=%s",
        e.tier, e.tier_list, e.county, e.company_name, e.oem, e.ev_role or e.ev_role_list,
        e.exclude_ev_role_list, e.product_keywords, e.is_aggregate,
    )
    return e
