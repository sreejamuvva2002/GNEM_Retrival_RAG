"""Configurable lexicons for deterministic query analysis.

Contains only generic query-language words and schema display labels.
No KB-specific values (company names, tiers, locations, etc.).
"""
from __future__ import annotations

# operation keyword -> normalized operation name
OPERATION_WORDS: dict[str, str] = {
    "show": "list",
    "list": "list",
    "find": "list",
    "give": "list",
    "which": "list",
    "what": "list",
    "display": "list",
    "get": "list",
    "identify": "list",
    "provide": "list",
    "count": "count",
    "how": "count",
    "many": "count",
    "number": "count",
    "total": "count",
    "compare": "compare",
    "versus": "compare",
    "vs": "compare",
}

# Multi-word operation phrases checked before single tokens
OPERATION_PHRASES: tuple[tuple[str, str], ...] = (
    ("give me", "list"),
    ("how many", "count"),
)

# singular form used in QueryAnalysisResult.target_entity
TARGET_ENTITY_WORDS: dict[str, str] = {
    "supplier": "suppliers",
    "suppliers": "suppliers",
    "company": "companies",
    "companies": "companies",
    "facility": "facilities",
    "facilities": "facilities",
    "manufacturer": "manufacturers",
    "manufacturers": "manufacturers",
    "oem": "oems",
    "oems": "oems",
}

CONNECTOR_WORDS: frozenset[str] = frozenset({
    "in",
    "at",
    "on",
    "for",
    "with",
    "by",
    "from",
    "to",
    "of",
    "and",
    "or",
    "the",
    "a",
    "an",
    "that",
    "who",
    "are",
    "is",
    "was",
    "were",
    "be",
    "been",
    "being",
    "involved",
    "including",
    "within",
    "across",
    "among",
    "between",
    "about",
    "into",
    "through",
    "during",
    "than",
    "then",
    "also",
    "only",
    "all",
    "any",
    "some",
    "me",
})

STOPWORDS_FOR_AMBIGUITY_EXTRACTION: frozenset[str] = CONNECTOR_WORDS | frozenset({
    "show",
    "list",
    "find",
    "give",
    "which",
    "what",
    "display",
    "get",
    "identify",
    "provide",
    "count",
    "how",
    "many",
    "number",
    "total",
    "compare",
    "versus",
    "vs",
})

# Schema column keys -> human-readable labels (not KB values)
SOURCE_COLUMN_DISPLAY: dict[str, str] = {
    "company": "Company",
    "category": "Category",
    "industry_group": "Industry Group",
    "updated_location": "Location",
    "address": "Address",
    "primary_facility_type": "Facility Type",
    "ev_supply_chain_role": "EV Supply Chain Role",
    "primary_oems": "Primary OEMs",
    "supplier_or_affiliation_type": "Supplier Affiliation Type",
    "product_service": "Product / Service",
    "ev_battery_relevant": "EV Battery Relevant",
    "classification_method": "Classification Method",
    "employment": "Employment",
    "latitude": "Latitude",
    "longitude": "Longitude",
}

# Tokens that should stay uppercase in canonical display
_CANONICAL_ACRONYMS: frozenset[str] = frozenset({
    "oem",
    "oems",
    "ev",
    "usa",
})
