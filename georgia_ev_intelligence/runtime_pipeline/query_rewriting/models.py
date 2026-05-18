"""Pure data models for the query rewriting subsystem.

Defines the structured representation of a parsed user question,
vocabulary term matches, and aggregated match results.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StructuredQuery:
    """The parsed, structured form of a raw user question.

    Each list contains normalised search terms for that field type.
    Empty list means the user did not specify that dimension.
    rewrite_successful=False means the LLM rewriter failed and
    all lists are empty -- pipeline must fall back to hybrid-only.
    """

    original_question: str
    locations: list[str] = field(default_factory=list)
    product_services: list[str] = field(default_factory=list)
    companies: list[str] = field(default_factory=list)
    oems: list[str] = field(default_factory=list)
    ev_supply_chain_roles: list[str] = field(default_factory=list)
    industry_groups: list[str] = field(default_factory=list)
    facility_types: list[str] = field(default_factory=list)
    rewrite_successful: bool = False
    rewrite_latency_ms: float = 0.0

    def has_filters(self) -> bool:
        """True if at least one filter dimension is non-empty."""
        return bool(
            self.locations
            or self.product_services
            or self.companies
            or self.oems
            or self.ev_supply_chain_roles
            or self.industry_groups
            or self.facility_types
        )

    def all_terms(self) -> list[tuple[str, str]]:
        """Return flat list of (term, term_type) pairs for all dimensions.

        term_type values match the term_type column in kb_vocabulary_terms:
          locations        -> "location"
          product_services -> "product_service"
          companies        -> "company"
          oems             -> "primary_oem"
          ev_supply_chain_roles -> "ev_supply_chain_role"
          industry_groups  -> "industry_group"
          facility_types   -> "facility_type"
        """
        mapping: list[tuple[list[str], str]] = [
            (self.locations, "location"),
            (self.product_services, "product_service"),
            (self.companies, "company"),
            (self.oems, "primary_oem"),
            (self.ev_supply_chain_roles, "ev_supply_chain_role"),
            (self.industry_groups, "industry_group"),
            (self.facility_types, "facility_type"),
        ]
        result: list[tuple[str, str]] = []
        for terms, term_type in mapping:
            for term in terms:
                result.append((term, term_type))
        return result


@dataclass(frozen=True)
class TermMatch:
    """A single term found in kb_vocabulary_terms matching a filter."""

    normalized_value: str
    term_type: str
    source_column: str
    row_ids: list[int]
    term_frequency: int
    match_type: str  # "exact" | "trigram" | "semantic"


@dataclass(frozen=True)
class VocabularyMatches:
    """All vocabulary terms matched for a StructuredQuery.

    Carries the resolved set of parent row_ids via intersection and union.
    """

    structured_query: StructuredQuery
    term_matches: list[TermMatch] = field(default_factory=list)
    intersected_row_ids: list[int] = field(default_factory=list)
    union_row_ids: list[int] = field(default_factory=list)
    match_count: int = 0
    has_matches: bool = False
