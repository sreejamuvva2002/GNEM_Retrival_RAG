"""Rule-based pre-router (README §11).

Catches strong, unambiguous patterns before the LLM. Only the zero-field routes
(``no_retrieval``, ``out_of_domain``) carry high enough confidence to bypass the
LLM in ``route_service``; aggregate/geo/disruption proposals stay below the
bypass threshold and act only as a hint while the LLM extracts entities/filters.

Even pre-router output still goes through the validator.
"""
from __future__ import annotations

import re

from ..schemas import Operation, RawRoute, RouteName
from . import field_mapping as fm

_GREETING_WORDS = {
    "hi", "hello", "hey", "thanks", "thank", "you", "there", "greetings",
    "good", "morning", "afternoon", "evening",
}


class PreRouter:
    """Deterministic first pass over a normalized question."""

    def route(self, normalized: dict) -> RawRoute | None:
        q = (normalized or {}).get("lowercase", "")
        if not q.strip():
            return None

        if self._is_greeting_or_meta(q):
            return RawRoute(
                route=RouteName.no_retrieval,
                confidence=0.97,
                operation=Operation.direct_response.value,
                reason="Message is only a greeting or meta question.",
            )

        if fm.matches_any(q, fm.OUT_OF_DOMAIN_SIGNALS):
            return RawRoute(
                route=RouteName.out_of_domain,
                confidence=0.9,
                operation=Operation.reject_out_of_domain.value,
                reason="Clear out-of-domain signal.",
            )

        # Disruption is checked before aggregate/geo: it is a strong, specific
        # intent that often co-occurs with location words.
        if fm.matches_any(q, fm.DISRUPTION_SIGNALS):
            return RawRoute(
                route=RouteName.disruption_analysis,
                confidence=0.8,
                operation=Operation.find_alternatives.value,
                reason="Disruption or alternative signal.",
            )

        if fm.matches_any(q, fm.AGGREGATE_SIGNALS):
            return RawRoute(
                route=RouteName.structured_sql,
                confidence=0.8,
                operation=Operation.aggregate_records.value,
                reason="Aggregation signal.",
            )

        if fm.matches_any(q, fm.GEO_SIGNALS):
            return RawRoute(
                route=RouteName.geo_search,
                confidence=0.8,
                operation=Operation.nearby_search.value,
                reason="Location or distance signal.",
            )

        return None

    def _is_greeting_or_meta(self, q: str) -> bool:
        if fm.matches_any(q, fm.META_SIGNALS):
            return True
        words = re.sub(r"[^a-z ]", " ", q).split()
        return bool(words) and all(word in _GREETING_WORDS for word in words)
