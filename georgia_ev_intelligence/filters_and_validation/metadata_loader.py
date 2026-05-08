"""
Phase 4 — KB metadata loader.

Loads distinct values for every CANONICAL_FIELDS column from Postgres at
runtime, with a TTL cache so repeated retrieval calls within the same minute
do not hammer the database.

This is the only sanctioned way for any Phase 4 module to know the set of
real KB values (companies, OEMs, counties, tiers, roles, etc.). Hardcoding
those values inside source code is forbidden; the audit grep tests in
tests/test_07_no_hardcoded_facts.py enforce that.
"""
from __future__ import annotations

import threading
from time import time
from typing import Any, Callable

from shared.db import Company, get_session
from shared.logger import get_logger
from shared.metadata_schema import CANONICAL_FIELDS, HARD_FILTER_FIELDS

logger = get_logger("filters_and_validation.metadata_loader")

_CACHE_TTL_SEC = 600  # 10 minutes — KB rarely changes during a session


class MetadataLoader:
    """Thread-safe TTL cache around distinct-value lookups in gev_companies."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def _cached(self, key: str, loader: Callable[[], Any]) -> Any:
        now = time()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None and (now - hit[0]) < _CACHE_TTL_SEC:
                return hit[1]
        value = loader()
        with self._lock:
            self._cache[key] = (now, value)
        return value

    # ── Public API ───────────────────────────────────────────────────────────

    def distinct(self, field: str) -> list[str]:
        """
        Return the sorted distinct non-null values for a canonical KB field.

        `field` must be a value of CANONICAL_FIELDS (i.e. an actual column on
        gev_companies). Anything else raises so callers cannot smuggle in a
        domain term and have it silently coerced into a column lookup.
        """
        if field not in CANONICAL_FIELDS.values():
            raise ValueError(
                f"unknown KB field {field!r}; must be one of "
                f"{sorted(CANONICAL_FIELDS.values())}"
            )
        return self._cached(f"distinct:{field}", lambda: self._load_distinct(field))

    def sample_distinct_values(self, per_field: int = 5) -> dict[str, list[str]]:
        """
        Return up to `per_field` example values per canonical column.

        Used by text_to_sql / text_to_cypher prompt builders to inject real
        KB values into the schema block at runtime instead of hardcoding
        company / county / OEM names in source code.
        """
        out: dict[str, list[str]] = {}
        for col in CANONICAL_FIELDS.values():
            values = self.distinct(col)
            out[col] = [str(v) for v in values[:per_field]]
        return out

    def kb_columns_supporting(self, term: str) -> list[str]:
        """
        Return canonical hard-filter columns whose distinct values contain
        `term` (case-insensitive substring).

        Used by ambiguity_resolver to decide which columns can ground a
        currently unresolved abstract term.
        """
        if not term:
            return []
        t = term.strip().lower()
        out: list[str] = []
        for col in HARD_FILTER_FIELDS:
            try:
                values = self.distinct(col)
            except Exception:
                continue
            if any(t in str(v).lower() for v in values):
                out.append(col)
        return out

    def invalidate(self) -> None:
        """Drop the cache. Call after a KB ingestion run."""
        with self._lock:
            self._cache.clear()

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_distinct(self, field: str) -> list[str]:
        col = getattr(Company, field, None)
        if col is None:
            logger.warning("metadata_loader: gev_companies has no column %r", field)
            return []
        session = get_session()
        try:
            rows = session.query(col).distinct().all()
        except Exception as exc:
            logger.warning("metadata_loader: distinct(%s) failed — %s", field, exc)
            return []
        finally:
            session.close()
        values = sorted({str(r[0]).strip() for r in rows if r[0] not in (None, "")})
        logger.debug("metadata_loader: loaded %d distinct values for %s", len(values), field)
        return values


# Module-level singleton. Import as: from filters_and_validation.metadata_loader import loader
loader = MetadataLoader()
