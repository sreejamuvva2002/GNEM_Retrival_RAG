"""PostgreSQL connection helper for the route execution stage.

Mirrors the connection pattern used across ``offline_pipeline`` and
``runtime_pipeline.retrieval`` — a plain psycopg2 connection to the Neon
database. No ORM, no pooling; callers are responsible for closing.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from georgia_ev_intelligence.shared import config

if TYPE_CHECKING:
    import psycopg2


def get_connection() -> "psycopg2.extensions.connection":
    """Return a new psycopg2 connection to the configured Neon database."""
    import psycopg2

    url = config.NEON_DATABASE_URL
    if not url:
        raise RuntimeError(
            "NEON_DATABASE_URL is not set. Add it to your .env file."
        )
    return psycopg2.connect(url)
