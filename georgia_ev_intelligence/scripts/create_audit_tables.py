"""
Create the retrieval-audit Postgres tables.

Run once before enabling the new pipeline:

    python scripts/create_audit_tables.py

The script is idempotent — re-running is safe and only creates tables
that do not already exist. It DOES NOT modify gev_companies, gev_documents,
gev_document_chunks, gev_extracted_facts, or gev_eval_results.

Tables created:
  - gev_retrieval_audit       (one row per question)
  - gev_retrieval_candidates  (one row per candidate considered)
  - gev_domain_mapping_rules  (human-approved abstract-term mappings)
"""
from __future__ import annotations

import sys

from shared.db import (
    create_audit_tables,
    get_engine,
    verify_connection,
)
from shared.logger import get_logger

logger = get_logger("scripts.create_audit_tables")


def main() -> int:
    if not verify_connection():
        logger.error("Database connection failed — aborting")
        return 1

    create_audit_tables()
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            __import__("sqlalchemy").text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name LIKE 'gev_%' "
                "ORDER BY table_name"
            )
        ).fetchall()
    logger.info("Public gev_* tables now present:")
    for (name,) in rows:
        logger.info("  - %s", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
