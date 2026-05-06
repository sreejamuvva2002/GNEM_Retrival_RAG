"""
Audit GNEM company-index freshness in Qdrant.

The KB is the source of truth for structured retrieval. This module checks that
the Qdrant master company points cover the same source rows and carry the same
source fingerprints as the current workbook load.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from phase2_embedding.chunker import (
    COMPANY_CHUNK_SCHEMA_VERSION,
    _company_row_id,
    _company_source_hash,
)

REBUILD_COMMAND = "python -m phase2_embedding.pipeline --companies-only --reembed"


def _payload_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("payload") or record


def audit_company_index(
    kb_companies: list[dict[str, Any]] | None = None,
    qdrant_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Compare current KB rows against Qdrant master company points.

    Args:
        kb_companies: Optional source rows for tests/offline audits.
        qdrant_records: Optional records with either a top-level payload key or
            direct payload fields. If omitted, Qdrant is scrolled live.
    """
    collection = None
    if kb_companies is None:
        from phase1_extraction.kb_loader import load_companies_from_excel

        kb_companies = load_companies_from_excel(apply_overrides=False)

    if qdrant_records is None:
        from phase2_embedding.vector_store import get_collection_name, scroll_points

        collection = get_collection_name()
        qdrant_records = scroll_points(
            filters={
                "source_type": "gnem_excel",
                "chunk_type": "company",
                "chunk_view": "master",
            },
            limit=None,
        )

    expected_hash_by_row_id = {
        _company_row_id(company): _company_source_hash(company)
        for company in kb_companies
    }
    expected_required_by_row_id = {
        _company_row_id(company): {
            "company_name": company.get("company_name"),
            "tier": company.get("tier"),
            "ev_supply_chain_role": company.get("ev_supply_chain_role"),
            "location_county": company.get("location_county"),
            "ev_battery_relevant": company.get("ev_battery_relevant"),
            "chunk_view": "master",
            "chunk_type": "company",
            "source_type": "gnem_excel",
        }
        for company in kb_companies
    }
    expected_row_ids = set(expected_hash_by_row_id)

    indexed_payloads = [_payload_from_record(record) for record in qdrant_records]
    indexed_by_row_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_row_id_count = 0

    for payload in indexed_payloads:
        row_id = payload.get("company_row_id")
        if not row_id:
            missing_row_id_count += 1
            continue
        indexed_by_row_id[str(row_id)].append(payload)

    indexed_row_ids = set(indexed_by_row_id)
    missing_row_ids = sorted(expected_row_ids - indexed_row_ids)
    extra_row_ids = sorted(indexed_row_ids - expected_row_ids)
    duplicate_row_ids = sorted(
        row_id for row_id, payloads in indexed_by_row_id.items() if len(payloads) > 1
    )

    stale_row_ids: set[str] = set()
    missing_source_hash_row_ids: set[str] = set()
    stale_schema_row_ids: set[str] = set()
    schema_versions = Counter()
    required_missing_fields = Counter()

    for payload in indexed_payloads:
        schema = payload.get("kb_schema_version")
        schema_versions[str(schema or "<missing>")] += 1

        row_id = payload.get("company_row_id")
        row_id = str(row_id) if row_id else None

        expected_required = expected_required_by_row_id.get(row_id or "", {})
        for field_name, expected_value in expected_required.items():
            value = payload.get(field_name)
            if expected_value not in (None, "") and (value is None or value == ""):
                required_missing_fields[field_name] += 1

        if schema != COMPANY_CHUNK_SCHEMA_VERSION and row_id:
            stale_schema_row_ids.add(row_id)

        source_hash = payload.get("source_row_hash")
        if not source_hash:
            if row_id:
                missing_source_hash_row_ids.add(row_id)
            continue

        expected_hash = expected_hash_by_row_id.get(row_id or "")
        if expected_hash and source_hash != expected_hash and row_id:
            stale_row_ids.add(row_id)

    stale_row_ids.update(missing_source_hash_row_ids)
    stale_row_ids.update(stale_schema_row_ids)

    ok = not any(
        [
            missing_row_ids,
            extra_row_ids,
            duplicate_row_ids,
            stale_row_ids,
            missing_row_id_count,
            required_missing_fields,
        ]
    )

    return {
        "ok": ok,
        "collection": collection,
        "expected_rows": len(expected_row_ids),
        "indexed_master_rows": len(indexed_payloads),
        "missing_row_ids": missing_row_ids,
        "extra_row_ids": extra_row_ids,
        "duplicate_row_ids": duplicate_row_ids,
        "stale_row_ids": sorted(stale_row_ids),
        "missing_source_hash_row_ids": sorted(missing_source_hash_row_ids),
        "stale_schema_row_ids": sorted(stale_schema_row_ids),
        "missing_row_id_count": missing_row_id_count,
        "schema_versions": dict(sorted(schema_versions.items())),
        "required_missing_fields": dict(sorted(required_missing_fields.items())),
        "recommended_rebuild_command": REBUILD_COMMAND,
    }
