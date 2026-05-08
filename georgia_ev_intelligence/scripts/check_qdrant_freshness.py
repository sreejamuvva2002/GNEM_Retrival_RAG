"""
Check whether the Qdrant GNEM company index matches the current KB workbook.

Usage:
  python scripts/check_qdrant_freshness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from embeddings_store.index_freshness import audit_company_index


def main() -> None:
    audit = audit_company_index()
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
