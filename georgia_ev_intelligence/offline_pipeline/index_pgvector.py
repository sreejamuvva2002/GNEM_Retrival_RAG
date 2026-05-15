"""
Build parent-child KB chunks, store parents in PostgreSQL and child vectors in pgvector.

Usage:
  python -m georgia_ev_intelligence.offline_pipeline.index_pgvector
  python -m georgia_ev_intelligence.offline_pipeline.index_pgvector --recreate-child-table
  python -m georgia_ev_intelligence.offline_pipeline.index_pgvector --dry-run --preview 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from georgia_ev_intelligence.shared import config
from georgia_ev_intelligence.shared.data import loader as kb_loader
from georgia_ev_intelligence.offline_pipeline.chunking.operations import (
    build_parent_child_chunks,
    export_child_chunks_to_xlsx,
    export_parent_chunks_to_xlsx,
)
from georgia_ev_intelligence.offline_pipeline.postgres_store import store_parents_postgres
from georgia_ev_intelligence.offline_pipeline.pgvector_store import index_kb_children


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index Georgia EV KB chunks into PostgreSQL + pgvector."
    )
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument(
        "--recreate-child-table",
        action="store_true",
        help=(
            "Drop and recreate the child_chunks pgvector table. Use this after "
            "changing embedding model/vector dimension."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview", type=int, default=0)
    args = parser.parse_args()

    df = kb_loader.load()
    artifacts = build_parent_child_chunks(df)

    # Always export debug Excel files
    outputs_dir = config.OUTPUTS_DIR
    outputs_dir.mkdir(parents=True, exist_ok=True)
    export_parent_chunks_to_xlsx(artifacts.parents, outputs_dir / "parent_chunks.xlsx")
    export_child_chunks_to_xlsx(artifacts.children, outputs_dir / "child_chunks.xlsx")
    print(f"Exported parent_chunks.xlsx and child_chunks.xlsx to {outputs_dir}")

    if args.preview:
        print(f"Parents: {len(artifacts.parents)}  Children: {len(artifacts.children)}\n")
        for child in artifacts.children[: args.preview]:
            print(f"[{child.chunk_type.value}] {child.chunk_id}")
            print(f"  Parent:  {child.parent_record_id}")
            print(f"  Text:    {child.embedding_text[:120]}")
            print()

    if args.dry_run:
        print(
            f"Built {len(artifacts.parents)} parents, "
            f"{len(artifacts.children)} child chunks. "
            "Dry run only; stores not updated."
        )
        return

    pg_count = store_parents_postgres(artifacts.parents)
    print(f"Stored {pg_count} parent chunks in PostgreSQL (parent_chunks table).")

    stats = index_kb_children(
        artifacts,
        model_name=args.model,
        recreate=args.recreate_child_table,
    )
    print(
        f"Indexed {stats.chunks_indexed} child chunks into pgvector (child_chunks table) "
        f"with {stats.vector_size}-dim vectors from {stats.embedding_model}."
    )


if __name__ == "__main__":
    main()
