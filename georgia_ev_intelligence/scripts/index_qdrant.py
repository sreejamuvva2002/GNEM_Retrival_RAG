"""
Build parent-child KB chunks and index them into Qdrant.

Usage:
  python -m georgia_ev_intelligence.scripts.index_qdrant --recreate
  python -m georgia_ev_intelligence.scripts.index_qdrant --dry-run --preview 3
  python -m georgia_ev_intelligence.scripts.index_qdrant --store-parent-chunks-postgres --postgres-url postgresql://user:password@localhost:5432/georgia_ev --skip-qdrant
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from georgia_ev_intelligence import config
from georgia_ev_intelligence.data import loader as kb_loader
from georgia_ev_intelligence.indexing.chunking import (
    build_parent_child_chunks,
    build_parent_chunks,
    chunks_to_dataframe,
    export_parent_chunks_to_xlsx,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Georgia EV KB chunks into Qdrant.")
    parser.add_argument(
        "--excel",
        type=Path,
        default=None,
        help="Optional KB Excel file path. Defaults to loader discovery.",
    )
    parser.add_argument("--collection", default=config.QDRANT_COLLECTION)
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--url", default=config.QDRANT_URL)
    parser.add_argument("--local-path", default=config.QDRANT_PATH)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-qdrant",
        action="store_true",
        help="Skip child chunk vector indexing after parent chunk work.",
    )
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument(
        "--postgres-url",
        default=config.DATABASE_URL,
        help="PostgreSQL URL for parent chunk storage. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--store-parent-chunks-postgres",
        action="store_true",
        help="Store complete parent chunks in PostgreSQL parent_chunks table.",
    )
    parser.add_argument(
        "--parent-chunks-xlsx",
        type=Path,
        default=None,
        help="Write parent chunks with complete row details to this XLSX file.",
    )
    args = parser.parse_args()

    if args.store_parent_chunks_postgres and not args.postgres_url:
        raise SystemExit(
            "--store-parent-chunks-postgres requires --postgres-url "
            "or DATABASE_URL."
        )

    df = _load_kb(args.excel)
    parents = build_parent_chunks(df)
    chunks = build_parent_child_chunks(df)
    print(f"Built {len(parents)} parent chunks.")

    if args.parent_chunks_xlsx:
        parent_df = export_parent_chunks_to_xlsx(parents, args.parent_chunks_xlsx)
        print(
            f"Wrote {len(parent_df)} parent chunks to "
            f"{args.parent_chunks_xlsx}."
        )

    if args.store_parent_chunks_postgres:
        if args.dry_run:
            print(
                "Dry run enabled: parent chunks would be stored in PostgreSQL, "
                "but no write was performed."
            )
        else:
            _store_parent_chunks_in_postgres(
                parents=parents,
                postgres_url=args.postgres_url,
            )

    if args.preview:
        preview_df = chunks_to_dataframe(chunks).head(args.preview)
        print(preview_df.to_string(index=False))

    if args.dry_run:
        print(f"Built {len(chunks)} chunks. Dry run only; Qdrant was not updated.")
        return

    if args.skip_qdrant:
        print("Skipping Qdrant indexing because --skip-qdrant was provided.")
        return

    from georgia_ev_intelligence.indexing.qdrant_store import build_client, index_kb_chunks

    client = build_client(url=args.url, path=args.local_path)
    stats = index_kb_chunks(
        df,
        collection_name=args.collection,
        model_name=args.model,
        recreate=args.recreate,
        client=client,
    )
    print(
        "Indexed "
        f"{stats.chunks_indexed} chunks into Qdrant collection "
        f"'{stats.collection_name}' with {stats.vector_size}-dim vectors "
        f"from {stats.embedding_model}."
    )


def _load_kb(excel_path: Path | None):
    """
    Load KB data while preventing loader.py from treating script flags as paths.

    loader.py supports an Excel path through sys.argv[1], so this script keeps
    that behavior isolated behind the explicit --excel argument.
    """
    original_argv = sys.argv[:]
    sys.argv = [original_argv[0]]

    if excel_path is not None:
        sys.argv.append(str(excel_path))

    try:
        return kb_loader.load()
    finally:
        sys.argv = original_argv


def _store_parent_chunks_in_postgres(parents, postgres_url: str) -> None:
    from georgia_ev_intelligence.indexing.storage.postgres_parent_store import (
        ParentChunkPostgresStore,
    )

    with ParentChunkPostgresStore(postgres_url) as store:
        stats = store.store_parent_chunks(parents)

    print(
        "Upserted "
        f"{stats.upserted_count} parent chunks into PostgreSQL "
        "(inserted or updated)."
    )
    print(f"Expected parent chunks for current KB: {len(parents)}.")
    print(f"PostgreSQL parent_chunks row count: {stats.table_count}.")


if __name__ == "__main__":
    main()
