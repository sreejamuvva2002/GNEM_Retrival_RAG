"""Build parent-child KB chunks, store parents and child vectors in PostgreSQL.

WHY THIS FILE EXISTS
--------------------
This is the offline indexing entry point: it reads ``Normalized_kb.xlsx``,
creates parent and child chunks, and writes them to Neon PostgreSQL with
pgvector embeddings.  It must be run ONCE (or on any KB update) before the
runtime retrieval pipeline can serve queries.

PIPELINE IT TRIGGERS
--------------------
  1. ``data.loader`` reads Normalized_kb.xlsx → list of ``KBRecord`` objects.
  2. ``chunking.operations.build_chunks_for_record()`` creates:
       - 1 ``ParentChunk`` per row (full structured text)
       - 5 ``ChildChunk`` per row (identity, product_role, oem_relationship,
         location_employment, classification)
  3. ``postgres_store`` writes parent_chunks to the ``parent_chunks`` table.
  4. ``pgvector_store`` embeds child chunk texts with ``"search_document:"``
     prefix and writes to the ``child_chunks`` table (with pgvector embeddings).

FLAGS
-----
  ``--recreate-child-table``  Drop and recreate the child_chunks table (full
                               re-index).  Use after KB schema changes.
  ``--dry-run``               Print chunk previews without writing to DB.
  ``--preview N``             Show the first N parent chunks in dry-run mode.

CORRECTNESS CONTRACT
--------------------
- Running this script twice is safe: upsert semantics prevent duplicates.
- The embedding dimension (768) must match the ``vector(768)`` column type
  in the DB schema and the ``EMBEDDING_MODEL`` in config.
- After re-indexing, restart the BM25 retriever (its in-memory index is
  stale until the next process start).

Usage:
  # Index the Excel KB (original behaviour — default)

  python -m georgia_ev_intelligence.offline_pipeline.index_pgvector

  # Index only new web documents from raw_documents table
  python -m georgia_ev_intelligence.offline_pipeline.index_pgvector --source web

  # Index both Excel KB and new web documents in one pass
  python -m georgia_ev_intelligence.offline_pipeline.index_pgvector --source all

  # Other flags (unchanged)
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
    ChunkingArtifacts,
)
from georgia_ev_intelligence.offline_pipeline.chunking.parent_chunk import ParentRecord
from georgia_ev_intelligence.offline_pipeline.chunking.relationship import (
    validate_relationships,
    build_child_chunks,
)
from georgia_ev_intelligence.offline_pipeline.chunking.child_chunk import ChildChunk
from georgia_ev_intelligence.offline_pipeline.postgres_store import (
    store_parents_postgres,
    fetch_new_raw_documents,
    update_raw_doc_status,
)
from georgia_ev_intelligence.offline_pipeline.pgvector_store import index_kb_children


# ---------------------------------------------------------------------------
# Excel KB path (original, unchanged)
# ---------------------------------------------------------------------------

def _index_excel(args: argparse.Namespace) -> ChunkingArtifacts:
    df = kb_loader.load()
    artifacts = build_parent_child_chunks(df)
    validate_relationships(artifacts.parents, artifacts.children)

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

    return artifacts


# ---------------------------------------------------------------------------
# Web KB path (new)
# ---------------------------------------------------------------------------

def _index_web(args: argparse.Namespace) -> ChunkingArtifacts:
    """Index raw_documents rows with ingestion_status='new' into pgvector."""
    from georgia_ev_intelligence.offline_pipeline.web_chunk_builder import (
        build_parent_record_from_raw_doc,
    )

    batch_size = getattr(args, "web_batch", 500)
    raw_docs = fetch_new_raw_documents(limit=batch_size)
    print(f"Fetched {len(raw_docs)} new web documents from raw_documents table.")

    if not raw_docs:
        return ChunkingArtifacts(parents=[], children=[])

    parents: list[ParentRecord] = []
    children: list[ChildChunk] = []
    failed_ids: list[str] = []

    for doc in raw_docs:
        try:
            parent = build_parent_record_from_raw_doc(doc)
            parents.append(parent)
            # Re-use the existing per-row child chunk builder
            # We pass an empty pandas Series — child chunk builders that need
            # structured fields will return empty strings for missing fields.
            import pandas as pd
            row = pd.Series(doc)
            children.extend(build_child_chunks(parent, row))
        except Exception as exc:
            print(f"  [warn] Skipping {doc.get('doc_id', '?')}: {exc}")
            failed_ids.append(doc["doc_id"])

    if args.preview and parents:
        print(f"\nWeb parents: {len(parents)}  Web children: {len(children)}\n")
        for child in children[: args.preview]:
            print(f"[{child.chunk_type.value}] {child.chunk_id}")
            print(f"  Parent:  {child.parent_record_id}")
            print(f"  Text:    {child.embedding_text[:120]}")
            print()

    if failed_ids and not args.dry_run:
        update_raw_doc_status(failed_ids, "error", "chunk build failed")

    return ChunkingArtifacts(parents=parents, children=children)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index Georgia EV KB chunks into PostgreSQL + pgvector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument(
        "--source",
        choices=["excel", "web", "all"],
        default="excel",
        help=(
            "Which KB source to index. "
            "'excel' = original GNEM Excel KB (default); "
            "'web' = new raw_documents rows only; "
            "'all' = both."
        ),
    )
    parser.add_argument(
        "--web-batch",
        type=int,
        default=500,
        metavar="N",
        dest="web_batch",
        help="Max number of new web docs to index per run (default: 500)",
    )
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

    all_parents: list[ParentRecord] = []
    all_children: list[ChildChunk] = []

    # Excel path
    if args.source in ("excel", "all"):
        excel_artifacts = _index_excel(args)
        all_parents.extend(excel_artifacts.parents)
        all_children.extend(excel_artifacts.children)

    # Web path
    if args.source in ("web", "all"):
        web_artifacts = _index_web(args)
        all_parents.extend(web_artifacts.parents)
        all_children.extend(web_artifacts.children)

    if args.dry_run:
        print(
            f"Built {len(all_parents)} parents, "
            f"{len(all_children)} child chunks. "
            "Dry run only; stores not updated."
        )
        return

    if not all_parents:
        print("No documents to index.")
        return

    combined = ChunkingArtifacts(parents=all_parents, children=all_children)

    pg_count = store_parents_postgres(combined.parents)
    print(f"Stored {pg_count} parent chunks in PostgreSQL (parent_chunks table).")

    stats = index_kb_children(
        combined,
        model_name=args.model,
        recreate=args.recreate_child_table,
    )
    print(
        f"Indexed {stats.chunks_indexed} child chunks into pgvector (child_chunks table) "
        f"with {stats.vector_size}-dim vectors from {stats.embedding_model}."
    )

    # Mark successfully indexed web docs
    if args.source in ("web", "all") and combined.parents:
        web_doc_ids = [
            p.raw_row.get("doc_id")
            for p in combined.parents
            if p.source_type != "excel" and isinstance(p.raw_row, dict)
        ]
        web_doc_ids = [d for d in web_doc_ids if d]
        if web_doc_ids:
            update_raw_doc_status(web_doc_ids, "indexed")
            print(f"Marked {len(web_doc_ids)} web docs as indexed in raw_documents.")


if __name__ == "__main__":
    main()
