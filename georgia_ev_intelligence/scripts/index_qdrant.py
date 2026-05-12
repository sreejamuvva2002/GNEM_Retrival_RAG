"""
Build parent-child KB chunks and index them into Qdrant.

Usage:
  python -m georgia_ev_intelligence.scripts.index_qdrant --recreate
  python -m georgia_ev_intelligence.scripts.index_qdrant --dry-run --preview 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from georgia_ev_intelligence import config
from georgia_ev_intelligence.data import loader as kb_loader
from georgia_ev_intelligence.indexing.chunker import build_parent_child_chunks, chunks_to_dataframe
from georgia_ev_intelligence.indexing.qdrant_store import build_client, index_kb_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Georgia EV KB chunks into Qdrant.")
    parser.add_argument("--collection", default=config.QDRANT_COLLECTION)
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--url", default=config.QDRANT_URL)
    parser.add_argument("--local-path", default=config.QDRANT_PATH)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview", type=int, default=0)
    args = parser.parse_args()

    df = kb_loader.load()
    chunks = build_parent_child_chunks(df)

    if args.preview:
        preview_df = chunks_to_dataframe(chunks).head(args.preview)
        print(preview_df.to_string(index=False))

    if args.dry_run:
        print(f"Built {len(chunks)} chunks. Dry run only; Qdrant was not updated.")
        return

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


if __name__ == "__main__":
    main()
