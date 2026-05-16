"""
Build KB vocabulary term index in PostgreSQL.

Usage:
  python -m georgia_ev_intelligence.offline_pipeline.index_vocabulary
  python -m georgia_ev_intelligence.offline_pipeline.index_vocabulary --dry-run
  python -m georgia_ev_intelligence.offline_pipeline.index_vocabulary --skip-embeddings
  python -m georgia_ev_intelligence.offline_pipeline.index_vocabulary --dry-run --skip-embeddings
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from georgia_ev_intelligence.shared import config
from georgia_ev_intelligence.shared.data import loader as kb_loader
from georgia_ev_intelligence.offline_pipeline.vocabulary_indexing.service import (
    index_vocabulary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and index KB vocabulary terms into PostgreSQL."
    )
    parser.add_argument(
        "--model",
        default=config.EMBEDDING_MODEL,
        help="Embedding model name (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.PGVECTOR_BATCH_SIZE,
        help="Batch size for embeddings and DB inserts (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract terms and export Excel, skip database operations.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (faster for testing).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for Excel export (default: project outputs dir).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Step 1: Load normalized DataFrame
    print("Loading normalized KB DataFrame...")
    df = kb_loader.load()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from KB.\n")

    # Step 2: Run vocabulary indexing pipeline
    output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = index_vocabulary(
        df,
        model_name=args.model,
        batch_size=args.batch_size,
        output_dir=output_dir,
        skip_embeddings=args.skip_embeddings,
        skip_db=args.dry_run,
    )

    # Step 3: Print summary
    print("\n" + "=" * 50)
    print("  VOCABULARY INDEXING SUMMARY")
    print("=" * 50)
    print(f"  Rows processed:       {stats.rows_processed}")
    print(f"  Columns processed:    {stats.columns_processed}")
    print(f"  Unique terms:         {stats.terms_extracted}")
    print(f"  Terms with vectors:   {stats.terms_with_vectors}")
    print(f"  Vector dimension:     {stats.vector_size}")
    print(f"  Embedding model:      {stats.embedding_model}")
    print(f"  PostgreSQL table:     {stats.table_name}")
    if stats.excel_path:
        print(f"  Excel export:         {stats.excel_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
