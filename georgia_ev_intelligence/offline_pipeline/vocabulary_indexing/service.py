"""Orchestrate the full vocabulary indexing pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from georgia_ev_intelligence.shared import config

from .config import EXCEL_FILENAME, TABLE_NAME
from .embedding_service import generate_embeddings, get_vector_dimension
from .excel_exporter import export_to_excel
from .extractor import extract_terms
from .repository import ensure_table, get_connection, insert_batch, truncate_table

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VocabularyIndexStats:
    """Summary statistics from a vocabulary indexing run."""

    terms_extracted: int
    terms_with_vectors: int
    rows_processed: int
    columns_processed: int
    vector_size: int
    embedding_model: str
    table_name: str
    excel_path: Path | None


def index_vocabulary(
    df: pd.DataFrame,
    model_name: str | None = None,
    batch_size: int | None = None,
    output_dir: Path | None = None,
    skip_embeddings: bool = False,
    skip_db: bool = False,
) -> VocabularyIndexStats:
    """Run the full vocabulary indexing pipeline.

    Steps:
    1. Extract vocabulary terms from the normalized DataFrame.
    2. Generate embeddings for each term (unless skip_embeddings).
    3. Store terms in PostgreSQL (truncate + insert, unless skip_db).
    4. Export Excel reference file.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized KB DataFrame from loader.load().
    model_name : str, optional
        Override embedding model.
    batch_size : int, optional
        Override batch size for embeddings and DB inserts.
    output_dir : Path, optional
        Directory for Excel export (defaults to config.OUTPUTS_DIR).
    skip_embeddings : bool
        If True, skip embedding generation.
    skip_db : bool
        If True, skip database operations (dry-run mode).

    Returns
    -------
    VocabularyIndexStats
    """
    model_id = model_name or config.EMBEDDING_MODEL
    size = batch_size or config.PGVECTOR_BATCH_SIZE
    out_dir = output_dir or config.OUTPUTS_DIR

    # Step 1: Extract terms
    logger.info("Step 1: Extracting vocabulary terms from DataFrame...")
    terms = extract_terms(df)
    logger.info("Extracted %d unique vocabulary terms.", len(terms))

    if not terms:
        logger.warning("No vocabulary terms extracted. Exiting.")
        return VocabularyIndexStats(
            terms_extracted=0,
            terms_with_vectors=0,
            rows_processed=len(df),
            columns_processed=0,
            vector_size=0,
            embedding_model=model_id,
            table_name=TABLE_NAME,
            excel_path=None,
        )

    # Step 2: Generate embeddings
    vector_size = 0
    if not skip_embeddings:
        logger.info("Step 2: Generating embeddings for %d terms...", len(terms))
        terms, vector_size = generate_embeddings(terms, model_name=model_id, batch_size=size)
        logger.info("Embeddings generated. Vector dimension: %d", vector_size)
    else:
        logger.info("Step 2: Skipping embedding generation (--skip-embeddings).")
        # Still need vector dimension for table creation
        if not skip_db:
            try:
                vector_size = get_vector_dimension(model_id)
            except Exception as exc:
                vector_size = 768
                logger.warning(
                    "Could not determine vector dimension from model "
                    "(using default %d): %s",
                    vector_size,
                    exc,
                )
        else:
            vector_size = 768

    # Step 3: Store in PostgreSQL
    if not skip_db:
        logger.info("Step 3: Storing vocabulary in PostgreSQL...")
        conn = get_connection()
        try:
            ensure_table(conn, vector_size)
            truncate_table(conn)

            total_inserted = 0
            for batch_start in range(0, len(terms), size):
                batch = terms[batch_start : batch_start + size]
                count = insert_batch(conn, batch, page_size=100)
                total_inserted += count

            conn.commit()
            logger.info(
                "Inserted %d terms into %s.", total_inserted, TABLE_NAME
            )
        except Exception:
            conn.rollback()
            logger.exception("Failed to store vocabulary in PostgreSQL.")
            raise
        finally:
            conn.close()
    else:
        logger.info("Step 3: Skipping database operations (--dry-run).")

    # Step 4: Export Excel
    logger.info("Step 4: Exporting vocabulary index to Excel...")
    excel_path = Path(out_dir) / EXCEL_FILENAME
    export_to_excel(terms, excel_path)

    # Compute stats
    terms_with_vectors = sum(1 for t in terms if t.term_vector is not None)
    columns_processed = len(set(t.source_column for t in terms))

    return VocabularyIndexStats(
        terms_extracted=len(terms),
        terms_with_vectors=terms_with_vectors,
        rows_processed=len(df),
        columns_processed=columns_processed,
        vector_size=vector_size,
        embedding_model=model_id,
        table_name=TABLE_NAME,
        excel_path=excel_path,
    )
