"""High-level offline chunking operations."""
from __future__ import annotations

import pandas as pd

from ..child_chunk import ChildChunk
from ..parent_chunk import build_parent_record
from ..relations import build_child_chunk


def build_parent_child_chunks(df: pd.DataFrame) -> list[ChildChunk]:
    """Build one child embedding chunk for each structured KB parent row."""
    chunks: list[ChildChunk] = []

    for idx, row in df.reset_index(drop=True).iterrows():
        parent = build_parent_record(row)
        chunks.append(build_child_chunk(idx, parent))

    return chunks


def chunks_to_dataframe(chunks: list[ChildChunk]) -> pd.DataFrame:
    """Return the image-schema fields as a DataFrame for inspection/export."""
    rows = []
    for chunk in chunks:
        parent = chunk.parent
        rows.append(
            {
                "Record_ID": parent.record_id,
                "Company": parent.company,
                "Company_Clean": parent.company_clean,
                "Employment": parent.employment,
                "Product / Service": parent.product_service,
                "County": parent.county,
                "Tier/Category (heuristic)": parent.tier_category_heuristic,
                "Tier_Level": parent.tier_level,
                "Tier_Confidence": parent.tier_confidence,
                "OEM (GA)": parent.oem_ga,
                "Industry Group": parent.industry_group,
                "Industry_Code": parent.industry_code,
                "Industry_Name": parent.industry_name,
                "PDF Page": parent.pdf_page,
                "Is_Announcement": parent.is_announcement,
                "Chunk_ID": chunk.chunk_id,
                "Embedding_Text": chunk.embedding_text,
                "Char_Count": len(chunk.embedding_text),
                "Token_Estimate": round(len(chunk.embedding_text) / 4),
            }
        )
    return pd.DataFrame(rows)
