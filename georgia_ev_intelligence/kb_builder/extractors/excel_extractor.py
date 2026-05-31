"""Excel extractor using pandas and openpyxl."""
from __future__ import annotations

import io

def extract(excel_bytes: bytes) -> tuple[str, str]:
    """Extract (title, body_text) from raw Excel bytes.

    Returns ("", "") on failure.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required: pip install pandas>=2.0") from exc

    try:
        # Load all sheets
        dfs = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=None)
        
        # We don't have a reliable way to get the file title from raw bytes,
        # but we can at least extract sheet names.
        sheet_names = list(dfs.keys())
        title = "Excel Document"
        if sheet_names:
            title = f"Excel Document - Sheets: {', '.join(sheet_names)}"
            
        body_parts = []
        for sheet_name, df in dfs.items():
            body_parts.append(f"--- Sheet: {sheet_name} ---")
            # Convert dataframe to string (csv format is usually most readable as text)
            body_parts.append(df.to_csv(index=False))
            
        body = "\n\n".join(body_parts).strip()
        return title, body
    except Exception:
        return "", ""
