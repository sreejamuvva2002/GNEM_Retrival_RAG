import re
import pandas as pd
from .. import config


def load() -> pd.DataFrame:
    df = pd.read_excel(config.GNEM_EXCEL)
    df.columns = [_norm(c) for c in df.columns]
    df = df.dropna(subset=["company"]).reset_index(drop=True)
    df["_row_id"] = df.index
    df = _apply_overrides(df)
    # Ensure numeric columns have proper dtype so they're found by select_dtypes
    if "employment" in df.columns:
        df["employment"] = pd.to_numeric(df["employment"], errors="coerce")
    return df


def _norm(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _apply_overrides(df: pd.DataFrame) -> pd.DataFrame:
    try:
        ov = pd.read_csv(config.EMPLOYMENT_OVERRIDES)
        ov.columns = [_norm(c) for c in ov.columns]
        name_col = next(c for c in ov.columns if "company" in c or "name" in c)
        emp_col = next(c for c in ov.columns if "employ" in c or "override" in c)
        for _, row in ov.iterrows():
            company_name = str(row[name_col]).strip().lower()
            mask = df["company"].str.lower().str.strip() == company_name
            if mask.any():
                df.loc[mask, "employment"] = row[emp_col]
    except Exception:
        pass
    return df
