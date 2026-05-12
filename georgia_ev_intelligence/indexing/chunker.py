"""
Parent-child KB chunking for Qdrant indexing.

The parent is the structured company/facility record. The child is the
embedding-ready text chunk shown in the uploaded schema:

  Company: ...
  Location: ... County, Georgia
  Employment: ... employees
  Industry: ...
  Supply Chain Tier: ... (...)
  Products/Services: ...
  OEM Status: ...
  Status: ...
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


SIC_INDUSTRY_CODES: dict[str, int] = {
    "textile products": 22,
    "paper and allied products": 26,
    "chemicals and allied products": 28,
    "rubber and miscellaneous plastic products": 30,
    "stone, clay, glass and concrete products": 32,
    "primary metal industries": 33,
    "fabricated metal products": 34,
    "machinery except electrical": 35,
    "electronic and other electrical equipment and components": 36,
    "transportation equipment": 37,
    "miscellaneous manufacturing industries": 39,
}


@dataclass(frozen=True)
class ParentRecord:
    record_id: str
    company: str
    company_clean: str
    employment: int | None
    product_service: str
    county: str | None
    tier_category_heuristic: str | None
    tier_level: str | None
    tier_confidence: str | None
    oem_ga: bool
    industry_group: str | None
    industry_code: int | None
    industry_name: str | None
    pdf_page: int | None
    is_announcement: bool
    source_row_id: int

    def payload(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "company": self.company,
            "company_clean": self.company_clean,
            "employment": self.employment,
            "product_service": self.product_service,
            "county": self.county,
            "tier_category_heuristic": self.tier_category_heuristic,
            "tier_level": self.tier_level,
            "tier_confidence": self.tier_confidence,
            "oem_ga": self.oem_ga,
            "industry_group": self.industry_group,
            "industry_code": self.industry_code,
            "industry_name": self.industry_name,
            "pdf_page": self.pdf_page,
            "is_announcement": self.is_announcement,
            "source_row_id": self.source_row_id,
        }


@dataclass(frozen=True)
class ChildChunk:
    chunk_id: str
    parent_record_id: str
    embedding_text: str
    parent: ParentRecord

    def payload(self) -> dict[str, Any]:
        payload = self.parent.payload()
        char_count = len(self.embedding_text)
        payload.update(
            {
                "chunk_id": self.chunk_id,
                "parent_record_id": self.parent_record_id,
                "embedding_text": self.embedding_text,
                "chunk_type": "company_profile",
                "char_count": char_count,
                "token_estimate": round(char_count / 4),
            }
        )
        return _json_safe(payload)


def build_parent_child_chunks(df: pd.DataFrame) -> list[ChildChunk]:
    """Build one child embedding chunk for each structured KB parent row."""
    chunks: list[ChildChunk] = []

    for idx, row in df.reset_index(drop=True).iterrows():
        parent = build_parent_record(row)
        chunk = ChildChunk(
            chunk_id=f"GA_AUTO_{idx:04d}",
            parent_record_id=parent.record_id,
            embedding_text=build_embedding_text(parent),
            parent=parent,
        )
        chunks.append(chunk)

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


def build_parent_record(row: pd.Series) -> ParentRecord:
    company = _clean_scalar(row.get("company")) or ""
    company_clean = _clean_company(company)
    is_announcement = "*" in company
    employment = _to_int(row.get("employment"))
    product_service = _clean_scalar(row.get("product_service")) or ""
    county = _extract_county(row.get("updated_location"))
    tier_level, tier_confidence = _extract_tier(row.get("category"))
    tier_category = _format_tier_category(tier_level, tier_confidence, row.get("category"))
    industry_name, industry_code, industry_group = _extract_industry(row.get("industry_group"))
    oem_ga = _is_oem(row)
    pdf_page = _to_int(row.get("pdf_page"))
    source_row_id = _to_int(row.get("_row_id"))
    if source_row_id is None:
        source_row_id = int(row.name) if row.name is not None else 0

    record_id = _record_id(
        [
            company_clean,
            _clean_scalar(row.get("updated_location")),
            product_service,
            _clean_scalar(row.get("category")),
            _clean_scalar(row.get("primary_facility_type")),
            source_row_id,
        ]
    )

    return ParentRecord(
        record_id=record_id,
        company=company,
        company_clean=company_clean,
        employment=employment,
        product_service=product_service,
        county=county,
        tier_category_heuristic=tier_category,
        tier_level=tier_level,
        tier_confidence=tier_confidence,
        oem_ga=oem_ga,
        industry_group=industry_group,
        industry_code=industry_code,
        industry_name=industry_name,
        pdf_page=pdf_page,
        is_announcement=is_announcement,
        source_row_id=source_row_id,
    )


def build_embedding_text(parent: ParentRecord) -> str:
    """Build the child text chunk in the uploaded example format."""
    lines = [f"Company: {parent.company_clean}"]

    if parent.county:
        lines.append(f"Location: {parent.county} County, Georgia")

    if parent.employment is not None:
        lines.append(f"Employment: {parent.employment} employees")

    if parent.industry_name:
        lines.append(f"Industry: {parent.industry_name}")

    if parent.tier_level:
        tier = parent.tier_level
        if parent.tier_confidence:
            tier = f"{tier} ({parent.tier_confidence})"
        lines.append(f"Supply Chain Tier: {tier}")

    if parent.product_service:
        lines.append(f"Products/Services: {parent.product_service}")

    lines.append(
        "OEM Status: OEM in Georgia"
        if parent.oem_ga
        else "OEM Status: Supplier/Service Provider"
    )
    lines.append("Status: Announced" if parent.is_announcement else "Status: Operational")
    return "\n".join(lines)


def _clean_scalar(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _clean_company(company: str) -> str:
    return re.sub(r"\s+", " ", company.replace("*", "")).strip()


def _to_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None


def _extract_county(location: Any) -> str | None:
    text = _clean_scalar(location)
    if not text:
        return None

    for part in text.split(","):
        part = part.strip()
        match = re.search(r"\b(.+?)\s+county\b", part, flags=re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()

    return None


def _extract_tier(category: Any) -> tuple[str | None, str | None]:
    text = _clean_scalar(category)
    if not text:
        return None, None

    confidence = None
    confidence_match = re.search(r"\(([^)]+)\)", text)
    if confidence_match:
        confidence = confidence_match.group(1).strip().lower()

    if confidence is None and text:
        confidence = "likely"

    tier_match = re.search(r"\btier\s+([0-9]+(?:/[0-9]+)?)\b", text, flags=re.IGNORECASE)
    if tier_match:
        return tier_match.group(1), confidence

    if re.search(r"\boem\b", text, flags=re.IGNORECASE):
        return "OEM", confidence

    return text, confidence


def _format_tier_category(
    tier_level: str | None,
    tier_confidence: str | None,
    raw_category: Any,
) -> str | None:
    raw = _clean_scalar(raw_category)
    if not raw and not tier_level:
        return None

    if raw and "(" in raw and ")" in raw:
        return raw

    label = f"Tier {tier_level}" if tier_level and tier_level != "OEM" else (tier_level or raw)
    if tier_confidence:
        return f"{label} ({tier_confidence})"
    return label


def _extract_industry(industry: Any) -> tuple[str | None, int | None, str | None]:
    name = _clean_scalar(industry)
    if not name:
        return None, None, None

    code_match = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", name)
    if code_match:
        code = int(code_match.group(1))
        clean_name = code_match.group(2).strip()
        return clean_name, code, f"{code}: {clean_name}"

    clean_name = name.strip()
    code = SIC_INDUSTRY_CODES.get(clean_name.lower())
    if code is not None:
        return clean_name, code, f"{code}: {clean_name}"
    return clean_name, None, clean_name


def _is_oem(row: pd.Series) -> bool:
    values = [
        _clean_scalar(row.get("category")),
        _clean_scalar(row.get("classification_method")),
        _clean_scalar(row.get("supplier_or_affiliation_type")),
    ]
    return any(bool(v and re.search(r"\boem\b", v, flags=re.IGNORECASE)) for v in values)


def _record_id(parts: list[Any]) -> str:
    basis = "|".join("" if part is None else str(part).strip().lower() for part in parts)
    return hashlib.md5(basis.encode("utf-8")).hexdigest()[:12]


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            safe[key] = None
        elif isinstance(value, (str, int, float, bool)):
            safe[key] = value
        elif pd.isna(value):
            safe[key] = None
        else:
            safe[key] = str(value)
    return safe

