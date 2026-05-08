"""
Phase 2 — Hierarchical Parent-Child Chunker

WHY THIS APPROACH:
  - Child chunks (256 tokens) = precise semantic search targets
  - Parent chunks (800 tokens) = full context returned to LLM
  - Pattern: search with small, answer with large
  - Reference: https://python.langchain.com/docs/how_to/parent_document_retriever/
  - Research: 256-512 token sweet spot for retrieval accuracy (2025 consensus)

TWO DOCUMENT TYPES:
  1. GNEM Company records → multi-view chunks per company row
     One master chunk plus focused role/product/OEM/location views
  2. Web documents → hierarchical parent-child split
     Parent = 800 tokens (sent to LLM as context)
     Child  = 256 tokens (used for vector search)
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from shared.logger import get_logger

logger = get_logger("chunking.chunker")

# ── Token size targets ────────────────────────────────────────────────────────
CHILD_TOKEN_TARGET = 256    # Chunk size for semantic search
PARENT_TOKEN_TARGET = 800   # Chunk size for LLM context
CHILD_OVERLAP = 32          # Overlap between child chunks (prevents boundary cuts)

# Approximate chars-per-token for English text (rough estimate for plain text)
# nomic-embed-text uses a BPE tokenizer; 1 token ≈ 4 chars on average
CHARS_PER_TOKEN = 4

CHILD_CHAR_TARGET = CHILD_TOKEN_TARGET * CHARS_PER_TOKEN    # ~1024 chars
PARENT_CHAR_TARGET = PARENT_TOKEN_TARGET * CHARS_PER_TOKEN  # ~3200 chars
OVERLAP_CHARS = CHILD_OVERLAP * CHARS_PER_TOKEN             # ~128 chars
COMPANY_CHUNK_SCHEMA_VERSION = "gnem-company-multiview-v3"
_COMPANY_CHUNK_NAMESPACE = uuid.UUID("6f7986f2-6c4b-4b05-9a35-7e9690f04a4f")


@dataclass
class Chunk:
    """
    A single text chunk ready for embedding.

    chunk_id       : Stable UUID — used as Qdrant point ID
    parent_id      : UUID of parent chunk (None for parent/company chunks)
    chunk_type     : "parent" | "child" | "company"
    text           : Text content to embed
    token_estimate : Rough token count
    metadata       : All metadata fields — stored as Qdrant payload
    """
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    chunk_type: str = "child"   # "parent" | "child" | "company"
    text: str = ""
    token_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: chars / 4. Good enough for chunking decisions."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove junk characters."""
    # Collapse multiple newlines (keep paragraph breaks as double newline)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Remove null bytes
    text = text.replace("\x00", "")
    return text.strip()


def _company_row_id(company: dict[str, Any]) -> str:
    """
    Stable identifier for one GNEM workbook row.

    We prefer any explicit row/source id already present, then fall back to a
    deterministic hash of the row content so all semantic views of the same row
    can be merged back together after retrieval.
    """
    for key in ("source_row_id", "row_id"):
        if company.get(key):
            return str(company[key])
    if company.get("id") is not None:
        return f"company-id-{company['id']}"

    identity_fields = [
        company.get("company_name", ""),
        company.get("tier", ""),
        company.get("ev_supply_chain_role", ""),
        company.get("primary_oems", ""),
        company.get("location_city", ""),
        company.get("location_county", ""),
        company.get("employment", ""),
        company.get("products_services", ""),
        company.get("classification_method", ""),
        company.get("supplier_affiliation_type", ""),
    ]
    raw = "|".join(str(v or "").strip() for v in identity_fields)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"gnem-row-{digest}"


def _company_source_hash(company: dict[str, Any]) -> str:
    """Canonical hash of source fields that should trigger re-embedding if changed."""
    source_fields = {
        key: company.get(key)
        for key in (
            "company_name",
            "tier",
            "ev_supply_chain_role",
            "primary_oems",
            "ev_battery_relevant",
            "industry_group",
            "facility_type",
            "location_city",
            "location_county",
            "location_state",
            "employment",
            "products_services",
            "classification_method",
            "supplier_affiliation_type",
            "latitude",
            "longitude",
        )
    }
    raw = json.dumps(source_fields, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _company_chunk_id(row_id: str, view_name: str) -> str:
    """Deterministic UUID so repeated company indexing upserts instead of duplicating."""
    return str(uuid.uuid5(_COMPANY_CHUNK_NAMESPACE, f"{row_id}:{view_name}"))


def _parse_record_id(company: dict[str, Any]) -> str:
    """MD5 hash of stable identity fields, first 12 hex chars — matches schema example a1b2c3d4e5f6."""
    identity = "|".join([
        str(company.get("company_name", "") or "").strip(),
        str(company.get("location_county", "") or "").strip(),
        str(company.get("employment", "") or "").strip(),
        str(company.get("industry_group", "") or "").strip(),
    ])
    return hashlib.md5(identity.encode("utf-8")).hexdigest()[:12]


def _clean_company_name(name: str) -> str:
    """Strip * announcement markers: '*Rivian' → 'Rivian'."""
    return name.strip("* ").strip()


def _is_announcement(name: str) -> bool:
    """Company name with leading/trailing * = announced, not yet operational."""
    s = name.strip()
    return s.startswith("*") or s.endswith("*")


def _parse_tier(tier_str: str) -> tuple[str, str]:
    """
    Returns (tier_level, tier_confidence).
    'Tier 1 (likely)'   → ('1', 'likely')
    'Tier 2/3 (likely)' → ('2/3', 'likely')
    'OEM'               → ('OEM', '')
    'Tier 1'            → ('1', '')
    """
    s = tier_str.strip()
    confidence = ""
    m = re.search(r"\(([^)]+)\)", s)
    if m:
        confidence = m.group(1).strip().lower()
        s = s[:m.start()].strip()
    level = re.sub(r"(?i)^tier\s*", "", s).strip()
    return level, confidence


def _is_oem_ga(company: dict[str, Any]) -> bool:
    """True if this company is a Georgia OEM (tier == OEM or role mentions OEM)."""
    tier = str(company.get("tier", "") or "").strip().lower()
    role = str(company.get("ev_supply_chain_role", "") or "").strip().lower()
    return tier == "oem" or "oem" in role


def _parse_industry_group(industry_str: str) -> tuple[int | None, str, str]:
    """
    Returns (code, name, full_string).
    '37: Transportation Equipment' → (37, 'Transportation Equipment', '37: Transportation Equipment')
    'Textile Products'             → (None, 'Textile Products', 'Textile Products')
    """
    s = industry_str.strip()
    m = re.match(r"^(\d+)\s*[:\-]\s*(.+)$", s)
    if m:
        return int(m.group(1)), m.group(2).strip(), s
    m2 = re.match(r"^(\d+)$", s)
    if m2:
        return int(m2.group(1)), "", s
    return None, s, s


def _normalize_county(county_str: str) -> str:
    """'Gordon County' → 'Gordon', 'Troup County' → 'Troup', 'Troup' → 'Troup'."""
    return re.sub(r"\s+[Cc]ounty$", "", county_str.strip()).strip()


def _nonempty_parts(parts: list[str]) -> list[str]:
    return [part for part in parts if part and part.strip()]


def _location_text(company: dict[str, Any]) -> str:
    return " | ".join(_nonempty_parts([
        str(company.get("location_city") or "").strip(),
        str(company.get("location_county") or "").strip(),
        str(company.get("location_state") or "Georgia").strip(),
    ]))


def _company_context_text(company: dict[str, Any], master_text: str) -> str:
    """Canonical source-row text used for retrieval context and reranking."""
    return _clean_text(master_text)


def _company_view_texts(company: dict[str, Any], master_text: str) -> list[tuple[str, str]]:
    """
    Build multiple semantic views for the same company row.

    The master view keeps the full structured record for answer generation,
    while focused views give the retriever cleaner semantic targets.
    """
    company_name = str(company.get("company_name") or "").strip()
    location = _location_text(company)
    tier = str(company.get("tier") or "").strip()
    role = str(company.get("ev_supply_chain_role") or "").strip()
    primary_oems = str(company.get("primary_oems") or "").strip()
    ev_relevant = str(company.get("ev_battery_relevant") or "").strip()
    industry = str(company.get("industry_group") or "").strip()
    facility = str(company.get("facility_type") or "").strip()
    products = str(company.get("products_services") or "").strip()
    classification = str(company.get("classification_method") or "").strip()
    affiliation = str(company.get("supplier_affiliation_type") or "").strip()
    employment = str(company.get("employment") or "").strip()

    views: list[tuple[str, str]] = [("master", master_text)]

    role_parts = _nonempty_parts([
        f"Company: {company_name}",
        "Focus: EV supply chain role and classification",
        f"Tier: {tier}" if tier else "",
        f"EV Role: {role}" if role else "",
        f"EV / Battery Relevant: {ev_relevant}" if ev_relevant else "",
        f"Classification: {classification}" if classification else "",
        f"Supplier Affiliation: {affiliation}" if affiliation else "",
        f"Primary OEMs: {primary_oems}" if primary_oems else "",
    ])
    if len(role_parts) > 2:
        views.append(("role", " | ".join(role_parts)))

    product_parts = _nonempty_parts([
        f"Company: {company_name}",
        "Focus: products, services, and technology",
        f"Products: {products}" if products else "",
        f"Industry: {industry}" if industry else "",
        f"EV Role: {role}" if role else "",
        f"EV / Battery Relevant: {ev_relevant}" if ev_relevant else "",
        f"Tier: {tier}" if tier else "",
    ])
    if products:
        views.append(("product", " | ".join(product_parts)))

    oem_parts = _nonempty_parts([
        f"Company: {company_name}",
        "Focus: OEM relationships and customer network",
        f"Primary OEMs: {primary_oems}" if primary_oems else "",
        f"Tier: {tier}" if tier else "",
        f"EV Role: {role}" if role else "",
        f"Supplier Affiliation: {affiliation}" if affiliation else "",
        f"EV / Battery Relevant: {ev_relevant}" if ev_relevant else "",
    ])
    if primary_oems:
        views.append(("oem", " | ".join(oem_parts)))

    classification_parts = _nonempty_parts([
        f"Company: {company_name}",
        "Focus: tier, classification, affiliation, and relevance",
        f"Tier: {tier}" if tier else "",
        f"Classification: {classification}" if classification else "",
        f"Supplier Affiliation: {affiliation}" if affiliation else "",
        f"EV / Battery Relevant: {ev_relevant}" if ev_relevant else "",
        f"EV Role: {role}" if role else "",
        f"Facility Type: {facility}" if facility else "",
        f"Primary OEMs: {primary_oems}" if primary_oems else "",
    ])
    if len(classification_parts) > 2:
        views.append(("classification", " | ".join(classification_parts)))

    capability_parts = _nonempty_parts([
        f"Company: {company_name}",
        "Focus: manufacturing capability, product fit, and OEM suitability",
        f"Products: {products}" if products else "",
        f"EV Role: {role}" if role else "",
        f"Industry: {industry}" if industry else "",
        f"Facility Type: {facility}" if facility else "",
        f"Primary OEMs: {primary_oems}" if primary_oems else "",
        f"EV / Battery Relevant: {ev_relevant}" if ev_relevant else "",
        f"Tier: {tier}" if tier else "",
    ])
    if len(capability_parts) > 2:
        views.append(("capability", " | ".join(capability_parts)))

    location_parts = _nonempty_parts([
        f"Company: {company_name}",
        "Focus: location, facility, and workforce",
        f"Location: {location}" if location else "",
        f"Facility Type: {facility}" if facility else "",
        f"Employment: {employment}" if employment else "",
        f"Industry: {industry}" if industry else "",
        f"Tier: {tier}" if tier else "",
    ])
    if location or facility or employment:
        views.append(("location", " | ".join(location_parts)))

    return [(view_name, _clean_text(text)) for view_name, text in views if _clean_text(text)]


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentence-level units.
    Uses a simple but robust regex — avoids NLTK dependency.
    """
    # Split on sentence-ending punctuation followed by whitespace/newline
    parts = re.split(r"(?<=[.!?])\s+", text)
    # Also split on paragraph breaks
    result = []
    for part in parts:
        sub = part.strip()
        if sub:
            result.append(sub)
    return result


def _split_by_char_target(
    sentences: list[str],
    target_chars: int,
    overlap_chars: int = 0,
) -> list[str]:
    """
    Group sentences into chunks of approximately target_chars.
    Adds overlap_chars from the previous chunk at the start of each new chunk.
    """
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0
    overlap_buffer = ""  # Last part of previous chunk for overlap

    for sentence in sentences:
        sentence_len = len(sentence)

        # If a single sentence is larger than target, force-split it
        if sentence_len > target_chars:
            # Flush current buffer first
            if current_parts:
                chunks.append(" ".join(current_parts))
                overlap_buffer = current_parts[-1][:overlap_chars] if overlap_chars else ""
                current_parts = [overlap_buffer] if overlap_buffer else []
                current_len = len(overlap_buffer)

            # Hard-split the long sentence
            for i in range(0, sentence_len, target_chars - overlap_chars):
                segment = sentence[i: i + target_chars]
                if segment.strip():
                    chunks.append(segment.strip())
            overlap_buffer = sentence[-overlap_chars:] if overlap_chars else ""
            current_parts = [overlap_buffer] if overlap_buffer else []
            current_len = len(overlap_buffer)
            continue

        # Normal case: add sentence to current chunk
        if current_len + sentence_len > target_chars and current_parts:
            # Flush current chunk
            chunks.append(" ".join(current_parts))
            # Prepare overlap for next chunk
            overlap_buffer = " ".join(current_parts)[-overlap_chars:] if overlap_chars else ""
            current_parts = [overlap_buffer] if overlap_buffer else []
            current_len = len(overlap_buffer)

        current_parts.append(sentence)
        current_len += sentence_len + 1  # +1 for space

    # Flush final chunk
    if current_parts:
        text = " ".join(current_parts).strip()
        if text:
            chunks.append(text)

    return [c for c in chunks if c.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def chunk_company_record(
    company: dict[str, Any],
    document_text: str,
) -> list[Chunk]:
    """
    Create multi-view "company" chunks from a GNEM company record.

    GNEM records are structured and company-centric, so we keep one master view
    plus a handful of focused semantic views. Retrieval can match the focused
    chunk, while generation still gets the master record text.

    Args:
        company     : Company dict from gev_companies / GNEM Excel
        document_text : Pre-formatted company text (from kb_loader.build_document_text)

    Returns:
        List of company chunks (master + focused views)
    """
    master_text = _clean_text(document_text)
    row_id = _company_row_id(company)
    source_hash = _company_source_hash(company)
    company_context = _company_context_text(company, master_text)

    raw_name     = str(company.get("company_name", "") or "").strip()
    tier_raw     = str(company.get("tier", "") or "").strip()
    industry_raw = str(company.get("industry_group", "") or "").strip()
    county_raw   = str(company.get("location_county", "") or "").strip()
    products     = str(company.get("products_services", "") or "").strip()

    tier_level, tier_confidence = _parse_tier(tier_raw)
    industry_code, industry_name, industry_full = _parse_industry_group(industry_raw)

    metadata = {
        # ── Schema-aligned fields (exact names from data dictionary) ──────
        "Record_ID":               _parse_record_id(company),
        "Company":                 raw_name,
        "Company_Clean":           _clean_company_name(raw_name),
        "Employment":              int(company["employment"]) if company.get("employment") is not None else None,
        "Product_Service":         products,
        "County":                  _normalize_county(county_raw),
        "Tier_Category_heuristic": tier_raw,
        "Tier_Level":              tier_level,
        "Tier_Confidence":         tier_confidence,
        "OEM_GA":                  _is_oem_ga(company),
        "Industry_Group":          industry_full,
        "Industry_Code":           industry_code,
        "Industry_Name":           industry_name,
        "PDF_Page":                company.get("pdf_page"),
        "Is_Announcement":         _is_announcement(raw_name),
        # Chunk_ID and Embedding_Text are injected per-view in the loop below
        # ── Legacy/internal fields (kept for backward compatibility) ──────
        "company_name":            _clean_company_name(raw_name),
        "company_id":              company.get("id"),
        "company_row_id":          row_id,
        "tier":                    tier_raw,
        "ev_supply_chain_role":    company.get("ev_supply_chain_role", ""),
        "primary_oems":            company.get("primary_oems", ""),
        "ev_battery_relevant":     company.get("ev_battery_relevant", ""),
        "industry_group":          industry_raw,
        "facility_type":           company.get("facility_type", ""),
        "location_city":           str(company.get("location_city", "") or "").strip(),
        "location_county":         county_raw,
        "location_state":          company.get("location_state", "Georgia"),
        "latitude":                company.get("latitude"),
        "longitude":               company.get("longitude"),
        "employment":              company.get("employment"),
        "products_services":       products,
        "products_services_full":  products,
        "classification_method":   company.get("classification_method", ""),
        "supplier_affiliation_type": company.get("supplier_affiliation_type", ""),
        "source_type":             "gnem_excel",
        "chunk_type":              "company",
        "kb_schema_version":       COMPANY_CHUNK_SCHEMA_VERSION,
        "source_row_hash":         source_hash,
        "company_context_text":    company_context,
        "master_text":             master_text,
        "document_id":             None,
    }

    chunks: list[Chunk] = []
    for view_name, view_text in _company_view_texts(company, master_text):
        chunk_id = _company_chunk_id(row_id, view_name)
        chunk = Chunk(
            chunk_id=chunk_id,
            chunk_type="company",
            parent_id=None,
            text=view_text,
            token_estimate=_estimate_tokens(view_text),
            metadata={
                **metadata,
                "Chunk_ID":       chunk_id,
                "Embedding_Text": view_text,
                "chunk_view":     view_name,
                "text_preview":   view_text[:200],
            },
        )
        chunks.append(chunk)

    logger.debug(
        "Company multi-view chunks: %s -> %d views",
        company.get("company_name"),
        len(chunks),
    )
    return chunks


def chunk_document(
    text: str,
    company_name: str,
    document_id: int | None,
    source_url: str,
    document_type: str = "web",
    extra_metadata: dict[str, Any] | None = None,
    company_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """
    Split a web document into hierarchical parent-child chunks.

    Strategy:
      1. Split text into parent chunks (~800 tokens / ~3200 chars)
      2. Each parent chunk is split into child chunks (~256 tokens / ~1024 chars)
      3. Children store parent_id → enables "retrieve child, return parent" pattern

    Args:
        text            : Extracted document text
        company_name    : Associated company
        document_id     : gev_documents.id (for PostgreSQL tracking)
        source_url      : Original URL
        document_type   : "press_release", "sec_filing", "news", etc.
        extra_metadata  : Additional metadata to add to all chunks
        company_metadata: Company data dict (for enriching chunk payload)

    Returns:
        List of Chunks (mix of parent + child)
    """
    text = _clean_text(text)
    if not text:
        logger.warning("Empty text for %s — skipping chunking", company_name)
        return []

    extra_metadata = extra_metadata or {}
    company_metadata = company_metadata or {}

    # Base metadata shared across all chunks from this document
    base_meta = {
        "company_name": company_name,
        "company_id": company_metadata.get("id"),
        "document_id": document_id,
        "source_url": source_url,
        "document_type": document_type,
        "source_type": "web_document",
        # Company classification fields (for metadata filtering)
        "tier": company_metadata.get("tier", ""),
        "ev_supply_chain_role": company_metadata.get("ev_supply_chain_role", ""),
        "primary_oems": company_metadata.get("primary_oems", ""),
        "ev_battery_relevant": company_metadata.get("ev_battery_relevant", ""),
        "industry_group": company_metadata.get("industry_group", ""),
        "location_city": company_metadata.get("location_city", ""),
        "location_county": company_metadata.get("location_county", ""),
        "location_state": company_metadata.get("location_state", "Georgia"),
        "employment": company_metadata.get("employment"),
        **extra_metadata,
    }

    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    # Step 1: Build parent chunks
    parent_texts = _split_by_char_target(sentences, PARENT_CHAR_TARGET, overlap_chars=0)

    all_chunks: list[Chunk] = []

    for p_idx, parent_text in enumerate(parent_texts):
        # Create parent chunk
        parent_id = str(uuid.uuid4())
        parent_chunk = Chunk(
            chunk_id=parent_id,
            parent_id=None,
            chunk_type="parent",
            text=parent_text,
            token_estimate=_estimate_tokens(parent_text),
            metadata={
                **base_meta,
                "chunk_type": "parent",
                "chunk_index": p_idx,
                "total_parent_chunks": len(parent_texts),
                "text_preview": parent_text[:200],
            },
        )
        all_chunks.append(parent_chunk)

        # Step 2: Split parent into child chunks
        child_sentences = _split_into_sentences(parent_text)
        child_texts = _split_by_char_target(
            child_sentences,
            CHILD_CHAR_TARGET,
            overlap_chars=OVERLAP_CHARS,
        )

        for c_idx, child_text in enumerate(child_texts):
            child_chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                parent_id=parent_id,
                chunk_type="child",
                text=child_text,
                token_estimate=_estimate_tokens(child_text),
                metadata={
                    **base_meta,
                    "chunk_type": "child",
                    "parent_chunk_id": parent_id,
                    "chunk_index": c_idx,
                    "parent_chunk_index": p_idx,
                    "total_child_chunks": len(child_texts),
                    "text_preview": child_text[:200],
                },
            )
            all_chunks.append(child_chunk)

    parents = sum(1 for c in all_chunks if c.chunk_type == "parent")
    children = sum(1 for c in all_chunks if c.chunk_type == "child")
    logger.debug(
        "Chunked [%s] %s → %d parents, %d children (%.0f tokens avg child)",
        company_name,
        source_url[-60:],
        parents,
        children,
        sum(c.token_estimate for c in all_chunks if c.chunk_type == "child") / max(children, 1),
    )

    return all_chunks


def get_child_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Return only child chunks (used for vector search)."""
    return [c for c in chunks if c.chunk_type in ("child", "company")]


def get_parent_chunk(chunks: list[Chunk], child: Chunk) -> Chunk | None:
    """Given a child chunk, find its parent chunk from the same list."""
    if child.parent_id is None:
        return None
    return next((c for c in chunks if c.chunk_id == child.parent_id), None)
