"""
Phase 2 — Hierarchical Parent-Child Chunker

WHY THIS APPROACH:
  - Child chunks (256 tokens) = precise semantic search targets
  - Parent chunks (800 tokens) = full context returned to LLM
  - Pattern: search with small, answer with large
  - Reference: https://python.langchain.com/docs/how_to/parent_document_retriever/
  - Research: 256-512 token sweet spot for retrieval accuracy (2025 consensus)

TWO DOCUMENT TYPES:
  1. GNEM Company records → 1 chunk per company (structured text, ~100 tokens)
     These are pre-formatted: "Company: X | Tier: Y | Location: Z | ..."
  2. Web documents → hierarchical parent-child split
     Parent = 800 tokens (sent to LLM as context)
     Child  = 256 tokens (used for vector search)
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from shared.logger import get_logger

logger = get_logger("phase2.chunker")

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
    Create a single "company" chunk from a GNEM company record.

    GNEM records are already structured (Company: X | Tier: Y | ...)
    They are short (~100 tokens) — no splitting needed.
    One chunk per company, stored with rich metadata.

    Args:
        company     : Company dict from gev_companies / GNEM Excel
        document_text : Pre-formatted company text (from kb_loader.build_document_text)

    Returns:
        List with exactly 1 Chunk (type="company")
    """
    text = _clean_text(document_text)

    metadata = {
        # Core identity
        "company_name": company.get("company_name", ""),
        "company_id": company.get("id"),
        # Supply chain classification
        "tier": company.get("tier", ""),
        "ev_supply_chain_role": company.get("ev_supply_chain_role", ""),
        "primary_oems": company.get("primary_oems", ""),
        "ev_battery_relevant": company.get("ev_battery_relevant", ""),
        "industry_group": company.get("industry_group", ""),
        "facility_type": company.get("facility_type", ""),
        # Location (critical for geo queries)
        "location_city": company.get("location_city", ""),
        "location_county": company.get("location_county", ""),
        "location_state": company.get("location_state", "Georgia"),
        "latitude": company.get("latitude"),
        "longitude": company.get("longitude"),
        # Size
        "employment": company.get("employment"),
        # Products / classification
        "products_services": (company.get("products_services") or "")[:300],
        "classification_method": company.get("classification_method", ""),
        "supplier_affiliation_type": company.get("supplier_affiliation_type", ""),
        # Chunk metadata
        "source_type": "gnem_excel",
        "chunk_type": "company",
        "document_id": None,
    }

    chunk = Chunk(
        chunk_type="company",
        parent_id=None,
        text=text,
        token_estimate=_estimate_tokens(text),
        metadata=metadata,
    )
    logger.debug("Company chunk: %s (%d tokens)", company.get("company_name"), chunk.token_estimate)
    return [chunk]


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
