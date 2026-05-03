"""
Shared database models and connection.
Extends the existing Kb_Enrichment ORM with new tables for Phase 1/2/3.
Uses SQLAlchemy 2.x with PostgreSQL (Supabase).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from shared.config import Config
from shared.logger import get_logger

logger = get_logger("shared.db")


class Base(DeclarativeBase):
    pass


# ── Table: companies ─────────────────────────────────────────────────────────
class Company(Base):
    """
    One row per company from GNEM Excel.
    205 rows total. Never grows — source of truth is the Excel.
    """
    __tablename__ = "gev_companies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_name = Column(String(500), nullable=False, unique=True, index=True)
    tier = Column(String(50))                    # OEM, Tier 1, Tier 1/2, Tier 2  (max=16)
    ev_supply_chain_role = Column(String(200))   # EV role description (max=57)
    primary_oems = Column(String(200))           # OEM names (max=32)
    ev_battery_relevant = Column(String(100))    # Yes / No / Indirect / Public OEM... (max=39)
    industry_group = Column(String(200))         # Industry classification (max=57)
    facility_type = Column(String(200))          # Manufacturing Plant / R&D / Assembly etc.
    location_city = Column(String(200))
    location_county = Column(String(200))
    location_state = Column(String(100), default="Georgia")
    employment = Column(Float)                   # Employee count
    products_services = Column(Text)             # Products/services (max=176, use Text)
    classification_method = Column(String(100))  # (max=39)
    supplier_affiliation_type = Column(String(200))  # Affiliation type (max=35)
    latitude = Column(Float)
    longitude = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    documents = relationship("Document", back_populates="company", lazy="dynamic")
    facts = relationship("ExtractedFact", back_populates="company", lazy="dynamic")


# ── Table: documents ─────────────────────────────────────────────────────────
class Document(Base):
    """
    Every downloaded document tracked here.
    Maps to a Backblaze B2 object key.
    """
    __tablename__ = "gev_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(Integer, ForeignKey("gev_companies.id"), nullable=True, index=True)
    company_name = Column(String(500), index=True)   # Denormalized for speed
    source_url = Column(String(2000), nullable=False)
    b2_key = Column(String(500))                     # Backblaze B2 object key
    content_type = Column(String(100))               # application/pdf, text/html, etc.
    content_hash_sha256 = Column(String(64), index=True)  # Deduplication
    file_size_bytes = Column(BigInteger)
    document_type = Column(String(100))              # Press Release, SEC Filing, Gov Report, etc.
    relevance_score = Column(Float)                  # 0.0-1.0 from searcher
    extraction_status = Column(String(50), default="pending")  # pending, extracted, failed
    extraction_error = Column(Text)
    word_count = Column(Integer)
    search_query = Column(String(500))               # Query that found this doc
    downloaded_at = Column(DateTime, default=datetime.utcnow)
    extracted_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    company = relationship("Company", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", lazy="dynamic")
    facts = relationship("ExtractedFact", back_populates="document", lazy="dynamic")


# ── Table: document_chunks ───────────────────────────────────────────────────
class DocumentChunk(Base):
    """
    Parent-child chunks stored here for tracking.
    Actual vectors live in Qdrant — this table tracks IDs and metadata.
    """
    __tablename__ = "gev_document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("gev_documents.id"), nullable=False, index=True)
    qdrant_id = Column(String(100), unique=True, index=True)  # UUID used in Qdrant
    chunk_type = Column(String(20), nullable=False)           # "parent" or "child"
    parent_qdrant_id = Column(String(100), index=True)        # NULL for parent chunks
    chunk_index = Column(Integer)                             # Position within document
    token_count = Column(Integer)
    char_count = Column(Integer)
    text_preview = Column(String(500))                        # First 500 chars for debugging
    embedded_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks")


# ── Table: extracted_facts ───────────────────────────────────────────────────
class ExtractedFact(Base):
    """
    Key structured facts extracted from documents.
    Your idea: store precise numbers in SQL so LLM never has to guess them.
    Queryable in <200ms vs. embedding search.
    """
    __tablename__ = "gev_extracted_facts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("gev_documents.id"), nullable=True, index=True)
    company_id = Column(Integer, ForeignKey("gev_companies.id"), nullable=True, index=True)
    company_name = Column(String(500), index=True)   # Denormalized

    # Fact fields
    fact_type = Column(String(100), index=True)      # investment, jobs, facility, expansion, etc.
    fact_value_text = Column(Text)                   # Raw text value
    fact_value_numeric = Column(Float)               # Parsed number (investment $, job count)
    fact_currency = Column(String(10))               # USD, etc.
    fact_unit = Column(String(50))                   # jobs, sqft, MW, etc.
    fact_year = Column(Integer, index=True)
    fact_quarter = Column(String(10))                # Q1, Q2, Q3, Q4
    location_city = Column(String(200))
    location_county = Column(String(200))
    location_state = Column(String(100), default="Georgia")
    oem_partner = Column(String(500))
    confidence_score = Column(Float)                 # 0.0-1.0 extraction confidence
    source_sentence = Column(Text)                   # The exact sentence it came from
    extracted_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="facts")
    company = relationship("Company", back_populates="facts")


# ── Table: eval_results ──────────────────────────────────────────────────────
class EvalResult(Base):
    """
    RAGAS evaluation results for the 50 golden questions.
    Same schema as ev_data_LLM_comparsions reporter output.
    """
    __tablename__ = "gev_eval_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(100), index=True)         # Timestamp-based run identifier
    question_id = Column(Integer, index=True)
    category = Column(String(200))
    question = Column(Text)
    golden_answer = Column(Text)
    generated_answer = Column(Text)
    retrieved_context = Column(Text)
    faithfulness = Column(Float)
    answer_relevancy = Column(Float)
    context_precision = Column(Float)
    context_recall = Column(Float)
    answer_correctness = Column(Float)
    final_score = Column(Float)
    faithfulness_reason = Column(Text)
    answer_relevancy_reason = Column(Text)
    retrieval_source = Column(String(50))            # "qdrant", "neo4j", "tavily", "hybrid"
    response_time_ms = Column(Integer)
    evaluated_at = Column(DateTime, default=datetime.utcnow)


# ── Engine & Session Factory ─────────────────────────────────────────────────
_engine = None
_SessionFactory = None


def get_engine():
    global _engine
    if _engine is None:
        cfg = Config.get()
        db_url = cfg.database_url

        # Neon.tech pooler rejects channel_binding in startup options.
        # Strip it completely — sslmode=require still enforces TLS.
        # Ref: https://neon.tech/docs/connect/connection-errors#unsupported-startup-parameter
        if "channel_binding" in db_url:
            db_url = (
                db_url
                .replace("&channel_binding=require", "")
                .replace("&channel_binding=disable", "")
                .replace("?channel_binding=require", "?")
                .replace("?channel_binding=disable", "?")
                .rstrip("?&")
            )
            logger.debug("Neon DB: stripped unsupported channel_binding parameter")

        _engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            echo=False,
        )
        provider = "Neon.tech" if "neon.tech" in db_url else "Supabase/PostgreSQL"
        logger.info("Database engine created (%s)", provider)
    return _engine


def get_session_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory


def get_session() -> Session:
    """Get a new database session."""
    return get_session_factory()()


def create_tables() -> None:
    """Create all tables that don't exist yet. Safe to run multiple times."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("All gev_* tables created/verified in PostgreSQL")


def verify_connection() -> bool:
    """Test that the DB is reachable. Returns True if OK."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL connection verified")
        return True
    except Exception as exc:
        logger.error("PostgreSQL connection FAILED: %s", exc)
        return False
