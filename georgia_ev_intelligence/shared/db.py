"""
Shared database models and connection.
Extends the existing Kb_Enrichment ORM with new tables for Phase 1/2/3.
Uses SQLAlchemy 2.x with PostgreSQL (Supabase).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    ARRAY,
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
from sqlalchemy.dialects.postgresql import JSONB, UUID
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


# ── Table: retrieval_audit ───────────────────────────────────────────────────
class RetrievalAudit(Base):
    """
    One row per question answered by the new retrieval pipeline.

    Captures the full provenance: classified intent, extracted entities,
    ambiguous terms and their branches, retrievers invoked, raw SQL/Cypher/
    Qdrant queries, final selected evidence, the produced answer, and a
    hallucination-risk score from the answer verifier.

    Note on schema: JSONB columns hold structured data so it stays queryable
    (`SELECT ... WHERE extracted_entities->>'tier' = 'Tier 1'`).
    """
    __tablename__ = "gev_retrieval_audit"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(String(100), index=True)
    question = Column(Text, nullable=False)
    query_class = Column(String(64), index=True)

    extracted_entities = Column(JSONB)
    hard_filters = Column(JSONB)
    ambiguous_terms = Column(JSONB)
    selected_interpretations = Column(JSONB)
    synonym_mappings = Column(JSONB)

    retrieval_methods_used = Column(ARRAY(String(32)))
    sql_query = Column(Text)
    cypher_query = Column(Text)
    qdrant_dense_query = Column(Text)
    qdrant_sparse_query = Column(Text)

    final_evidence = Column(JSONB)
    answer_text = Column(Text)
    support_level = Column(String(20))
    hallucination_risk = Column(Integer, default=0)
    audit_comment = Column(Text)
    elapsed_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    candidates = relationship("RetrievalCandidate", back_populates="audit", lazy="dynamic")


# ── Table: retrieval_candidates ──────────────────────────────────────────────
class RetrievalCandidate(Base):
    """
    One row per candidate that was considered for a question. Includes both
    selected and rejected candidates so post-hoc analysis can answer
    'why did Foo Inc. not appear in the answer?'.

    `scores` keeps per-source raw scores (dense, sparse, reranker, ...).
    `fused_score` is the final weighted score from retrieval_fusion.
    `rejection_reason` is set whenever hard_filter_passed is False.
    """
    __tablename__ = "gev_retrieval_candidates"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    audit_id = Column(BigInteger, ForeignKey("gev_retrieval_audit.id"), nullable=False, index=True)
    branch_id = Column(String(8))
    company_row_id = Column(Integer, ForeignKey("gev_companies.id"), nullable=True, index=True)
    canonical_name = Column(String(500), index=True)
    sources = Column(ARRAY(String(16)))
    scores = Column(JSONB)
    fused_score = Column(Float)
    hard_filter_passed = Column(Boolean, default=True)
    rejection_reason = Column(Text)
    final_selected = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    audit = relationship("RetrievalAudit", back_populates="candidates")


# ── Table: domain_mapping_rules ──────────────────────────────────────────────
#
# WRITE POLICY — DO NOT BREAK
#
# Rows in gev_domain_mapping_rules are the ONLY sanctioned home for
# domain-specific natural-language → KB-filter mappings (e.g.
# "small scale" → employment < 200, "sole-sourced" → exactly one primary OEM).
#
# Allowed writers:
#   - scripts/approve_mapping_rule.py (admin CLI, requires human confirmation)
#   - manual SQL run by an operator
#
# FORBIDDEN writers:
#   - any module under phase4_agent/* (no INSERT / UPDATE / DELETE)
#   - any LLM-driven path (filter_interpreter, synonym_expander, pipeline)
#
# Reasoning: a mapping that has not been reviewed by a human is a model
# guess — persisting it would let the system silently overfit to one
# question's wording. Audit logs may *record* a guess as
# `support_basis="llm_suggestion"`; only an explicit approval moves it
# here. See the refactor plan §E ("No feedback = no permanent learning").
#
# Read policy: phase4_agent/synonym_expander.py is the only module that
# reads this table at retrieval time. It filters
# `WHERE status='approved'` (or 'active' for legacy rows). Anything not
# approved is treated as if absent.
class DomainMappingRule(Base):
    """
    Human-approved conditional mappings from natural-language abstract terms
    to concrete KB filters.

    CRITICAL: rows are NEVER inserted automatically by the agent. The audit
    pipeline may capture LLM-suggested mappings in the audit log, but only
    explicit human approval (via a separate admin tool) writes a row here.
    See plan §E ("No feedback = no permanent learning").
    """
    __tablename__ = "gev_domain_mapping_rules"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    term = Column(String(200), nullable=False, index=True)
    mapped_column = Column(String(100), nullable=False)
    mapped_value_or_condition = Column(String(500), nullable=False)
    valid_when = Column(Text)
    invalid_when = Column(Text)
    source = Column(String(50), default="human_approved")
    status = Column(String(20), default="active", index=True)
    confidence = Column(Float, default=1.0)
    example_question = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


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


def create_audit_tables() -> None:
    """
    Create only the new retrieval-audit tables (gev_retrieval_audit,
    gev_retrieval_candidates, gev_domain_mapping_rules). Safe to run
    multiple times. Used by scripts/create_audit_tables.py so the audit
    schema can be applied without touching the existing gev_companies
    or gev_documents tables.
    """
    engine = get_engine()
    target_tables = [
        Base.metadata.tables[name]
        for name in (
            "gev_retrieval_audit",
            "gev_retrieval_candidates",
            "gev_domain_mapping_rules",
        )
    ]
    Base.metadata.create_all(engine, tables=target_tables)
    logger.info("Retrieval audit tables created/verified")


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
