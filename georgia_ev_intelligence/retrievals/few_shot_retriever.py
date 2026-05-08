"""
phase5_fewshot/few_shot_retriever.py
─────────────────────────────────────────────────────────────────────────────
High-level API: embed a new question, find similar verified examples,
format them as a few-shot prompt block.

Used by:
  - text_to_sql.py  → inject SQL few-shot examples
  - text_to_cypher.py → inject Cypher few-shot examples

USAGE:
    from retrievals.few_shot_retriever import get_few_shot_block

    # In text_to_sql.py prompt:
    few_shot = get_few_shot_block(question, query_type="sql", top_k=3)
    prompt = FEW_SHOT_SQL_PROMPT.format(few_shot_examples=few_shot, question=question, schema=schema)
"""
from __future__ import annotations

from shared.logger import get_logger
from embeddings_store.few_shot_embedder import embed_text
from embeddings_store.qdrant_store import search_similar, count_examples

logger = get_logger("phase5.retriever")

# Similarity threshold — only inject examples with cosine score >= threshold
# Below this: the examples are probably not helpful and may confuse the LLM
_MIN_SCORE = 0.70


def get_few_shot_examples(
    question: str,
    query_type: str = "sql",   # "sql" | "cypher"
    top_k: int = 3,
) -> list[dict]:
    """
    Find the top-k most similar verified examples for this question.
    Returns only examples above the similarity threshold.

    Args:
        question:   The current user question
        query_type: "sql" or "cypher" — filters the store by query type
        top_k:      Maximum examples to return

    Returns:
        List of example dicts with keys: question, sql/cypher, answer, score
    """
    if count_examples() == 0:
        logger.debug("Few-shot store is empty — skipping retrieval")
        return []

    vector = embed_text(question)
    hits   = search_similar(vector, top_k=top_k, query_type_filter=query_type)

    # Filter by minimum similarity score
    filtered = [h for h in hits if h.get("score", 0) >= _MIN_SCORE]
    logger.info(
        "Few-shot: %d/%d examples above threshold (query_type=%s)",
        len(filtered), len(hits), query_type,
    )
    return filtered


def get_few_shot_block(
    question: str,
    query_type: str = "sql",
    top_k: int = 3,
) -> str:
    """
    Return a formatted string block of few-shot examples for injection into prompts.
    Returns empty string if no examples found (prompt falls back to zero-shot).

    SQL format:
        Example 1:
        Question: Which county has the highest Tier 1 employment?
        SQL: SELECT location_county, SUM(employment) FROM gev_companies WHERE tier='Tier 1' GROUP BY location_county ORDER BY SUM(employment) DESC LIMIT 1
        Answer: Troup County

    Cypher format:
        Example 1:
        Question: Which companies supply copper foil in Georgia?
        Cypher: MATCH (c:Company) WHERE toLower(c.products_services) CONTAINS 'copper foil' RETURN c.name, c.tier, c.location_county
        Answer: Duckyang (Tier 2/3, Jackson County)
    """
    examples = get_few_shot_examples(question, query_type=query_type, top_k=top_k)
    if not examples:
        return ""

    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i} (similarity={ex.get('score', 0):.2f}):")
        lines.append(f"Question: {ex['question']}")
        if query_type == "sql" and ex.get("sql"):
            lines.append(f"SQL: {ex['sql']}")
        elif query_type == "cypher" and ex.get("cypher"):
            lines.append(f"Cypher: {ex['cypher']}")
        lines.append(f"Answer: {ex['answer'][:200]}")
        lines.append("")

    block = "\n".join(lines)
    logger.debug("Few-shot block: %d chars, %d examples", len(block), len(examples))
    return block
