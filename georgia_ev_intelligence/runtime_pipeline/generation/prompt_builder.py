"""Build the final LLM prompt for grounded answer generation."""
from __future__ import annotations


_SYSTEM_INSTRUCTION = (
    "You are a Georgia EV supply chain knowledge base assistant. "
    "You answer questions using ONLY the provided KB context records below. "
    "Follow these rules strictly:\n"
    "- Cite every factual claim using the source IDs provided (e.g., [S1], [S2]).\n"
    "- Do not invent companies, products, locations, values, or any information "
    "not present in the provided context.\n"
    "- Preserve exact field values from the KB records. Do not paraphrase names, "
    "categories, or roles.\n"
    "- If the answer is not found in the provided context, say: "
    "\"This information is not found in the provided KB context.\"\n"
    "- For list questions, include ALL matching records present in the context.\n"
    "- For count questions, count only records present in the provided context "
    "and state the count explicitly.\n"
    "- Be concise and factual. Use bullet points or tables when listing multiple items.\n"
)


def build_prompt(question: str, context: str) -> str:
    """Build the full prompt combining system instruction, context, and question.

    The prompt instructs the model to use only provided context and cite sources.
    Does not use chain-of-thought prompting.
    """
    return (
        f"{_SYSTEM_INSTRUCTION}\n"
        f"--- KB Context ---\n"
        f"{context}\n"
        f"--- End KB Context ---\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
