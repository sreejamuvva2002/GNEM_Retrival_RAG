"""
LLM synthesis: one call per question using evidence rows as context.
Supports Anthropic API (if ANTHROPIC_API_KEY set) or Ollama (fallback).
"""
from __future__ import annotations
import json
import requests
from . import config


_SYSTEM = (
    "You are a Georgia EV supply chain analyst. "
    "Answer the question using ONLY the KB evidence provided. "
    "Do not invent or add information beyond what the evidence contains. "
    "Be concise and factual. Use bullet lists or tables when listing multiple items."
)


def _build_prompt(question: str, evidence: list[str]) -> str:
    kb_block = "\n".join(f"[{i+1}] {e}" for i, e in enumerate(evidence)) if evidence else "No matching KB records found."
    return (
        f"KB Evidence:\n{kb_block}\n\n"
        f"Question: {question}\n\n"
        "Answer (based strictly on the evidence above):"
    )


def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _call_ollama(prompt: str) -> str:
    full_prompt = f"{_SYSTEM}\n\n{prompt}"
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={"model": config.OLLAMA_LLM_MODEL, "prompt": full_prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def synthesize(question: str, evidence: list[str]) -> tuple[str, str]:
    """
    Returns (answer, hallucination_risk).
    hallucination_risk: 'Low' | 'Medium' | 'High'
    """
    prompt = _build_prompt(question, evidence)

    if config.USE_ANTHROPIC:
        answer = _call_anthropic(prompt)
    else:
        answer = _call_ollama(prompt)

    risk = _assess_risk(answer, evidence)
    return answer, risk


def _assess_risk(answer: str, evidence: list[str]) -> str:
    if not evidence:
        return "High"
    evidence_text = " ".join(evidence).lower()
    answer_words = [w.strip(".,;?!\"'()").lower() for w in answer.split() if len(w) > 4]
    unsupported = [w for w in answer_words if w not in evidence_text]
    ratio = len(unsupported) / max(len(answer_words), 1)
    if ratio < 0.15:
        return "Low"
    if ratio < 0.40:
        return "Medium"
    return "High"

