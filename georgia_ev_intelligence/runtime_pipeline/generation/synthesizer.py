"""
LLM synthesis: one call per question using evidence rows as context.
Supports Anthropic API if configured, otherwise Ollama fallback.

Important:
  - This file should NOT do retrieval or aggregation.
  - It should only format and explain evidence already selected upstream.
  - For exhaustive questions, the prompt strongly instructs the model to include
    every evidence item and avoid summarizing away rows.
"""
from __future__ import annotations

import re
import requests

from ...shared import config


_SYSTEM = (
    "You are a Georgia EV supply chain analyst. "
    "Answer the question using ONLY the KB evidence provided. "
    "Do not invent or add information beyond what the evidence contains. "
    "Be concise and factual. Use bullet lists or tables when listing multiple items."
)

_SYSTEM_EXHAUSTIVE = (
    "You are a Georgia EV supply chain analyst. "
    "Answer the question using ONLY the KB evidence provided. "
    "Do not invent or add information beyond what the evidence contains. "
    "IMPORTANT: This question requests a COMPLETE list, count, total, ranking, or exhaustive result. "
    "Include ALL items from the evidence. Do not summarize, abbreviate, skip, merge, or omit entries. "
    "If the evidence contains numbered records, preserve every record in the answer. "
    "If the evidence contains a count/total/ranking result, report that result exactly. "
    "Use bullet lists or tables when listing multiple items."
)


def _cfg(name: str, default):
    """Safe optional config getter so older config.py files do not break."""
    return getattr(config, name, default)


def _build_prompt(question: str, evidence: list[str], exhaustive: bool = False) -> str:
    kb_block = (
        "\n".join(f"[{i + 1}] {e}" for i, e in enumerate(evidence))
        if evidence
        else "No matching KB records found."
    )

    if exhaustive:
        task_instruction = (
            "Answer using the evidence above.\n"
            "Because this is an exhaustive/analytical question, include every relevant evidence item. "
            "Do not summarize the list into examples. Do not say 'including' unless you also provide all listed items. "
            "If the evidence has one aggregate row, report the aggregate directly."
        )
    else:
        task_instruction = (
            "Answer using only the evidence above. Be concise, but do not omit key evidence."
        )

    return (
        f"KB Evidence:\n{kb_block}\n\n"
        f"Question: {question}\n\n"
        f"{task_instruction}\n\n"
        "Answer:"
    )


def _call_anthropic(prompt: str, system: str = _SYSTEM, exhaustive: bool = False) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    max_tokens = int(
        _cfg(
            "ANTHROPIC_EXHAUSTIVE_MAX_TOKENS" if exhaustive else "ANTHROPIC_MAX_TOKENS",
            4096 if exhaustive else 2048,
        )
    )

    msg = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )

    return msg.content[0].text.strip()


def _call_ollama(prompt: str, system: str = _SYSTEM, exhaustive: bool = False) -> str:
    """
    Ollama generation with explicit options.

    Without num_predict, local models may truncate long exhaustive lists depending
    on model defaults. This function sets deterministic/low-temperature behavior
    and gives exhaustive answers more output budget.
    """
    full_prompt = f"{system}\n\n{prompt}"

    options = {
        "temperature": float(_cfg("OLLAMA_SYNTH_TEMPERATURE", 0.1)),
        "top_p": float(_cfg("OLLAMA_SYNTH_TOP_P", 0.9)),
        "num_predict": int(
            _cfg(
                "OLLAMA_SYNTH_EXHAUSTIVE_NUM_PREDICT" if exhaustive else "OLLAMA_SYNTH_NUM_PREDICT",
                4096 if exhaustive else 2048,
            )
        ),
    }

    # Optional context window. If not defined, do not force it.
    num_ctx = _cfg("OLLAMA_SYNTH_NUM_CTX", None)
    if num_ctx is not None:
        options["num_ctx"] = int(num_ctx)

    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": config.OLLAMA_LLM_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": options,
        },
        timeout=int(_cfg("OLLAMA_SYNTH_TIMEOUT", 180)),
    )

    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def synthesize(question: str, evidence: list[str], exhaustive: bool = False) -> tuple[str, str]:
    """
    Returns:
      (answer, hallucination_risk)

    hallucination_risk:
      'Low' | 'Medium' | 'High'
    """
    evidence = [str(e).strip() for e in evidence if str(e).strip()]

    prompt = _build_prompt(question, evidence, exhaustive=exhaustive)
    system = _SYSTEM_EXHAUSTIVE if exhaustive else _SYSTEM

    if config.USE_ANTHROPIC:
        answer = _call_anthropic(prompt, system=system, exhaustive=exhaustive)
    else:
        answer = _call_ollama(prompt, system=system, exhaustive=exhaustive)

    answer = _clean_answer(answer)
    risk = _assess_risk(answer, evidence)

    return answer, risk


def _clean_answer(answer: str) -> str:
    """
    Remove common local-model artifacts without changing factual content.
    """
    if not answer:
        return ""

    cleaned = answer.strip()

    # Remove DeepSeek/Ollama-style thinking traces if they appear.
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()

    # Remove accidental markdown code fences around normal prose.
    cleaned = re.sub(r"^```(?:text|markdown)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    return cleaned


def _assess_risk(answer: str, evidence: list[str]) -> str:
    """
    Lightweight heuristic hallucination-risk estimate.

    Note:
    This is not a factual verifier. It only estimates how many answer words are
    unsupported by the provided evidence text. Domain/general words are ignored.
    """
    if not evidence:
        return "High"

    evidence_text = " ".join(evidence).lower()

    stop_words = {
        "about", "above", "according", "answer", "based", "because", "below",
        "company", "companies", "complete", "contains", "county", "data",
        "does", "each", "evidence", "following", "from", "georgia", "include",
        "includes", "item", "items", "listed", "provided", "record", "records",
        "result", "results", "shows", "supplier", "suppliers", "table", "there",
        "these", "this", "total", "using", "with", "would",
    }

    answer_words = [
        w.strip(".,;?!\"'()[]{}:").lower()
        for w in answer.split()
        if len(w.strip(".,;?!\"'()[]{}:")) > 4
    ]

    answer_words = [
        w for w in answer_words
        if w and w not in stop_words and not w.isdigit()
    ]

    if not answer_words:
        return "Low"

    unsupported = [w for w in answer_words if w not in evidence_text]
    ratio = len(unsupported) / max(len(answer_words), 1)

    if ratio < 0.20:
        return "Low"
    if ratio < 0.45:
        return "Medium"
    return "High"
