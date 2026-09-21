"""Layer 4 - reference-free RAG metrics (RAGAS-style, implemented on Gemini).

These need no ground truth (except context_recall), which is what makes them
usable in production on questions nobody has hand-answered:

  faithfulness      - split the answer into atomic claims, ask whether each is
                      entailed by the retrieved context. Catches the failure the
                      business cares about: confident claims the data doesn't make.
  answer_relevancy  - have the model write the questions the answer would be a
                      good answer to, then embed them and compare to the real
                      question. Catches evasion and padding.
  context_precision - of the retrieved records, how many were actually useful,
                      weighted by rank. Low value = the retriever is noisy.
  context_recall    - which sentences of the golden answer are supported by the
                      retrieved context. Low value = the retriever missed data
                      that exists. (This one does need the golden answer.)

Implemented directly against Gemini rather than via the ragas package so the
prompts are inspectable, the calls are cached, and there is one model provider
in the stack.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import Settings, settings as default_settings
from ..gemini_client import GeminiClient

CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
    "required": ["claims"],
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["claim", "supported", "reason"],
            },
        }
    },
    "required": ["verdicts"],
}

USEFULNESS_SCHEMA = {
    "type": "object",
    "properties": {"useful": {"type": "array", "items": {"type": "boolean"}}},
    "required": ["useful"],
}

QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
}


def _truncate(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def faithfulness(*, answer: str, contexts: list[str], client: GeminiClient, cfg: Settings | None = None) -> dict[str, Any]:
    """Share of the answer's atomic claims that the context entails."""
    cfg = cfg or default_settings
    if not answer.strip() or not contexts:
        return {"faithfulness": float("nan"), "n_claims": 0, "unsupported_claims": []}

    claims_payload = client.generate_json(
        "Break this answer into atomic factual claims: one verifiable statement each, "
        "self-contained (resolve pronouns), no opinions, no restatements of the question.\n\n"
        f"ANSWER\n{_truncate(answer, 8000)}",
        schema=CLAIMS_SCHEMA,
        model=cfg.judge_model,
        temperature=0.0,
    )
    claims = [c for c in claims_payload.get("claims", []) if str(c).strip()][:25]
    if not claims:
        return {"faithfulness": float("nan"), "n_claims": 0, "unsupported_claims": []}

    numbered = "\n".join(f"{i}. {claim}" for i, claim in enumerate(claims, start=1))
    verdict_payload = client.generate_json(
        "For each claim, decide whether the CONTEXT directly supports it. "
        "Supported means the context states it or it follows by simple arithmetic or "
        "counting over the context. Plausible-but-absent is NOT supported.\n\n"
        f"CONTEXT\n{_truncate(chr(10).join(contexts), 20000)}\n\nCLAIMS\n{numbered}",
        schema=VERDICT_SCHEMA,
        model=cfg.judge_model,
        temperature=0.0,
    )
    verdicts = verdict_payload.get("verdicts", [])
    if not verdicts:
        return {"faithfulness": float("nan"), "n_claims": len(claims), "unsupported_claims": []}
    supported = [v for v in verdicts if v.get("supported")]
    unsupported = [str(v.get("claim", "")) for v in verdicts if not v.get("supported")]
    return {
        "faithfulness": len(supported) / len(verdicts),
        "n_claims": len(verdicts),
        "unsupported_claims": unsupported,
    }


def answer_relevancy(*, question: str, answer: str, client: GeminiClient, n: int = 3, cfg: Settings | None = None) -> float:
    """Cosine similarity between the real question and questions reverse-generated
    from the answer. An evasive or padded answer generates different questions."""
    cfg = cfg or default_settings
    if not answer.strip():
        return float("nan")
    payload = client.generate_json(
        f"Write {n} different questions that this answer would be a direct and complete "
        "answer to. Use only information present in the answer.\n\n"
        f"ANSWER\n{_truncate(answer, 6000)}",
        schema=QUESTIONS_SCHEMA,
        model=cfg.judge_model,
        temperature=0.0,
    )
    generated = [q for q in payload.get("questions", []) if str(q).strip()][:n]
    if not generated:
        return float("nan")
    vectors = client.embed([question] + generated, task_type="SEMANTIC_SIMILARITY")
    target, rest = vectors[0], vectors[1:]
    return float(np.mean(rest @ target))


def context_precision(*, question: str, reference: str, contexts: list[str], client: GeminiClient, cfg: Settings | None = None) -> float:
    """Rank-weighted average precision over 'was this record useful?' labels."""
    cfg = cfg or default_settings
    if not contexts:
        return float("nan")
    blocks = contexts[:20]
    listing = "\n\n".join(f"RECORD {i}:\n{_truncate(block, 1500)}" for i, block in enumerate(blocks, start=1))
    payload = client.generate_json(
        "For each record, answer true if it contains information useful for answering the "
        "question (or for producing the reference answer), false otherwise. Return one "
        "boolean per record, in order.\n\n"
        f"QUESTION\n{question}\n\nREFERENCE ANSWER\n{_truncate(reference, 4000)}\n\n{listing}",
        schema=USEFULNESS_SCHEMA,
        model=cfg.judge_model,
        temperature=0.0,
    )
    flags = payload.get("useful", [])[: len(blocks)]
    if not flags:
        return float("nan")
    hits, weighted = 0, 0.0
    for rank, flag in enumerate(flags, start=1):
        if flag:
            hits += 1
            weighted += hits / rank
    return weighted / hits if hits else 0.0


def context_recall(*, reference: str, contexts: list[str], client: GeminiClient, cfg: Settings | None = None) -> float:
    """Share of golden-answer sentences attributable to the retrieved context."""
    cfg = cfg or default_settings
    if not reference.strip() or not contexts:
        return float("nan")
    payload = client.generate_json(
        "Split the reference answer into sentences. For each sentence, answer true if the "
        "CONTEXT supports it, false otherwise. Return one boolean per sentence, in order.\n\n"
        f"CONTEXT\n{_truncate(chr(10).join(contexts), 20000)}\n\n"
        f"REFERENCE ANSWER\n{_truncate(reference, 6000)}",
        schema=USEFULNESS_SCHEMA,
        model=cfg.judge_model,
        temperature=0.0,
    )
    flags = payload.get("useful", [])
    if not flags:
        return float("nan")
    return sum(1 for flag in flags if flag) / len(flags)


def score_rag(
    *,
    question: str,
    answer: str,
    reference: str,
    contexts: list[str],
    client: GeminiClient,
    cfg: Settings | None = None,
) -> dict[str, Any]:
    faith = faithfulness(answer=answer, contexts=contexts, client=client, cfg=cfg)
    return {
        "faithfulness": faith["faithfulness"],
        "n_claims": faith["n_claims"],
        "rag_unsupported_claims": faith["unsupported_claims"],
        "answer_relevancy": answer_relevancy(question=question, answer=answer, client=client, cfg=cfg),
        "context_precision": context_precision(
            question=question, reference=reference, contexts=contexts, client=client, cfg=cfg
        ),
        "context_recall": context_recall(reference=reference, contexts=contexts, client=client, cfg=cfg),
    }
