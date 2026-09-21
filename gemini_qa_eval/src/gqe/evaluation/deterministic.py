"""Layer 1 - deterministic, reference-based metrics.

Cheap, reproducible, no API calls. These do not measure whether an answer is
*good*; they measure whether it contains the facts the golden answer contains.
Run them on every answer, every time, as the regression signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..data import normalise_name

_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*)(?:\.\d+)?(?![\w])")
_REFUSAL = (
    "i don't know", "i do not know", "cannot determine", "can't determine",
    "not contain", "no information", "unable to answer", "not available in",
    "insufficient information", "does not include", "not specified in",
)


def tokens(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def token_f1(prediction: str, reference: str) -> dict[str, float]:
    """Bag-of-words overlap (SQuAD-style). Blunt, but it moves when content moves."""
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    common = set(pred) & set(ref)
    matched = sum(min(pred.count(t), ref.count(t)) for t in common)
    precision = matched / len(pred)
    recall = matched / len(ref)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure via longest common subsequence - rewards ordered overlap."""
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return 0.0
    previous = [0] * (len(ref) + 1)
    for p in pred:
        current = [0]
        for j, r in enumerate(ref):
            current.append(previous[j] + 1 if p == r else max(current[j], previous[j + 1]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def entity_scores(prediction: str, reference: str, known_entities: list[str]) -> dict[str, float]:
    """The metric that actually matters for this dataset.

    The golden answers are lists of companies. So: which companies does the
    golden answer name, which does the prediction name, and how do those sets
    compare? Recall catches omissions; precision catches invented suppliers.
    """
    pred_entities = find_entities(prediction, known_entities)
    ref_entities = find_entities(reference, known_entities)
    if not ref_entities:
        return {
            "entity_precision": float("nan"),
            "entity_recall": float("nan"),
            "entity_f1": float("nan"),
            "entities_expected": 0,
            "entities_found": len(pred_entities),
            "entities_hallucinated": float("nan"),
        }
    hits = pred_entities & ref_entities
    precision = len(hits) / len(pred_entities) if pred_entities else 0.0
    recall = len(hits) / len(ref_entities)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "entities_expected": len(ref_entities),
        "entities_found": len(pred_entities),
        # Named in the answer, absent from the golden answer. Not automatically
        # wrong (the golden answer can be incomplete) but always worth a look.
        "entities_hallucinated": len(pred_entities - ref_entities),
    }


def find_entities(text: str, known_entities: list[str]) -> set[str]:
    """Match known company names in free text, suffix- and case-insensitively."""
    haystack = " " + normalise_name(text) + " "
    found = set()
    for name in known_entities:
        key = normalise_name(name)
        if len(key) < 3:
            continue
        if f" {key} " in haystack:
            found.add(key)
    return found


def numeric_agreement(prediction: str, reference: str) -> dict[str, float]:
    """Do the counts and figures in the answer match the golden answer?

    "There are 18 Tier 1/2 companies" vs "There are 12" is the single most
    visible failure mode for this question set, and no text-overlap metric
    catches it.
    """
    pred_numbers = _numbers(prediction)
    ref_numbers = _numbers(reference)
    if not ref_numbers:
        return {"numeric_recall": float("nan"), "numeric_match": float("nan"), "numbers_expected": 0}
    matched = len(pred_numbers & ref_numbers)
    return {
        "numeric_recall": matched / len(ref_numbers),
        # The leading figure is usually the headline count; check it separately.
        "numeric_match": float(bool(pred_numbers & ref_numbers)),
        "numbers_expected": len(ref_numbers),
    }


def _numbers(text: str) -> set[float]:
    values = set()
    for raw in _NUMBER.findall(str(text)):
        try:
            values.add(float(raw.replace(",", "")))
        except ValueError:
            continue
    return values


def is_refusal(text: str) -> bool:
    """Abstention detection. An abstention is a good outcome when the evidence
    is genuinely absent and a bad one when it is not - the report splits them."""
    lowered = str(text).lower()
    return any(marker in lowered for marker in _REFUSAL)


def citation_rate(text: str) -> float:
    """Fraction of sentences carrying a [C#] or [WEB] marker (modes 2 and 3)."""
    parts = re.split(r"(?<=[.!?])\s+", str(text))
    sentences: list[str] = []
    for part in parts:
        # A trailing "[C3]" on its own belongs to the sentence it follows.
        if sentences and re.fullmatch(r"[\[\](),.;:\sCWEBcweb0-9]+", part):
            sentences[-1] += " " + part
        else:
            sentences.append(part)
    sentences = [s for s in sentences if len(s.split()) > 3]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r"\[(C\d+|WEB)\]", s))
    return cited / len(sentences)


@dataclass
class DeterministicScores:
    scores: dict[str, float]

    def as_dict(self) -> dict[str, float]:
        return self.scores


def score_answer(prediction: str, reference: str, known_entities: list[str]) -> dict[str, float]:
    overlap = token_f1(prediction, reference)
    result: dict[str, float] = {
        "token_precision": overlap["precision"],
        "token_recall": overlap["recall"],
        "token_f1": overlap["f1"],
        "rouge_l": rouge_l(prediction, reference),
        "answer_length_words": float(len(tokens(prediction))),
        "is_refusal": float(is_refusal(prediction)),
        "citation_rate": citation_rate(prediction),
    }
    result.update(entity_scores(prediction, reference, known_entities))
    result.update(numeric_agreement(prediction, reference))
    return result
