"""
Shared text utilities for query processing.

Extracted from term_matcher.py to break the tight coupling between
term_matcher and keyword_resolver. Both modules import from here
instead of one importing private functions from the other.
"""
from __future__ import annotations

import re


# Minimum character length for a KB value/component to be considered a match.
MIN_MATCH_LEN = 3


def norm_text(text: str) -> str:
    """Lowercase and normalize whitespace."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def tokens(text: str) -> list[str]:
    """Tokenize while preserving slash-style tokens such as 1/2."""
    return re.findall(r"[a-z0-9]+(?:/[a-z0-9]+)?", norm_text(text))


def singularize_token(tok: str) -> str:
    """
    Very small singularization helper.

    Enough for supplier/suppliers, companies/company, counties/county.
    Avoids external dependencies.
    """
    tok = tok.lower()

    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def token_set(text: str) -> set[str]:
    toks = tokens(text)
    out = set(toks)
    out.update(singularize_token(t) for t in toks)
    return {t for t in out if t}


def contains_phrase(text: str, phrase: str) -> bool:
    """
    Safer phrase containment with flexible spacing and word boundaries.

    This avoids matching tiny substrings accidentally.
    """
    phrase = str(phrase).strip()
    if not phrase:
        return False

    pattern = r"\b" + r"\s+".join(re.escape(p) for p in phrase.lower().split()) + r"\b"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def normalise_for_comparison(text: str) -> str:
    """Normalise text for value comparison: lowercase, collapse whitespace/slashes."""
    return re.sub(r"[\s/]+", " ", str(text).lower()).strip()


def extract_question_ngrams(question: str, max_ngram: int = 6) -> list[str]:
    """
    Extract word n-grams from the question for matching against live KB values.

    Preserves slash notation (e.g. "1/2") as single tokens and generates
    n-grams from 1 to max_ngram words, longest first.
    """
    # Tokenize preserving slashes inside words
    toks = re.findall(r"[A-Za-z0-9]+(?:/[A-Za-z0-9]+)*", question)
    if not toks:
        return []

    ngrams: list[str] = []
    for n in range(min(max_ngram, len(toks)), 0, -1):
        for i in range(len(toks) - n + 1):
            phrase = " ".join(toks[i : i + n])
            if len(phrase) >= MIN_MATCH_LEN:
                ngrams.append(phrase)
    return ngrams
