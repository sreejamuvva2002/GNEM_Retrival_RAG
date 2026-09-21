"""Retrieval over the internal company workbook.

Two strategies:

* ``bm25``   - pure-Python Okapi BM25, no API calls, fully deterministic.
* ``hybrid`` - BM25 plus Gemini embeddings, fused with Reciprocal Rank Fusion.

Hybrid matters here because the questions mix keyword-ish lookups ("Tier 1/2
suppliers") with semantic ones ("who could supply a battery plant"), and the two
retrievers fail on opposite halves of that split.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .config import Settings, settings as default_settings
from .data import Doc
from .gemini_client import GeminiClient

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "on", "with", "is", "are",
    "which", "what", "who", "how", "all", "list", "show", "their", "that", "this", "by",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(str(text).lower()) if t not in _STOP and len(t) > 1]


@dataclass
class Hit:
    doc: Doc
    score: float
    rank: int


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75). Small corpora, so a plain dict index is fine."""

    def __init__(self, docs: list[Doc], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(doc.text) for doc in docs]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / max(len(docs), 1)
        self.freqs = [Counter(tokens) for tokens in self.doc_tokens]
        df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        n = len(docs)
        self.idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        terms = tokenize(query)
        scores: list[float] = []
        for index, freq in enumerate(self.freqs):
            score = 0.0
            for term in terms:
                tf = freq.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[index] / max(self.avg_len, 1e-9))
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            scores.append(score)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in ranked[:k] if scores[i] > 0]


class Retriever:
    def __init__(self, docs: list[Doc], client: GeminiClient | None = None, cfg: Settings | None = None):
        self.cfg = cfg or default_settings
        self.docs = docs
        self.client = client or GeminiClient(self.cfg)
        self.bm25 = BM25(docs)
        self._matrix = None

    @property
    def matrix(self):
        if self._matrix is None:
            self._matrix = self.client.embed([doc.text for doc in self.docs], task_type="RETRIEVAL_DOCUMENT")
        return self._matrix

    def search(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or self.cfg.top_k
        lexical = self.bm25.search(query, k * 2)
        if self.cfg.retrieval_strategy != "hybrid":
            return [Hit(self.docs[i], score, rank) for rank, (i, score) in enumerate(lexical[:k], start=1)]

        query_vector = self.client.embed([query], task_type="RETRIEVAL_QUERY")[0]
        sims = self.matrix @ query_vector
        dense_order = sorted(range(len(self.docs)), key=lambda i: float(sims[i]), reverse=True)[: k * 2]
        dense = [(i, float(sims[i])) for i in dense_order]
        return self._fuse(lexical, dense, k)

    def _fuse(self, lexical: list[tuple[int, float]], dense: list[tuple[int, float]], k: int, c: int = 60) -> list[Hit]:
        """Reciprocal Rank Fusion: rank-based, so the two score scales never have
        to be calibrated against each other."""
        fused: dict[int, float] = {}
        for ranking in (lexical, dense):
            for rank, (index, _) in enumerate(ranking, start=1):
                fused[index] = fused.get(index, 0.0) + 1.0 / (c + rank)
        order = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
        return [Hit(self.docs[index], score, rank) for rank, (index, score) in enumerate(order, start=1)]


def format_context(hits: list[Hit], max_chars: int = 24000) -> str:
    """Render hits as numbered blocks the model is told to cite as [C1], [C2]..."""
    blocks, used = [], 0
    for hit in hits:
        block = f"[C{hit.rank}] {hit.doc.text}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)
