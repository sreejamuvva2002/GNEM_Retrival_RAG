"""
Enhanced orchestrator that combines wiki retrieval with hybrid BM25/dense/rerank pipeline.
"""

from dataclasses import dataclass

from .orchestrator import HybridRetrievalOrchestrator
from .wiki_retriever import WikiRetriever, WikiRetrievalResult


@dataclass
class EnhancedRetrievalResult:
    """Combined result from wiki + hybrid retrieval."""
    question: str
    wiki_results: list[WikiRetrievalResult]
    hybrid_context: str  # Original hybrid retrieval output
    combined_context: str  # Wiki + hybrid combined


class WikiEnhancedOrchestrator:
    """Orchestrator that combines wiki and hybrid retrieval."""

    def __init__(
        self,
        hybrid_orchestrator: HybridRetrievalOrchestrator,
        wiki_retriever: WikiRetriever,
        wiki_top_k: int = 3,
    ):
        self.hybrid = hybrid_orchestrator
        self.wiki = wiki_retriever
        self.wiki_top_k = wiki_top_k

    def retrieve_enhanced(self, question: str) -> EnhancedRetrievalResult:
        """
        Retrieve context from both wiki and hybrid pipeline.
        Wiki results come first (pre-synthesized), then hybrid retrieval.
        """
        # Get wiki results
        wiki_results = self.wiki.retrieve(question, top_k=self.wiki_top_k)

        # Get hybrid retrieval results (original pipeline)
        hybrid_response = self.hybrid.retrieve(question)
        hybrid_context = hybrid_response.context

        # Combine contexts: wiki first, then hybrid
        combined_context = ""

        if wiki_results:
            combined_context += self.wiki.format_for_context(wiki_results) + "\n"

        combined_context += hybrid_context

        return EnhancedRetrievalResult(
            question=question,
            wiki_results=wiki_results,
            hybrid_context=hybrid_context,
            combined_context=combined_context,
        )

    def get_hybrid_orchestrator(self) -> HybridRetrievalOrchestrator:
        """Get the underlying hybrid orchestrator for direct access."""
        return self.hybrid

    def get_wiki_retriever(self) -> WikiRetriever:
        """Get the wiki retriever for direct access."""
        return self.wiki
