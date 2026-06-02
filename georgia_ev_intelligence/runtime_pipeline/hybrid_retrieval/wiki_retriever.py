"""
Wiki-based retriever for hybrid retrieval pipeline.
Complements BM25 and dense retrieval by searching pre-synthesized wiki pages.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from georgia_ev_intelligence.kb_builder.llm_wiki import LLMWiki, WikiPage


@dataclass
class WikiRetrievalResult:
    """Result from wiki retrieval."""
    title: str
    entity_type: str
    content: str
    score: float
    source_count: int


class WikiRetriever:
    """Retrieves relevant information from the LLM wiki."""

    def __init__(self, wiki_dir: str = "kb/wiki"):
        self.wiki_dir = Path(wiki_dir)
        self.wiki: Optional[LLMWiki] = None

        if self.wiki_dir.exists():
            try:
                self.wiki = LLMWiki(wiki_dir=wiki_dir)
                self.available = True
            except Exception as e:
                print(f"Warning: Could not load wiki: {e}")
                self.available = False
        else:
            self.available = False

    def retrieve(self, query: str, top_k: int = 5) -> list[WikiRetrievalResult]:
        """Retrieve wiki pages relevant to the query."""
        if not self.available or not self.wiki:
            return []

        results = []

        # Search the wiki index
        search_results = self.wiki.search(query, top_k=top_k)

        for result in search_results:
            page = self.wiki.get_page(result["title"])
            if page:
                results.append(
                    WikiRetrievalResult(
                        title=result["title"],
                        entity_type=result["entity_type"],
                        content=page.content,
                        score=result["score"],
                        source_count=len(result["sources"]),
                    )
                )

        return results

    def get_page(self, title: str) -> Optional[WikiPage]:
        """Retrieve a specific wiki page."""
        if not self.available or not self.wiki:
            return None
        return self.wiki.get_page(title)

    def list_entities(self, entity_type: Optional[str] = None) -> list[str]:
        """List all entities in the wiki."""
        if not self.available or not self.wiki:
            return []
        return self.wiki.list_pages(entity_type=entity_type)

    def get_related(self, title: str) -> list[str]:
        """Get related entities for a title."""
        if not self.available or not self.wiki:
            return []
        return self.wiki.get_related_pages(title)

    def format_for_context(self, results: list[WikiRetrievalResult]) -> str:
        """Format wiki results as context string for LLM."""
        if not results:
            return ""

        context = "## Wiki Context\n\n"
        for result in results:
            context += f"### {result.title} ({result.entity_type})\n\n"
            context += result.content[:500] + "\n\n"
            if result.source_count > 0:
                context += f"*[From {result.source_count} source(s)]*\n\n"

        return context
