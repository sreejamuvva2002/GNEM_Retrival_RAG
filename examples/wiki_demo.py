"""
Demo script: Building and querying an LLM-based wiki from DDG search results.

This script demonstrates:
1. Ingesting documents into a wiki
2. Searching the wiki
3. Integrating with hybrid retrieval
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from georgia_ev_intelligence.kb_builder.llm_wiki import LLMWiki
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.wiki_retriever import (
    WikiRetriever,
)


def demo_basic_operations():
    """Demo: Basic wiki operations (assume wiki is already built)."""
    print("\n" + "="*60)
    print("DEMO: Basic Wiki Operations")
    print("="*60)

    wiki = LLMWiki(wiki_dir="kb/wiki")

    # Show statistics
    print(f"\nWiki Statistics:")
    print(f"  Total pages: {len(wiki.index['pages'])}")
    print(f"  Documents processed: {len(wiki.index['sources_processed'])}")

    # List some pages
    pages = wiki.list_pages()[:5]
    print(f"\nFirst 5 pages:")
    for title in pages:
        print(f"  - {title}")

    # Search
    print(f"\nSearching for 'investment':")
    results = wiki.search("investment", top_k=3)
    for result in results:
        print(f"  ✓ {result['title']} (score: {result['score']})")

    # Show a page
    if pages:
        first_page = pages[0]
        print(f"\n{'='*60}")
        print(f"Page: {first_page}")
        print(f"{'='*60}")
        page = wiki.get_page(first_page)
        if page:
            print(f"Type: {page.entity_type}")
            print(f"Sources: {len(page.sources)}")
            print(f"Related: {', '.join(page.related_entities[:3]) if page.related_entities else 'None'}")
            print(f"\n{page.content[:300]}...\n")


def demo_wiki_retriever():
    """Demo: Using wiki retriever for lookup."""
    print("\n" + "="*60)
    print("DEMO: Wiki Retriever")
    print("="*60)

    retriever = WikiRetriever(wiki_dir="kb/wiki")

    if not retriever.available:
        print("Wiki not available yet. Run ingestion first.")
        return

    # Search queries
    queries = [
        "SK Innovation",
        "battery manufacturing",
        "Georgia investment",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=2)
        if results:
            for result in results:
                print(f"  ✓ {result.title}")
                print(f"    Entity Type: {result.entity_type}")
                print(f"    Score: {result.score}")
                print(f"    Preview: {result.content[:100]}...")
        else:
            print(f"  (no results)")


def demo_integration():
    """Demo: Show how wiki would integrate with hybrid retrieval."""
    print("\n" + "="*60)
    print("DEMO: Integration with Hybrid Retrieval")
    print("="*60)

    print("""
When integrated with your hybrid pipeline:

1. User asks: "What companies are investing in Georgia EV?"

2. Wiki retriever searches index (10ms):
   ✓ SK Innovation
   ✓ Duckyang
   ✓ Vanderlande

3. Hybrid retrieval runs in parallel (1000ms):
   ✓ BM25: finds 250 child chunks
   ✓ Dense: finds 250 child chunks
   ✓ Rerank: keeps top 45 parent chunks

4. Combined context passed to LLM:
   ---
   ## Wiki Context

   ### SK Innovation (company)
   South Korea's largest energy company...

   ### Duckyang (company)
   Battery modules and storage supplier...

   ## Hybrid Retrieved Context
   [Detailed facts from original documents...]
   ---

5. LLM generates comprehensive answer with both:
   - Synthesized wiki facts (clear, organized)
   - Raw document evidence (detailed, cited)
    """)


def demo_setup_instructions():
    """Show instructions for full demo."""
    print("\n" + "="*60)
    print("FULL DEMO SETUP (in order)")
    print("="*60)

    print("""
1. First, ingest documents into the wiki:

   python -m georgia_ev_intelligence.kb_builder.wiki_cli ingest \\
     --source kb/raw_docs/ddg_search.jsonl \\
     --limit 20 \\
     --export outputs/wiki_preview.md

   (This creates kb/wiki/ with pages and index)

2. Then run this demo:

   python examples/wiki_demo.py

3. For full integration with your pipeline, see:

   docs/LLM_WIKI.md

4. Search and browse the wiki:

   python -m georgia_ev_intelligence.kb_builder.wiki_cli search "SK"
   python -m georgia_ev_intelligence.kb_builder.wiki_cli show "SK Innovation"
   python -m georgia_ev_intelligence.kb_builder.wiki_cli list
    """)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("LLM-Wiki Demo")
    print("="*60)

    # Check if wiki exists
    wiki_path = Path("kb/wiki")
    if not wiki_path.exists() or not (wiki_path / "_index.json").exists():
        print("\n⚠️  Wiki not found at kb/wiki/")
        demo_setup_instructions()
    else:
        demo_basic_operations()
        demo_wiki_retriever()
        demo_integration()

    print("\n" + "="*60)
    print("For more details, see: docs/LLM_WIKI.md")
    print("="*60 + "\n")
