"""
CLI tool for managing the LLM wiki.
Usage: python -m georgia_ev_intelligence.kb_builder.wiki_cli [command] [options]
"""

import argparse
import json
import sys
from pathlib import Path

from .llm_wiki import LLMWiki


def cmd_ingest(args):
    """Ingest documents from JSONL."""
    from .ingest_wiki import ingest

    wiki = ingest(
        source_file=args.source,
        limit=args.limit,
        wiki_dir=args.wiki_dir,
    )

    if args.export:
        wiki.export_as_markdown(args.export)
        print(f"✓ Exported wiki to {args.export}")


def cmd_search(args):
    """Search the wiki."""
    wiki = LLMWiki(wiki_dir=args.wiki_dir)

    results = wiki.search(args.query, top_k=args.top_k)

    if not results:
        print(f"No results found for: {args.query}")
        return

    print(f"\n🔍 Search Results for '{args.query}':\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} [{result['entity_type']}]")
        print(f"   Score: {result['score']}")
        print(f"   Sources: {len(result['sources'])}")
        print(f"   Preview: {result['preview'][:100]}...")
        print()


def cmd_show(args):
    """Display a wiki page."""
    wiki = LLMWiki(wiki_dir=args.wiki_dir)
    page = wiki.get_page(args.title)

    if not page:
        print(f"Page not found: {args.title}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Title: {page.title}")
    print(f"Type: {page.entity_type}")
    print(f"Last Updated: {page.last_updated}")
    print(f"Related: {', '.join(page.related_entities) if page.related_entities else 'None'}")
    print(f"Sources: {', '.join(page.sources)}")
    print(f"{'='*60}\n")
    print(page.content)
    print(f"\n{'='*60}")


def cmd_list(args):
    """List all wiki pages."""
    wiki = LLMWiki(wiki_dir=args.wiki_dir)

    pages = wiki.list_pages(entity_type=args.type)

    if not pages:
        print("No pages found")
        return

    print(f"\nWiki Pages ({len(pages)} total):\n")

    for title in pages[:args.limit or None]:
        page = wiki.get_page(title)
        if page:
            print(f"• {title}")
            print(f"  Type: {page.entity_type}")
            print(f"  Sources: {len(page.sources)}")
            print(f"  Related: {len(page.related_entities)}")
            print()


def cmd_export(args):
    """Export wiki to markdown."""
    wiki = LLMWiki(wiki_dir=args.wiki_dir)
    output = wiki.export_as_markdown(args.output)
    print(f"✓ Exported wiki to {output}")


def cmd_stats(args):
    """Show wiki statistics."""
    wiki = LLMWiki(wiki_dir=args.wiki_dir)
    index = wiki.index

    print(f"\n📊 Wiki Statistics:\n")
    print(f"Total Pages: {len(index['pages'])}")
    print(f"Documents Processed: {len(index['sources_processed'])}")
    print(f"Unique Entities: {len(index['entities'])}")
    print(f"Last Updated: {index.get('last_updated', 'Never')}")

    # Entity type breakdown
    by_type = {}
    for title, metadata in index["pages"].items():
        entity_type = metadata.get("entity_type", "other")
        by_type[entity_type] = by_type.get(entity_type, 0) + 1

    print(f"\nBy Entity Type:")
    for entity_type in sorted(by_type.keys()):
        print(f"  {entity_type}: {by_type[entity_type]}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Manage the LLM-based wiki",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest documents
  python -m georgia_ev_intelligence.kb_builder.wiki_cli ingest --source kb/raw_docs/ddg_search.jsonl

  # Search
  python -m georgia_ev_intelligence.kb_builder.wiki_cli search "SK Innovation"

  # Show page
  python -m georgia_ev_intelligence.kb_builder.wiki_cli show "SK Innovation"

  # List all companies
  python -m georgia_ev_intelligence.kb_builder.wiki_cli list --type company

  # Export to markdown
  python -m georgia_ev_intelligence.kb_builder.wiki_cli export --output wiki.md

  # Show statistics
  python -m georgia_ev_intelligence.kb_builder.wiki_cli stats
        """,
    )

    parser.add_argument(
        "--wiki-dir",
        default="kb/wiki",
        help="Wiki directory (default: kb/wiki)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents from JSONL")
    p_ingest.add_argument(
        "--source",
        default="kb/raw_docs/ddg_search.jsonl",
        help="Source JSONL file",
    )
    p_ingest.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit documents (0=all)",
    )
    p_ingest.add_argument(
        "--export",
        help="Export to markdown after ingestion",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # search
    p_search = subparsers.add_parser("search", help="Search wiki")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results",
    )
    p_search.set_defaults(func=cmd_search)

    # show
    p_show = subparsers.add_parser("show", help="Display a page")
    p_show.add_argument("title", help="Page title")
    p_show.set_defaults(func=cmd_show)

    # list
    p_list = subparsers.add_parser("list", help="List pages")
    p_list.add_argument(
        "--type",
        help="Filter by entity type",
    )
    p_list.add_argument(
        "--limit",
        type=int,
        help="Limit results",
    )
    p_list.set_defaults(func=cmd_list)

    # export
    p_export = subparsers.add_parser("export", help="Export wiki")
    p_export.add_argument(
        "--output",
        default="wiki.md",
        help="Output markdown file",
    )
    p_export.set_defaults(func=cmd_export)

    # stats
    p_stats = subparsers.add_parser("stats", help="Show statistics")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
