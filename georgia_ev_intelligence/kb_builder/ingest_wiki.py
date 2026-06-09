"""
Ingest documents into the LLM-based wiki.
Usage: python -m georgia_ev_intelligence.kb_builder.ingest_wiki [--source file.jsonl] [--limit N]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Generator

from .llm_wiki import LLMWiki


def load_jsonl(file_path: str) -> Generator[dict, None, None]:
    """Stream JSONL documents from a file."""
    with open(file_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON line, skipping", file=sys.stderr)
                    continue


def ingest(source_file: str, limit: int = 0, wiki_dir: str = "kb/wiki"):
    """Ingest documents from a JSONL file into the wiki."""
    wiki = LLMWiki(wiki_dir=wiki_dir)

    print(f"[wiki] Ingesting from {source_file}")
    print(f"[wiki] Wiki directory: {wiki_dir}")

    doc_count = 0
    page_updates = 0

    for doc in load_jsonl(source_file):
        if limit > 0 and doc_count >= limit:
            break

        doc_id = doc.get("doc_id", f"doc_{doc_count}")

        # Skip if already processed
        if doc_id in wiki.index["sources_processed"]:
            print(f"[wiki] Skipping already-processed: {doc_id}")
            continue

        print(f"[wiki] Processing {doc_count + 1}: {doc.get('title', 'Untitled')}")

        try:
            updated = wiki.ingest_document(doc_id, doc)
            page_updates += len(updated)
            print(f"      -> Updated {len(updated)} pages: {', '.join(updated)}")
        except Exception as e:
            print(f"      ERROR: {e}", file=sys.stderr)

        doc_count += 1

    print(f"\n[wiki] Done!")
    print(f"[wiki] Documents processed: {doc_count}")
    print(f"[wiki] Page updates: {page_updates}")
    print(f"[wiki] Total pages: {len(wiki.index['pages'])}")

    return wiki


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into LLM wiki")
    parser.add_argument(
        "--source",
        default="kb/raw_docs/ddg_search.jsonl",
        help="Source JSONL file (default: kb/raw_docs/ddg_search.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of documents to process (0 = all)",
    )
    parser.add_argument(
        "--wiki-dir",
        default="kb/wiki",
        help="Wiki output directory (default: kb/wiki)",
    )
    parser.add_argument(
        "--export",
        help="Export wiki to markdown file after ingestion",
    )

    args = parser.parse_args()

    # Convert relative paths to absolute
    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path

    if not source_path.exists():
        print(f"Error: Source file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    wiki = ingest(str(source_path), limit=args.limit, wiki_dir=args.wiki_dir)

    if args.export:
        export_path = Path(args.export)
        if not export_path.is_absolute():
            export_path = Path.cwd() / export_path
        export_path.parent.mkdir(parents=True, exist_ok=True)

        output = wiki.export_as_markdown(str(export_path))
        print(f"[wiki] Exported to {output}")


if __name__ == "__main__":
    main()
