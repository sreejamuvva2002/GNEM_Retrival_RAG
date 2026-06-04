"""Command-line interface for the Markdown extraction pipeline (README §33, §34).

Examples
--------
Dry run (list + report only, converts nothing):
    python -m georgia_ev_intelligence.markdown_extraction --dry-run --limit 50

Balanced sample batch, local output only:
    python -m georgia_ev_intelligence.markdown_extraction --sample --local-only

Convert a small batch and upload to B2:
    python -m georgia_ev_intelligence.markdown_extraction --batch-size 20 --limit 20

Re-process everything for a new extraction version:
    python -m georgia_ev_intelligence.markdown_extraction --force --extraction-version v2
"""
from __future__ import annotations

import argparse
import logging

from georgia_ev_intelligence.shared import config

from . import detector
from .converters import SUPPORTED_TYPES
from .manifest import ManifestStore
from .pipeline import PipelineOptions, run_batch
from .registry import build_registry

logger = logging.getLogger(__name__)

# Balanced sample plan (README §31, Phase 1).
_SAMPLE_PLAN = {
    "pdf": 20, "html": 20, "docx": 10, "excel": 5, "csv": 5,
    "json": 5, "xml": 5, "image": 10, "text": 10,
}


def _type_of(record: dict) -> str | None:
    return detector.EXTENSION_TO_TYPE.get(record.get("file_extension", ""))


def _select_records(registry: dict[str, dict], args: argparse.Namespace) -> list[dict]:
    """Pick which registry records to process given the CLI flags."""
    selected: list[dict] = []
    for rec in registry.values():
        ftype = _type_of(rec)
        if args.file_type != "all" and ftype != args.file_type:
            continue
        status = rec.get("processing_status")
        if args.force:
            selected.append(rec)
        elif args.retry_failed and status == "failed":
            selected.append(rec)
        elif args.reprocess_needs_review and status == "needs_review":
            selected.append(rec)
        elif status in (None, "pending"):
            selected.append(rec)
    return selected


def _apply_sample(records: list[dict]) -> list[dict]:
    """Take a balanced sample across file types (README §31)."""
    buckets: dict[str, list[dict]] = {}
    for rec in records:
        buckets.setdefault(_type_of(rec) or "other", []).append(rec)
    sample: list[dict] = []
    for ftype, quota in _SAMPLE_PLAN.items():
        sample.extend(buckets.get(ftype, [])[:quota])
    return sample


def _print_dry_run(registry: dict[str, dict], to_process: list[dict]) -> None:
    total = len(registry)
    supported = sum(1 for r in registry.values() if _type_of(r))
    unsupported = total - supported
    dist: dict[str, int] = {}
    for rec in registry.values():
        dist[_type_of(rec) or "unsupported"] = dist.get(_type_of(rec) or "unsupported", 0) + 1

    print("\n=== Dry run report ===")
    print(f"Total objects found: {total:,}")
    print(f"Supported files:     {supported:,}")
    print(f"Unsupported files:   {unsupported:,}")
    for ftype in sorted(dist):
        print(f"  {ftype:<12} {dist[ftype]:,}")
    print(f"Would process:       {len(to_process):,} files")
    print("Sample source keys:")
    for rec in to_process[:5]:
        print(f"  - {rec['source_key']}")
        print(f"      -> {config.MARKDOWN_B2_PREFIX.rstrip('/')}/source_documents/{rec['document_id']}.md")
    print("Dry run complete. No files converted.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GNEM — raw B2 documents → Markdown extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source-prefix", action="append", default=None,
                        help="B2 prefix(es) to list (repeatable / comma list). "
                             "Default: RAW_B2_PREFIXES from config.")
    parser.add_argument("--output-prefix", default=config.MARKDOWN_B2_PREFIX,
                        help=f"Markdown output prefix (default: {config.MARKDOWN_B2_PREFIX})")
    parser.add_argument("--manifest-prefix", default=config.MANIFEST_B2_PREFIX,
                        help=f"Manifest prefix (default: {config.MANIFEST_B2_PREFIX})")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="Process only the first N selected files (0 = unlimited)")
    parser.add_argument("--file-type",
                        choices=["all", *SUPPORTED_TYPES],
                        default="all", help="Restrict to one internal file type")
    parser.add_argument("--dry-run", action="store_true",
                        help="List + report only; convert nothing")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even already-successful documents")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Include previously failed documents")
    parser.add_argument("--reprocess-needs-review", action="store_true",
                        help="Re-process documents previously marked needs_review")
    parser.add_argument("--sample", action="store_true",
                        help="Balanced sample across file types (README §31)")
    parser.add_argument("--local-only", action="store_true",
                        help="Read raw files from B2 but write Markdown/manifests locally only")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip raw_documents DB enrichment of the registry")
    parser.add_argument("--extraction-version", default=config.EXTRACTION_VERSION)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    # Resolve source prefixes (CLI overrides config; supports comma lists).
    if args.source_prefix:
        prefixes: list[str] = []
        for item in args.source_prefix:
            prefixes.extend(p.strip() for p in item.split(",") if p.strip())
    else:
        prefixes = config.RAW_B2_PREFIXES

    store = ManifestStore(use_b2=not args.local_only)

    logger.info("Building registry from %d prefix(es)...", len(prefixes))
    registry = build_registry(prefixes, store=store, enrich_db=not args.no_db)
    logger.info("Registry has %d document(s).", len(registry))

    to_process = _select_records(registry, args)
    if args.sample:
        to_process = _apply_sample(to_process)
    if args.limit:
        to_process = to_process[: args.limit]

    if args.dry_run:
        _print_dry_run(registry, to_process)
        return

    if not to_process:
        print("Nothing to process (all selected files already converted). "
              "Use --force or --retry-failed to re-run.")
        return

    opts = PipelineOptions(
        output_prefix=args.output_prefix,
        extraction_version=args.extraction_version,
        use_b2=not args.local_only,
        write_local=True,
        force=args.force,
        max_file_size_mb=config.MAX_FILE_SIZE_MB,
    )

    counts = run_batch(
        to_process, store=store, opts=opts, registry=registry,
        batch_size=args.batch_size,
    )

    print("\n=== Conversion summary ===")
    for status in sorted(counts):
        print(f"  {status:<20} {counts[status]:,}")
    print(f"  {'total':<20} {sum(counts.values()):,}")
    if not args.local_only and config.B2_BUCKET_NAME:
        print(f"Markdown uploaded under: {args.output_prefix}")
        print(f"Manifests synced under:  {args.manifest_prefix}")
    print(f"Local Markdown: {config.LOCAL_MARKDOWN_DIR}")
    print(f"Local manifests: {config.LOCAL_MANIFEST_DIR}")


if __name__ == "__main__":
    main()
