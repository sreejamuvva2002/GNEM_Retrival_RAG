"""Command-line interface for fine-tuning pipeline."""
import argparse
import logging
import sys
from pathlib import Path

from .data_augmentation import augment_dataset
from .dataset_formatter import format_validated_dataset
from .validation import validate_augmented_dataset

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def command_augment(args):
    """Run data augmentation."""
    logger.info("Starting data augmentation...")
    augment_dataset(
        paraphrase_count=args.paraphrase_count,
        kb_questions_per_chunk=args.kb_questions_per_chunk,
        include_adversarial=args.include_adversarial,
    )
    logger.info("Data augmentation complete!")


def command_validate(args):
    """Run validation on augmented data."""
    logger.info("Starting validation...")
    passed, failed = validate_augmented_dataset(
        input_path=Path(args.input) if args.input else None,
        min_score=args.min_score,
    )
    logger.info(
        f"Validation summary: {len(passed)} passed, {len(failed)} failed"
    )


def command_format(args):
    """Format validated data into training datasets."""
    logger.info("Formatting datasets...")
    train_path, val_path = format_validated_dataset(
        input_path=Path(args.input) if args.input else None,
        train_ratio=args.train_ratio,
    )
    if train_path and val_path:
        logger.info(f"Training dataset: {train_path}")
        logger.info(f"Validation dataset: {val_path}")


def command_pipeline(args):
    """Run full pipeline: augment -> validate -> format."""
    logger.info("Running full pipeline...")

    # Step 1: Augment
    logger.info("\n[1/3] Data Augmentation...")
    augment_dataset(
        paraphrase_count=args.paraphrase_count,
        kb_questions_per_chunk=args.kb_questions_per_chunk,
        include_adversarial=args.include_adversarial,
    )

    # Step 2: Validate
    logger.info("\n[2/3] Validation...")
    passed, failed = validate_augmented_dataset(min_score=args.min_score)

    # Step 3: Format
    logger.info("\n[3/3] Dataset Formatting...")
    train_path, val_path = format_validated_dataset(
        train_ratio=args.train_ratio
    )

    logger.info("\n✓ Pipeline complete!")
    logger.info(f"  Training dataset: {train_path}")
    logger.info(f"  Validation dataset: {val_path}")
    logger.info(f"  Quality: {len(passed)}/{len(passed)+len(failed)} pairs passed validation")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fine-tuning pipeline for Qwen model"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Augmentation command
    augment_parser = subparsers.add_parser(
        "augment",
        help="Generate synthetic Q&A pairs",
    )
    augment_parser.add_argument(
        "--paraphrase-count",
        type=int,
        default=8,
        help="Number of paraphrases per question",
    )
    augment_parser.add_argument(
        "--kb-questions-per-chunk",
        type=int,
        default=3,
        help="Number of KB-driven questions per chunk",
    )
    augment_parser.add_argument(
        "--no-adversarial",
        dest="include_adversarial",
        action="store_false",
        help="Skip adversarial question generation",
    )
    augment_parser.set_defaults(func=command_augment)

    # Validation command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate synthetic Q&A pairs",
    )
    validate_parser.add_argument(
        "--input",
        help="Path to augmented_questions.jsonl",
    )
    validate_parser.add_argument(
        "--min-score",
        type=int,
        default=4,
        help="Minimum acceptable average score",
    )
    validate_parser.set_defaults(func=command_validate)

    # Format command
    format_parser = subparsers.add_parser(
        "format",
        help="Format validated data for fine-tuning",
    )
    format_parser.add_argument(
        "--input",
        help="Path to validated_questions.jsonl",
    )
    format_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default 0.8)",
    )
    format_parser.set_defaults(func=command_format)

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run full augmentation -> validation -> formatting pipeline",
    )
    pipeline_parser.add_argument(
        "--paraphrase-count",
        type=int,
        default=8,
        help="Number of paraphrases per question",
    )
    pipeline_parser.add_argument(
        "--kb-questions-per-chunk",
        type=int,
        default=3,
        help="Number of KB-driven questions per chunk",
    )
    pipeline_parser.add_argument(
        "--no-adversarial",
        dest="include_adversarial",
        action="store_false",
        help="Skip adversarial question generation",
    )
    pipeline_parser.add_argument(
        "--min-score",
        type=int,
        default=4,
        help="Minimum acceptable validation score",
    )
    pipeline_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio",
    )
    pipeline_parser.set_defaults(func=command_pipeline)

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)

    # Run command
    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
