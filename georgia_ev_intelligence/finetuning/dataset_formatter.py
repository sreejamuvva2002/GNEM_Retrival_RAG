"""Format validated Q&A pairs into ChatML JSONL for Qwen fine-tuning."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


class ChatMLFormatter:
    """Convert Q&A pairs to Qwen ChatML format."""

    SYSTEM_PROMPT = (
        "You are an expert assistant specialized in Georgia EV Intelligence, "
        "providing accurate and helpful information about electric vehicles, "
        "charging infrastructure, logistics, and related topics. "
        "Answer based on the knowledge you've been trained on, and be honest "
        "if you don't know something or if it's outside your domain."
    )

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT

    def convert_single(self, question: str, answer: str) -> dict:
        """Convert a single Q&A pair to ChatML format.

        Returns:
            Dict with 'messages' key containing ChatML format list
        """
        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }

    def convert_batch(self, questions: list[dict]) -> list[dict]:
        """Convert a batch of Q&A pairs to ChatML format.

        Args:
            questions: List of dicts with 'question' and 'answer' keys

        Returns:
            List of dicts in ChatML format
        """
        formatted = []
        for qa in questions:
            formatted.append(self.convert_single(qa["question"], qa["answer"]))
        return formatted

    def save_jsonl(self, formatted_questions: list[dict], output_path: Path):
        """Save formatted questions to JSONL file."""
        with open(output_path, "w") as f:
            for qa in formatted_questions:
                f.write(json.dumps(qa) + "\n")
        logger.info(f"Saved {len(formatted_questions)} formatted examples to {output_path}")


def train_val_split(
    questions: list[dict],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split questions into train/validation sets.

    Args:
        questions: List of Q&A dicts
        train_ratio: Fraction for training (default 0.8)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_questions, val_questions)
    """
    df = pd.DataFrame(questions)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    train = train_df.to_dict("records")
    val = val_df.to_dict("records")

    logger.info(
        f"Train/validation split: {len(train)} train, {len(val)} validation "
        f"({train_ratio*100:.0f}/{(1-train_ratio)*100:.0f})"
    )
    return train, val


def format_validated_dataset(
    input_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    train_ratio: float = 0.8,
) -> tuple[Path, Path]:
    """Format validated dataset into ChatML JSONL files.

    Args:
        input_path: Path to validated_questions.jsonl
        output_dir: Directory to save formatted datasets
        train_ratio: Train/validation split ratio

    Returns:
        Tuple of (train_path, val_path)
    """
    if input_path is None:
        input_path = config.VALIDATED_QUESTIONS_JSONL
    if output_dir is None:
        output_dir = config.FINETUNING_OUTPUTS_DIR

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return None, None

    # Load validated questions
    questions = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    logger.info(f"Loaded {len(questions)} validated questions")

    # Split into train/validation
    train_questions, val_questions = train_val_split(
        questions, train_ratio=train_ratio
    )

    # Format to ChatML
    formatter = ChatMLFormatter()
    train_formatted = formatter.convert_batch(train_questions)
    val_formatted = formatter.convert_batch(val_questions)

    # Save
    train_path = output_dir / "train_dataset.jsonl"
    val_path = output_dir / "val_dataset.jsonl"

    formatter.save_jsonl(train_formatted, train_path)
    formatter.save_jsonl(val_formatted, val_path)

    logger.info(f"Training examples: {len(train_formatted)}")
    logger.info(f"Validation examples: {len(val_formatted)}")

    return train_path, val_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    format_validated_dataset()
