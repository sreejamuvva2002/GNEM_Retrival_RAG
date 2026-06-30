"""Data augmentation for fine-tuning - generates synthetic Q&A pairs."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config
from .llm_client import LLMClient, get_client

logger = logging.getLogger(__name__)


class QuestionParaphraser:
    """Generate paraphrases of validated questions."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_client()

    def paraphrase_single(
        self, question: str, answer: str, num_variations: int = 5
    ) -> list[dict]:
        """Generate paraphrases of a single question-answer pair.

        Args:
            question: Original question
            answer: Golden answer
            num_variations: Number of variations to generate

        Returns:
            List of dicts with 'question' and 'answer' keys
        """
        prompt = f"""Given this Q&A pair from an EV Intelligence knowledge base:

Question: {question}
Answer: {answer}

Generate {num_variations} natural variations of this question. Vary the tone, formality, and phrasing while keeping the core intent identical.

Format your response as a JSON array where each item has "question" and "answer" keys:
[
  {{"question": "variation 1", "answer": "answer"}},
  {{"question": "variation 2", "answer": "answer"}},
  ...
]

Only return valid JSON, no other text."""

        try:
            response = self.client.generate(prompt)
            variations = json.loads(response)
            if not isinstance(variations, list):
                logger.warning(f"Expected list from paraphrase, got: {type(variations)}")
                return []
            return variations
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to paraphrase '{question[:50]}...': {e}")
            return []

    def paraphrase_batch(
        self,
        questions: list[dict],
        num_variations_per_question: int = 5,
    ) -> list[dict]:
        """Paraphrase a batch of questions.

        Args:
            questions: List of dicts with 'question' and 'answer' keys
            num_variations_per_question: Variations per original question

        Returns:
            List of all paraphrased Q&A pairs
        """
        all_variations = []
        for i, qa in enumerate(questions):
            logger.info(
                f"Paraphrasing {i+1}/{len(questions)}: {qa['question'][:50]}..."
            )
            variations = self.paraphrase_single(
                qa["question"],
                qa["answer"],
                num_variations=num_variations_per_question,
            )
            all_variations.extend(variations)

        logger.info(f"Generated {len(all_variations)} paraphrased Q&A pairs")
        return all_variations


class KBQuestionGenerator:
    """Generate new questions grounded in Knowledge Base."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_client()
        self.kb_chunks = self._load_kb_chunks()

    def _load_kb_chunks(self) -> list[str]:
        """Load and chunk the Knowledge Base."""
        try:
            df = pd.read_excel(config.KB_DATA, sheet_name="chunks")
            # Assuming there's a text column (adjust as needed)
            chunks = df["text"].dropna().astype(str).tolist()
            logger.info(f"Loaded {len(chunks)} KB chunks")
            return chunks[:100]  # Limit to first 100 chunks for initial run
        except Exception as e:
            logger.warning(f"Failed to load KB chunks: {e}. Using empty list.")
            return []

    def generate_from_chunk(self, chunk: str, num_questions: int = 3) -> list[dict]:
        """Generate Q&A pairs from a single KB chunk.

        Args:
            chunk: Text chunk from Knowledge Base
            num_questions: Number of questions to generate

        Returns:
            List of dicts with 'question' and 'answer' keys
        """
        # Truncate chunk if too long
        chunk = chunk[:1000]

        prompt = f"""You are an EV Intelligence expert. Based on the following knowledge base text:

{chunk}

Generate {num_questions} diverse, domain-specific questions that can be answered ONLY using this text.
For each question, provide a detailed answer grounded entirely in the provided text.

Important:
- Questions should be realistic and challenging
- Answers must use only information from the text (no external knowledge)
- Vary question types (what, how, why, when, where)

Format as JSON array with "question" and "answer" keys:
[
  {{"question": "...", "answer": "..."}},
  ...
]

Return only valid JSON."""

        try:
            response = self.client.generate(prompt)
            questions = json.loads(response)
            if not isinstance(questions, list):
                return []
            return questions
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to generate questions from chunk: {e}")
            return []

    def generate_batch(self, num_questions_per_chunk: int = 3) -> list[dict]:
        """Generate questions from all KB chunks.

        Args:
            num_questions_per_chunk: Questions per chunk

        Returns:
            List of all generated Q&A pairs
        """
        all_questions = []
        for i, chunk in enumerate(self.kb_chunks):
            logger.info(f"Generating from chunk {i+1}/{len(self.kb_chunks)}")
            questions = self.generate_from_chunk(chunk, num_questions=num_questions_per_chunk)
            all_questions.extend(questions)

        logger.info(f"Generated {len(all_questions)} KB-driven Q&A pairs")
        return all_questions

    def generate_adversarial(self, num_samples: int = 50) -> list[dict]:
        """Generate adversarial 'I don't know' questions.

        Returns:
            List of dicts with 'question' and 'answer' keys
        """
        prompt = """Generate {num_samples} realistic questions about Electric Vehicles and related topics
that CANNOT be answered using a knowledge base about EV Intelligence, charging infrastructure,
and logistics in Georgia.

For each, provide an answer that appropriately declines to answer, like:
"I don't have information about this topic in my knowledge base."

Format as JSON:
[
  {{"question": "...", "answer": "I don't have information about..."}},
  ...
]

Return only valid JSON."""

        try:
            response = self.client.generate(prompt)
            questions = json.loads(response)
            return questions if isinstance(questions, list) else []
        except Exception as e:
            logger.error(f"Failed to generate adversarial questions: {e}")
            return []


def load_validated_questions() -> list[dict]:
    """Load the 50 human-validated questions."""
    try:
        df = pd.read_excel(config.HUMAN_QA_EXCEL)
        questions = []
        for _, row in df.iterrows():
            questions.append(
                {
                    "question": str(row.get("question", "")),
                    "answer": str(row.get("answer", "")),
                }
            )
        logger.info(f"Loaded {len(questions)} validated questions")
        return questions
    except Exception as e:
        logger.error(f"Failed to load validated questions: {e}")
        return []


def save_augmented_questions(questions: list[dict], output_path: Optional[Path] = None):
    """Save augmented questions to JSONL."""
    if output_path is None:
        output_path = config.AUGMENTED_QUESTIONS_JSONL

    with open(output_path, "w") as f:
        for qa in questions:
            f.write(json.dumps(qa) + "\n")

    logger.info(f"Saved {len(questions)} augmented questions to {output_path}")


def augment_dataset(
    paraphrase_count: int = config.PARAPHRASE_COUNT,
    kb_questions_per_chunk: int = config.KB_QUESTIONS_PER_CHUNK,
    include_adversarial: bool = config.INCLUDE_ADVERSARIAL,
) -> list[dict]:
    """Run full data augmentation pipeline.

    Returns:
        List of all augmented Q&A pairs
    """
    logger.info("Starting data augmentation pipeline...")

    # Load validated questions
    validated = load_validated_questions()
    if not validated:
        logger.error("No validated questions loaded. Aborting.")
        return []

    # Paraphrase validated questions
    logger.info(f"Paraphrasing {len(validated)} validated questions...")
    paraphraser = QuestionParaphraser()
    paraphrases = paraphraser.paraphrase_batch(
        validated, num_variations_per_question=paraphrase_count
    )

    # Generate KB-driven questions
    logger.info("Generating KB-driven questions...")
    kb_generator = KBQuestionGenerator()
    kb_questions = kb_generator.generate_batch(num_questions_per_chunk=kb_questions_per_chunk)

    # Generate adversarial questions
    adversarial = []
    if include_adversarial:
        logger.info("Generating adversarial questions...")
        adversarial = kb_generator.generate_adversarial(num_samples=50)

    # Combine all
    all_augmented = validated + paraphrases + kb_questions + adversarial
    logger.info(
        f"Total augmented dataset: {len(all_augmented)} Q&A pairs "
        f"({len(validated)} original + {len(paraphrases)} paraphrases + "
        f"{len(kb_questions)} KB-driven + {len(adversarial)} adversarial)"
    )

    # Save
    save_augmented_questions(all_augmented)

    return all_augmented


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    augment_dataset()
