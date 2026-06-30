"""Validation pipeline - LLM-as-judge scoring for synthetic data quality."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config
from .llm_client import LLMClient, get_client

logger = logging.getLogger(__name__)


class QAValidator:
    """Validate Q&A pairs using LLM-as-judge scoring."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_client()

    def score_single(self, question: str, answer: str) -> dict:
        """Score a single Q&A pair on accuracy, completeness, and relevance.

        Returns:
            Dict with 'accuracy', 'completeness', 'relevance' scores (1-5)
            and 'comments' field.
        """
        prompt = f"""You are an expert evaluator for an EV Intelligence Q&A system.
Score the following question-answer pair on three dimensions (1-5 scale):

1. ACCURACY: Does the answer correctly address the question?
   1=incorrect/contradictory, 5=perfectly accurate

2. COMPLETENESS: Does the answer fully address the question?
   1=incomplete/missing key details, 5=comprehensive

3. RELEVANCE: Is the question relevant to EV Intelligence/logistics domain?
   1=not relevant, 5=highly relevant and useful

Question: {question}
Answer: {answer}

Respond in JSON format only:
{{
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "relevance": <1-5>,
  "comments": "<brief explanation>"
}}"""

        try:
            response = self.client.generate(prompt, max_tokens=500)
            data = json.loads(response)
            return {
                "accuracy": int(data.get("accuracy", 3)),
                "completeness": int(data.get("completeness", 3)),
                "relevance": int(data.get("relevance", 3)),
                "comments": data.get("comments", ""),
            }
        except (json.JSONDecodeError, ValueError, Exception) as e:
            logger.error(f"Failed to score Q&A pair: {e}")
            return {
                "accuracy": 0,
                "completeness": 0,
                "relevance": 0,
                "comments": f"Scoring failed: {str(e)}",
            }

    def score_batch(
        self, questions: list[dict], min_score: int = config.VALIDATION_MIN_SCORE
    ) -> tuple[list[dict], list[dict]]:
        """Score a batch of Q&A pairs.

        Args:
            questions: List of dicts with 'question' and 'answer'
            min_score: Minimum acceptable score (default 4)

        Returns:
            Tuple of (passed, failed) lists, each with scoring info
        """
        passed = []
        failed = []

        for i, qa in enumerate(questions):
            logger.info(f"Scoring {i+1}/{len(questions)}: {qa['question'][:50]}...")

            scores = self.score_single(qa["question"], qa["answer"])
            avg_score = (
                scores["accuracy"] + scores["completeness"] + scores["relevance"]
            ) / 3

            qa_with_scores = {**qa, "scores": scores, "avg_score": avg_score}

            if avg_score >= min_score:
                passed.append(qa_with_scores)
            else:
                failed.append(qa_with_scores)

        logger.info(
            f"Validation complete: {len(passed)} passed, {len(failed)} failed "
            f"({len(passed)/len(questions)*100:.1f}% acceptance rate)"
        )
        return passed, failed

    def save_results(
        self, passed: list[dict], failed: list[dict], output_dir: Optional[Path] = None
    ):
        """Save validation results to files."""
        if output_dir is None:
            output_dir = config.FINETUNING_OUTPUTS_DIR

        # Save passed questions
        passed_path = output_dir / "validated_questions.jsonl"
        with open(passed_path, "w") as f:
            for qa in passed:
                # Remove scores for final dataset
                clean_qa = {"question": qa["question"], "answer": qa["answer"]}
                f.write(json.dumps(clean_qa) + "\n")
        logger.info(f"Saved {len(passed)} validated questions to {passed_path}")

        # Save detailed validation report
        report_path = output_dir / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump(
                {
                    "total_scored": len(passed) + len(failed),
                    "passed": len(passed),
                    "failed": len(failed),
                    "acceptance_rate": len(passed) / (len(passed) + len(failed))
                    if (len(passed) + len(failed)) > 0
                    else 0,
                    "passed_examples": passed[:10],  # Save first 10 examples
                    "failed_examples": failed[:10],
                },
                f,
                indent=2,
            )
        logger.info(f"Saved validation report to {report_path}")


def validate_augmented_dataset(
    input_path: Optional[Path] = None,
    min_score: int = config.VALIDATION_MIN_SCORE,
) -> tuple[list[dict], list[dict]]:
    """Load augmented questions and validate them.

    Args:
        input_path: Path to augmented_questions.jsonl
        min_score: Minimum acceptable average score

    Returns:
        Tuple of (passed, failed) lists
    """
    if input_path is None:
        input_path = config.AUGMENTED_QUESTIONS_JSONL

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return [], []

    # Load questions
    questions = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    logger.info(f"Loaded {len(questions)} augmented questions")

    # Validate
    validator = QAValidator()
    passed, failed = validator.score_batch(questions, min_score=min_score)

    # Save results
    validator.save_results(passed, failed)

    return passed, failed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    validate_augmented_dataset()
