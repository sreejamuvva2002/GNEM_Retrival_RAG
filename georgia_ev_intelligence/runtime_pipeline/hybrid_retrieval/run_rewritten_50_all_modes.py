"""Run current, Only RAG, and Only Pre-Trained answer modes on 50 questions."""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from georgia_ev_intelligence.runtime_pipeline.generation.llm_client import generate_answer

from .factory import build_default_pipeline
from .only_pretrained_pipeline import OnlyPretrainedAnswerPipeline
from .only_rag_pipeline import OnlyRagAnswerPipeline
from .run_rewritten_50 import (
    _format_retrieved_context,
    _load_questions,
    _project_root,
    build_prompt as build_current_prompt,
)


OUTPUT_COLUMNS = [
    "s.no",
    "question",
    "golden answer",
    "LLM answer",
    "retrieved context",
]

NO_CONTEXT_OUTPUT_COLUMNS = [
    "s.no",
    "question",
    "golden answer",
    "LLM answer",
]


class ContextualAnswerPipeline(Protocol):
    """Generate an answer from a question and retrieved context."""

    def answer(
        self,
        question: str,
        retrieved_context: str,
        timeout: int = 180,
    ) -> str:
        """Return the generated answer."""


class NonContextualAnswerPipeline(Protocol):
    """Generate an answer from only a question."""

    def answer(self, question: str, timeout: int = 180) -> str:
        """Return the generated answer."""


class CurrentAnswerPipeline:
    """Run the existing final-answer prompt used by the current batch flow."""

    def __init__(
        self,
        prompt_builder: Callable[[str, str], str] = build_current_prompt,
        answer_generator: Callable[[str, int], str] = generate_answer,
    ) -> None:
        self._prompt_builder = prompt_builder
        self._answer_generator = answer_generator

    def answer(
        self,
        question: str,
        retrieved_context: str,
        timeout: int = 180,
    ) -> str:
        prompt = self._prompt_builder(
            user_question=question,
            retrieved_parent_chunks=retrieved_context,
        )
        return self._answer_generator(prompt, timeout)


@dataclass(frozen=True)
class PipelineSet:
    """Injected answer-generation strategies for the three-mode batch run."""

    only_rag: ContextualAnswerPipeline
    only_pretrained: NonContextualAnswerPipeline
    rag_plus_pretrained: ContextualAnswerPipeline


@dataclass(frozen=True)
class PipelineOutputSpec:
    """Output metadata for one answer-generation mode."""

    key: str
    filename: str
    include_retrieved_context: bool = True


@dataclass(frozen=True)
class QuestionModeAnswers:
    """Generated answers for one question across the three modes."""

    serial_number: object
    question: str
    golden_answer: str
    retrieved_context: str
    answers_by_mode: dict[str, str]


class RunOutputDirectoryFactory:
    """Create a new output directory for each all-modes run."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or (
            _project_root()
            / "georgia_ev_intelligence"
            / "outputs"
            / "hybrid_retrieval_rewritten_50"
        )

    def create(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self._root / f"{timestamp}_all_modes"
        suffix = 1
        while output_dir.exists():
            output_dir = self._root / f"{timestamp}_all_modes_{suffix}"
            suffix += 1
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir


class PipelineWorkbookWriter:
    """Write one XLSX workbook per answer-generation mode."""

    def __init__(self, output_dir: Path, specs: tuple[PipelineOutputSpec, ...]) -> None:
        self._output_dir = output_dir
        self._specs = specs
        self._rows_by_mode: dict[str, list[dict[str, object]]] = {
            spec.key: [] for spec in specs
        }

    def append(self, answers: QuestionModeAnswers) -> None:
        for spec in self._specs:
            row = {
                "s.no": answers.serial_number,
                "question": answers.question,
                "golden answer": answers.golden_answer,
                "LLM answer": answers.answers_by_mode[spec.key],
            }
            if spec.include_retrieved_context:
                row["retrieved context"] = answers.retrieved_context
            self._rows_by_mode[spec.key].append(row)
        self.write()

    def write(self) -> None:
        for spec in self._specs:
            path = self._output_dir / spec.filename
            columns = (
                OUTPUT_COLUMNS
                if spec.include_retrieved_context
                else NO_CONTEXT_OUTPUT_COLUMNS
            )
            dataframe = pd.DataFrame(self._rows_by_mode[spec.key], columns=columns)
            dataframe.to_excel(path, index=False)


class Rewritten50AllModesRunner:
    """Coordinate retrieval and the three answer-generation modes."""

    def __init__(
        self,
        retrieval_pipeline,
        pipelines: PipelineSet,
        llm_timeout: int = 180,
    ) -> None:
        self._retrieval_pipeline = retrieval_pipeline
        self._pipelines = pipelines
        self._llm_timeout = llm_timeout

    def run(self, questions) -> list[QuestionModeAnswers]:
        return list(self.iter_rows(questions))

    def iter_rows(self, questions):
        """Yield one output row per question."""
        for index, question_row in enumerate(questions, start=1):
            print(f"[{index}/{len(questions)}] {question_row.question}")
            retrieved_context = self._retrieve_context(question_row.question)
            yield QuestionModeAnswers(
                serial_number=question_row.serial_number,
                question=question_row.question,
                golden_answer=question_row.golden_answer,
                retrieved_context=retrieved_context,
                answers_by_mode={
                    "only_rag": self._answer_with_context(
                        self._pipelines.only_rag,
                        question_row.question,
                        retrieved_context,
                    ),
                    "only_pretrained": self._answer_without_context(
                        self._pipelines.only_pretrained,
                        question_row.question,
                    ),
                    "rag_plus_pretrained": self._answer_with_context(
                        self._pipelines.rag_plus_pretrained,
                        question_row.question,
                        retrieved_context,
                    ),
                },
            )

    def _retrieve_context(self, question: str) -> str:
        try:
            parent_contexts = self._retrieval_pipeline.retrieve(question)
            return _format_retrieved_context(parent_contexts)
        except Exception as exc:
            return f"ERROR: retrieval failed: {exc}"

    def _answer_with_context(
        self,
        pipeline: ContextualAnswerPipeline,
        question: str,
        retrieved_context: str,
    ) -> str:
        if retrieved_context.startswith("ERROR: retrieval failed:"):
            return retrieved_context
        try:
            return pipeline.answer(
                question=question,
                retrieved_context=retrieved_context,
                timeout=self._llm_timeout,
            )
        except Exception as exc:
            return f"ERROR: LLM generation failed: {exc}"

    def _answer_without_context(
        self,
        pipeline: NonContextualAnswerPipeline,
        question: str,
    ) -> str:
        try:
            return pipeline.answer(question=question, timeout=self._llm_timeout)
        except Exception as exc:
            return f"ERROR: LLM generation failed: {exc}"


def main() -> int:
    args = _parse_args()
    input_path = _project_root() / "kb" / "Rewritten_50_questions.xlsx"
    output_dir = RunOutputDirectoryFactory(root=args.output_root).create()

    questions = _load_questions(input_path=input_path, sheet_name=args.sheet)
    if args.limit is not None:
        questions = questions[: args.limit]

    runner = Rewritten50AllModesRunner(
        retrieval_pipeline=build_default_pipeline(),
        pipelines=PipelineSet(
            only_rag=OnlyRagAnswerPipeline(),
            only_pretrained=OnlyPretrainedAnswerPipeline(),
            rag_plus_pretrained=CurrentAnswerPipeline(),
        ),
        llm_timeout=args.llm_timeout,
    )

    writer = PipelineWorkbookWriter(output_dir=output_dir, specs=_pipeline_output_specs())
    for row in runner.iter_rows(questions):
        writer.append(row)

    print(f"Saved {len(questions)} rows per pipeline to {output_dir}")
    return 0


def _pipeline_output_specs() -> tuple[PipelineOutputSpec, ...]:
    return (
        PipelineOutputSpec(
            key="only_rag",
            filename="only_rag.xlsx",
        ),
        PipelineOutputSpec(
            key="only_pretrained",
            filename="only_pre_trained.xlsx",
            include_retrieved_context=False,
        ),
        PipelineOutputSpec(
            key="rag_plus_pretrained",
            filename="rag_plus_pre_trained.xlsx",
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Only RAG, Only pre-trained, and Rag + Pre-Trained answer "
            "generation for kb/Rewritten_50_questions.xlsx."
        )
    )
    parser.add_argument(
        "--sheet",
        default="Q&A",
        help="Worksheet name inside kb/Rewritten_50_questions.xlsx.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory for run folders. Each invocation creates "
            "a new timestamped child folder. Defaults to "
            "georgia_ev_intelligence/outputs/hybrid_retrieval_rewritten_50."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke testing.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=180,
        help="Timeout in seconds for each LLM generation call.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
