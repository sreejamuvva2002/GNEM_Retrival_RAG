"""Run rag_only, hybrid_rag, and pretrained_only pipelines on human-validated QA."""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from georgia_ev_intelligence.runtime_pipeline.generation.llm_client import generate_answer
from georgia_ev_intelligence.runtime_pipeline.schemas import RetrievedChildChunk

from .factory import build_default_pipeline
from .pretrained_only_pipeline import OnlyPretrainedAnswerPipeline
from .rag_only_pipeline import OnlyRagAnswerPipeline
from .run_hybrid_rag import (
    DEFAULT_OUTPUT_DIR_NAME,
    DEFAULT_QUESTIONS_SHEET,
    DEFAULT_QUESTIONS_WORKBOOK,
    _empty_trace_values,
    _default_input_path,
    _format_retrieved_context,
    _load_questions,
    _project_root,
    _trace_values,
    build_prompt as build_current_prompt,
)


TRACE_COLUMNS = [
    "sparse_child_count",
    "dense_child_count",
    "merged_child_result_count",
    "unique_child_chunk_count",
    "unique_parent_id_count",
    "parent_context_count_before_rerank",
    "parent_context_count_after_rerank",
]

OUTPUT_COLUMNS = [
    "question",
    "golden_answer",
    "final_llm_answer",
    *TRACE_COLUMNS,
]

NO_CONTEXT_OUTPUT_COLUMNS = [
    "question",
    "golden_answer",
    "final_llm_answer",
    *TRACE_COLUMNS,
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
        prompt = self._prompt_builder(question, retrieved_context)
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
    dense_retrieved_context: str
    sparse_retrieved_context: str
    trace_values: dict[str, object]
    answers_by_mode: dict[str, str]


@dataclass(frozen=True)
class RetrievedContextBundle:
    """Formatted retrieval contexts used for workbook traces."""

    final_context: str
    dense_context: str
    sparse_context: str
    trace_values: dict[str, object]


class RunOutputDirectoryFactory:
    """Create a new output directory for each all-modes run."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or (
            _project_root()
            / "georgia_ev_intelligence"
            / "outputs"
            / DEFAULT_OUTPUT_DIR_NAME
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
    """Write one XLSX workbook per answer-generation mode and one JSON with retrieved contexts."""

    def __init__(self, output_dir: Path, specs: tuple[PipelineOutputSpec, ...]) -> None:
        self._output_dir = output_dir
        self._specs = specs
        self._rows_by_mode: dict[str, list[dict[str, object]]] = {
            spec.key: [] for spec in specs
        }
        self._context_rows: list[dict[str, object]] = []

    def append(self, answers: QuestionModeAnswers) -> None:
        for spec in self._specs:
            row = {
                "question": answers.question,
                "golden_answer": answers.golden_answer,
                "final_llm_answer": answers.answers_by_mode[spec.key],
                **answers.trace_values,
            }
            self._rows_by_mode[spec.key].append(row)
        self._context_rows.append({
            "question": answers.question,
            "retrieved_parent_chunks_after_reranking": answers.retrieved_context,
            "dense_retrieved_context": answers.dense_retrieved_context,
            "sparse_retrieved_context": answers.sparse_retrieved_context,
        })
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

        json_path = self._output_dir / "retrieved_contexts.json"
        json_path.write_text(
            json.dumps(self._context_rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


class Rewritten50AllModesRunner:
    """Coordinate retrieval and the three answer-generation modes."""

    def __init__(
        self,
        retrieval_pipeline_factory,
        pipelines: PipelineSet,
        llm_timeout: int = 180,
    ) -> None:
        self._retrieval_pipeline_factory = retrieval_pipeline_factory
        self._retrieval_pipeline = None
        self._retrieval_load_error = ""
        self._pipelines = pipelines
        self._llm_timeout = llm_timeout

    def run(self, questions) -> list[QuestionModeAnswers]:
        return list(self.iter_rows(questions))

    def iter_rows(self, questions):
        """Yield one output row per question."""
        for index, question_row in enumerate(questions, start=1):
            print(f"[{index}/{len(questions)}] {question_row.question}")
            retrieved_contexts = self._retrieve_contexts(question_row.question)
            yield QuestionModeAnswers(
                serial_number=question_row.serial_number,
                question=question_row.question,
                golden_answer=question_row.golden_answer,
                retrieved_context=retrieved_contexts.final_context,
                dense_retrieved_context=retrieved_contexts.dense_context,
                sparse_retrieved_context=retrieved_contexts.sparse_context,
                trace_values=retrieved_contexts.trace_values,
                answers_by_mode={
                    "only_rag": self._answer_with_context(
                        self._pipelines.only_rag,
                        question_row.question,
                        retrieved_contexts.final_context,
                    ),
                    "only_pretrained": self._answer_without_context(
                        self._pipelines.only_pretrained,
                        question_row.question,
                    ),
                    "rag_plus_pretrained": self._answer_with_context(
                        self._pipelines.rag_plus_pretrained,
                        question_row.question,
                        retrieved_contexts.final_context,
                    ),
                },
            )

    def _retrieve_contexts(self, question: str) -> RetrievedContextBundle:
        try:
            retrieval_pipeline = self._get_retrieval_pipeline()
            if hasattr(retrieval_pipeline, "retrieve_with_sources"):
                retrieval_result = retrieval_pipeline.retrieve_with_sources(question)
                return RetrievedContextBundle(
                    final_context=_format_retrieved_context(
                        retrieval_result.parent_contexts,
                    ),
                    dense_context=_format_child_contexts(
                        retrieval_result.dense_children,
                    ),
                    sparse_context=_format_child_contexts(
                        retrieval_result.sparse_children,
                    ),
                    trace_values=_trace_values(retrieval_result.trace),
                )

            parent_contexts = retrieval_pipeline.retrieve(question)
            return RetrievedContextBundle(
                final_context=_format_retrieved_context(parent_contexts),
                dense_context="",
                sparse_context="",
                trace_values=_empty_trace_values(),
            )
        except Exception as exc:
            error = f"ERROR: retrieval failed: {exc}"
            return RetrievedContextBundle(
                final_context=error,
                dense_context=error,
                sparse_context=error,
                trace_values=_empty_trace_values(),
            )

    def _get_retrieval_pipeline(self):
        if self._retrieval_pipeline is not None:
            return self._retrieval_pipeline
        if self._retrieval_load_error:
            raise RuntimeError(self._retrieval_load_error)
        try:
            self._retrieval_pipeline = self._retrieval_pipeline_factory()
        except Exception as exc:
            self._retrieval_load_error = str(exc)
            raise
        return self._retrieval_pipeline

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


def _format_child_contexts(children: list[RetrievedChildChunk]) -> str:
    """Format child retrieval results for workbook inspection."""
    sections: list[str] = []
    for index, child in enumerate(children, start=1):
        lines = [
            f"[{index}] chunk_id: {child.chunk_id}",
            f"parent_record_id: {child.parent_record_id}",
            f"chunk_type: {child.chunk_type}",
        ]
        for field_name, value in child.metadata.items():
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            lines.append(f"{field_name}: {text}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def main() -> int:
    args = _parse_args()
    input_path = args.input or _default_input_path()
    output_dir = RunOutputDirectoryFactory(root=args.output_root).create()

    questions = _load_questions(input_path=input_path, sheet_name=args.sheet)
    if args.limit is not None:
        questions = questions[: args.limit]

    runner = Rewritten50AllModesRunner(
        retrieval_pipeline_factory=build_default_pipeline,
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
            f"generation for kb/{DEFAULT_QUESTIONS_WORKBOOK}."
        )
    )
    parser.add_argument(
        "--sheet",
        default=DEFAULT_QUESTIONS_SHEET,
        help=f"Worksheet name inside kb/{DEFAULT_QUESTIONS_WORKBOOK}.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional QA workbook path. Defaults to the human-validated QA workbook.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory for run folders. Each invocation creates "
            "a new timestamped child folder. Defaults to "
            f"georgia_ev_intelligence/outputs/{DEFAULT_OUTPUT_DIR_NAME}."
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
