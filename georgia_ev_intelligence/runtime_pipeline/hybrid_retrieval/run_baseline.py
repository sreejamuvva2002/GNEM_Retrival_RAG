"""Baseline runner: all models × all pipelines × human-validated questions → JSONL.

Usage:
    python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_baseline
    python -m georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_baseline \
        --models qwen2.5:7b llama3.1:8b \
        --limit 5
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from georgia_ev_intelligence.runtime_pipeline.generation.llm_adapter import OllamaAdapter
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.direct_kb_pipeline import (
    DirectKBAnswerPipeline,
)
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.pretrained_only_pipeline import (
    OnlyPretrainedAnswerPipeline,
)
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.rag_only_pipeline import (
    OnlyRagAnswerPipeline,
)
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.run_hybrid_rag import (
    DEFAULT_QUESTIONS_SHEET,
    DEFAULT_QUESTIONS_WORKBOOK,
    QuestionRow,
    _empty_trace_values,
    _format_retrieved_context,
    _load_questions,
    _project_root,
    _trace_values,
    build_prompt as build_hybrid_rag_prompt,
)
from georgia_ev_intelligence.runtime_pipeline.hybrid_retrieval.factory import build_default_pipeline
from georgia_ev_intelligence.runtime_pipeline.schemas import RetrievedChildChunk


ALL_MODELS = [
    "qwen2.5:7b",
    "llama3.1:8b",
    "mistral-small3.2:24b",
    "qwen3.5:35b-a3b",
    "qwen2.5:32b",
    "gemma3:27b",
    "qwen2.5:14b",
]

ALL_PIPELINES = ["rag_only", "hybrid_rag", "pretrained_only", "direct_kb"]


@dataclass
class BaselineRecord:
    """One answer record written as a single JSONL line."""

    question_id: object
    question: str
    ground_truth: str
    answer: str
    contexts: list[str]
    pipeline: str
    model: str
    trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "answer": self.answer,
            "contexts": self.contexts,
            "pipeline": self.pipeline,
            "model": self.model,
            "trace": self.trace,
        }


class RetrievalCache:
    """Run retrieval once per question and reuse the result across all models."""

    def __init__(self, pipeline_factory: Callable) -> None:
        self._factory = pipeline_factory
        self._pipeline = None
        self._load_error = ""
        self._cache: dict[str, dict] = {}

    def retrieve(self, question: str) -> dict:
        """Return cached retrieval result for a question."""
        if question in self._cache:
            return self._cache[question]
        result = self._run_retrieval(question)
        self._cache[question] = result
        return result

    def _run_retrieval(self, question: str) -> dict:
        try:
            pipeline = self._get_pipeline()
            if hasattr(pipeline, "retrieve_with_sources"):
                result = pipeline.retrieve_with_sources(question)
                parent_texts = [
                    p.parent_chunk_text
                    for p in result.parent_contexts
                    if p.parent_chunk_text
                ]
                return {
                    "contexts": parent_texts,
                    "formatted_context": "\n\n".join(parent_texts),
                    "trace": _trace_values(result.trace),
                }
            parent_contexts = pipeline.retrieve(question)
            parent_texts = [p.parent_chunk_text for p in parent_contexts if p.parent_chunk_text]
            return {
                "contexts": parent_texts,
                "formatted_context": "\n\n".join(parent_texts),
                "trace": _empty_trace_values(),
            }
        except Exception as exc:
            error = f"ERROR: retrieval failed: {exc}"
            return {
                "contexts": [],
                "formatted_context": error,
                "trace": _empty_trace_values(),
                "error": error,
            }

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if self._load_error:
            raise RuntimeError(self._load_error)
        try:
            self._pipeline = self._factory()
        except Exception as exc:
            self._load_error = str(exc)
            raise
        return self._pipeline


class BaselineRunner:
    """Loop over all models × all pipelines × all questions and write JSONL."""

    def __init__(
        self,
        output_dir: Path,
        questions: list[QuestionRow],
        models: list[str],
        pipelines: list[str],
        llm_timeout: int = 180,
    ) -> None:
        self._output_dir = output_dir
        self._questions = questions
        self._models = models
        self._pipelines = pipelines
        self._llm_timeout = llm_timeout
        self._retrieval_cache = RetrievalCache(build_default_pipeline)

    def run(self) -> None:
        total_models = len(self._models)
        for model_index, model_name in enumerate(self._models, start=1):
            print(f"\n[Model {model_index}/{total_models}] {model_name}")
            adapter = OllamaAdapter(model_name=model_name)
            direct_kb_pipeline = DirectKBAnswerPipeline(answer_generator=adapter.generate)

            for pipeline_name in self._pipelines:
                print(f"  Pipeline: {pipeline_name}")
                output_path = self._output_dir / _jsonl_filename(model_name, pipeline_name)
                self._run_pipeline(
                    model_name=model_name,
                    pipeline_name=pipeline_name,
                    adapter=adapter,
                    direct_kb_pipeline=direct_kb_pipeline,
                    output_path=output_path,
                )

        print(f"\nDone. Results saved to {self._output_dir}")

    def _run_pipeline(
        self,
        model_name: str,
        pipeline_name: str,
        adapter: OllamaAdapter,
        direct_kb_pipeline: DirectKBAnswerPipeline,
        output_path: Path,
    ) -> None:
        total = len(self._questions)
        with output_path.open("w", encoding="utf-8") as fh:
            for index, row in enumerate(self._questions, start=1):
                print(f"    [{index}/{total}] {row.question[:80]}")
                record = self._answer_one(
                    row=row,
                    pipeline_name=pipeline_name,
                    model_name=model_name,
                    adapter=adapter,
                    direct_kb_pipeline=direct_kb_pipeline,
                )
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                fh.flush()

    def _answer_one(
        self,
        row: QuestionRow,
        pipeline_name: str,
        model_name: str,
        adapter: OllamaAdapter,
        direct_kb_pipeline: DirectKBAnswerPipeline,
    ) -> BaselineRecord:
        if pipeline_name == "pretrained_only":
            return self._answer_pretrained(row, model_name, adapter)
        if pipeline_name == "direct_kb":
            return self._answer_direct_kb(row, model_name, direct_kb_pipeline)
        retrieval = self._retrieval_cache.retrieve(row.question)
        if pipeline_name == "rag_only":
            return self._answer_rag_only(row, model_name, adapter, retrieval)
        if pipeline_name == "hybrid_rag":
            return self._answer_hybrid_rag(row, model_name, adapter, retrieval)
        raise ValueError(f"Unknown pipeline: {pipeline_name}")

    def _answer_rag_only(
        self,
        row: QuestionRow,
        model_name: str,
        adapter: OllamaAdapter,
        retrieval: dict,
    ) -> BaselineRecord:
        pipeline = OnlyRagAnswerPipeline(answer_generator=adapter.generate)
        answer = _safe_answer(
            lambda: pipeline.answer(
                question=row.question,
                retrieved_context=retrieval["formatted_context"],
                timeout=self._llm_timeout,
            ),
            retrieval.get("error"),
        )
        return BaselineRecord(
            question_id=row.serial_number,
            question=row.question,
            ground_truth=row.golden_answer,
            answer=answer,
            contexts=retrieval["contexts"],
            pipeline="rag_only",
            model=model_name,
            trace=retrieval["trace"],
        )

    def _answer_hybrid_rag(
        self,
        row: QuestionRow,
        model_name: str,
        adapter: OllamaAdapter,
        retrieval: dict,
    ) -> BaselineRecord:
        answer = _safe_answer(
            lambda: adapter.generate(
                build_hybrid_rag_prompt(
                    user_question=row.question,
                    retrieved_parent_chunks=retrieval["formatted_context"],
                ),
                self._llm_timeout,
            ),
            retrieval.get("error"),
        )
        return BaselineRecord(
            question_id=row.serial_number,
            question=row.question,
            ground_truth=row.golden_answer,
            answer=answer,
            contexts=retrieval["contexts"],
            pipeline="hybrid_rag",
            model=model_name,
            trace=retrieval["trace"],
        )

    def _answer_pretrained(
        self,
        row: QuestionRow,
        model_name: str,
        adapter: OllamaAdapter,
    ) -> BaselineRecord:
        pipeline = OnlyPretrainedAnswerPipeline(answer_generator=adapter.generate)
        answer = _safe_answer(
            lambda: pipeline.answer(question=row.question, timeout=self._llm_timeout)
        )
        return BaselineRecord(
            question_id=row.serial_number,
            question=row.question,
            ground_truth=row.golden_answer,
            answer=answer,
            contexts=[],
            pipeline="pretrained_only",
            model=model_name,
            trace={},
        )

    def _answer_direct_kb(
        self,
        row: QuestionRow,
        model_name: str,
        direct_kb_pipeline: DirectKBAnswerPipeline,
    ) -> BaselineRecord:
        answer = _safe_answer(
            lambda: direct_kb_pipeline.answer(
                question=row.question,
                timeout=self._llm_timeout,
            )
        )
        return BaselineRecord(
            question_id=row.serial_number,
            question=row.question,
            ground_truth=row.golden_answer,
            answer=answer,
            contexts=direct_kb_pipeline.get_kb_records_as_text_list(),
            pipeline="direct_kb",
            model=model_name,
            trace={},
        )


def _safe_answer(fn: Callable, retrieval_error: str | None = None) -> str:
    if retrieval_error:
        return retrieval_error
    try:
        return fn()
    except Exception as exc:
        return f"ERROR: LLM generation failed: {exc}"


def _jsonl_filename(model_name: str, pipeline_name: str) -> str:
    safe_model = model_name.replace(":", "_").replace("/", "_")
    return f"{safe_model}__{pipeline_name}.jsonl"


def _write_config(
    output_dir: Path,
    models: list[str],
    pipelines: list[str],
    input_path: Path,
    sheet_name: str,
    question_count: int,
    llm_timeout: int,
) -> None:
    config_data = {
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "pipelines": pipelines,
        "input_file": str(input_path),
        "sheet_name": sheet_name,
        "question_count": question_count,
        "llm_timeout_seconds": llm_timeout,
        "total_llm_calls": len(models) * len(pipelines) * question_count,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()

    input_path = args.input or (_project_root() / "kb" / DEFAULT_QUESTIONS_WORKBOOK)
    questions = _load_questions(input_path=input_path, sheet_name=args.sheet)
    if args.limit is not None:
        questions = questions[: args.limit]

    models = args.models or ALL_MODELS
    pipelines = args.pipelines or ALL_PIPELINES

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        _project_root()
        / "georgia_ev_intelligence"
        / "outputs"
        / "baselines"
        / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_config(
        output_dir=output_dir,
        models=models,
        pipelines=pipelines,
        input_path=input_path,
        sheet_name=args.sheet,
        question_count=len(questions),
        llm_timeout=args.llm_timeout,
    )

    print(f"Output directory: {output_dir}")
    print(f"Models ({len(models)}): {', '.join(models)}")
    print(f"Pipelines ({len(pipelines)}): {', '.join(pipelines)}")
    print(f"Questions: {len(questions)}")
    print(f"Total LLM calls: {len(models) * len(pipelines) * len(questions)}")

    runner = BaselineRunner(
        output_dir=output_dir,
        questions=questions,
        models=models,
        pipelines=pipelines,
        llm_timeout=args.llm_timeout,
    )
    runner.run()
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all models × all pipelines over the human-validated QA workbook "
            "and save results as JSONL for RAGAS evaluation."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help=(
            "Ollama model names to run. Defaults to all 7 models. "
            f"Available: {', '.join(ALL_MODELS)}"
        ),
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=None,
        choices=ALL_PIPELINES,
        help=f"Pipelines to run. Defaults to all 4: {', '.join(ALL_PIPELINES)}",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=f"QA workbook path. Defaults to kb/{DEFAULT_QUESTIONS_WORKBOOK}.",
    )
    parser.add_argument(
        "--sheet",
        default=DEFAULT_QUESTIONS_SHEET,
        help=f"Worksheet name inside the QA workbook.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N questions. Useful for smoke tests.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=300,
        help="Timeout in seconds per LLM call (default 300; larger models need more).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
