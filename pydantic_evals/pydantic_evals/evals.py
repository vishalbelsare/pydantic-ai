# TODO: Add support for dataset-wide and case-specific assertions.
# TODO: Add a score to the evaluate span that tracks the percentage of cases that pass all assertions.
from __future__ import annotations as _annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Generic

import logfire
from pydantic_core import to_jsonable_python
from typing_extensions import TypeVar

from ._utils import get_unwrapped_function_name
from .assessments.scoring import ScoringContext
from .assessments.spec import Assessment, AssessmentDetail, AssessmentSpec, BoundAssessmentFunction
from .datasets import EvaluationRow
from .otel.context_in_memory_span_exporter import context_subtree_spans
from .otel.span_tree import SpanTree
from .reporting.reports import EvalReport, EvalReportCase, EvalReportCaseAggregate

# while waiting for https://github.com/pydantic/logfire/issues/745
try:
    import logfire._internal.stack_info
except ImportError:
    pass
else:
    from pathlib import Path

    logfire._internal.stack_info.NON_USER_CODE_PREFIXES += (str(Path(__file__).parent.absolute()),)

__all__ = ('Evaluation', 'increment_eval_metric')

_logfire = logfire.Logfire(otel_scope='pydantic-evals')


InputsT = TypeVar('InputsT', default=dict[str, Any])
OutputT = TypeVar('OutputT', default=dict[str, Any])
MetadataT = TypeVar('MetadataT', default=dict[str, Any])


@dataclass(init=False)
class Evaluation(Generic[InputsT, OutputT, MetadataT]):
    """A container for evaluation cases.

    This should generally not be instantiated directly; instead, use the `evaluation` context manager.
    """

    task: Callable[[InputsT], Awaitable[OutputT]]

    name: str
    span: logfire.LogfireSpan = field(repr=False)

    def __init__(
        self,
        task: Callable[[InputsT], Awaitable[OutputT]],
        *,
        name: str | None = None,
        cases: list[EvaluationRow[InputsT, OutputT, MetadataT]] | None = None,
        default_assessments: list[Assessment[InputsT, OutputT, MetadataT]] | None = None,
    ):
        if name is None:
            name = get_unwrapped_function_name(task)

        self.task = task
        self.name = name

        self.cases = cases or []
        # self.eval_cases: list[EvalCase[InputsT, OutputT, MetadataT]] = [EvalCase(c, scoring) for c in cases or []]
        self.span = _logfire.span('evaluate {name}', name=self.name)

        self._default_assessments: list[Assessment[InputsT, OutputT, MetadataT]] = default_assessments or []
        self._assessments_by_name: dict[str, list[Assessment[InputsT, OutputT, MetadataT]]] = defaultdict(list)

    def default_assessment(
        self, function: BoundAssessmentFunction[InputsT, OutputT, MetadataT], /
    ) -> BoundAssessmentFunction[InputsT, OutputT, MetadataT]:
        """A decorator that applies the decorated function as an assessment to all cases in the evaluation."""
        self._default_assessments.append(Assessment[InputsT, OutputT, MetadataT].from_function(function))
        return function

    def case_assessment(
        self, name: str
    ) -> Callable[
        [BoundAssessmentFunction[InputsT, OutputT, MetadataT]], BoundAssessmentFunction[InputsT, OutputT, MetadataT]
    ]:
        """A decorator that applies the decorated function as an assessment to the named case in the evaluation.

        If the named case does not exist, an error will be raised when the evaluation is run.
        """

        def decorator(
            function: BoundAssessmentFunction[InputsT, OutputT, MetadataT],
        ) -> BoundAssessmentFunction[InputsT, OutputT, MetadataT]:
            spec = AssessmentSpec(function.__name__)
            self._assessments_by_name[name].append(
                Assessment[InputsT, OutputT, MetadataT](spec=spec, function=function)
            )
            return function

        return decorator

    def add_case(
        self,
        name: str,
        inputs: InputsT,
        metadata: MetadataT,
        expected_output: OutputT | None = None,
        assessments: list[Assessment[InputsT, OutputT, MetadataT]] | None = None,
    ) -> Callable[
        [BoundAssessmentFunction[InputsT, OutputT, MetadataT]], BoundAssessmentFunction[InputsT, OutputT, MetadataT]
    ]:
        """Adds a case to the evaluation.

        Can be used as a decorator if you want to add a case-specific assessment.
        """
        row = EvaluationRow[InputsT, OutputT, MetadataT](
            name=name,
            inputs=inputs,
            metadata=metadata,
            expected_output=expected_output,
            assessments=assessments or [],
        )
        self.cases.append(row)

        def assess_case(
            function: BoundAssessmentFunction[InputsT, OutputT, MetadataT],
        ) -> BoundAssessmentFunction[InputsT, OutputT, MetadataT]:
            spec = AssessmentSpec(function.__name__)
            row.assessments.append(Assessment[InputsT, OutputT, MetadataT](spec=spec, function=function))
            return function

        return assess_case

    async def run(self, max_concurrency: int | None = None) -> EvalReport:
        with self.span:
            limiter = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else nullcontext()

            async def _handle_case(case: EvaluationRow[InputsT, OutputT, MetadataT]) -> EvalReportCase:
                async with limiter:
                    assessments = self._default_assessments + self._assessments_by_name.get(case.name, [])
                    return await run_case(self.task, case, assessments)

            async_tasks: list[asyncio.Task[EvalReportCase]] = []
            async with asyncio.TaskGroup() as group:
                for case in self.cases:
                    async_tasks.append(group.create_task(_handle_case(case), name=case.name))

            report = EvalReport(name=self.name, cases=[x.result() for x in async_tasks])

            # TODO: This attribute will be too big in general; remove it once we can use child spans in details panel:
            self.span.set_attribute('cases', report.cases)
            # TODO: Maybe remove this 'averages' attribute if we can compute it from child spans
            self.span.set_attribute('averages', EvalReportCaseAggregate.average(report.cases))
        return report


@dataclass
class _TaskRun:
    attributes: dict[str, Any] = field(init=False, default_factory=dict)
    metrics: dict[str, int | float] = field(init=False, default_factory=dict)

    def record_metric(self, name: str, value: int | float) -> None:
        self.metrics[name] = value

    def increment_metric(self, name: str, amount: int | float) -> None:
        current_value = self.metrics.get(name, 0)
        incremented_value = current_value + amount
        if current_value == 0 and incremented_value == 0:
            return  # Avoid recording a metric that is always zero
        self.record_metric(name, incremented_value)

    def record_attribute(self, name: str, value: Any) -> None:
        self.attributes[name] = value


async def _run_task(
    task: Callable[[InputsT], Awaitable[OutputT]], case: EvaluationRow[InputsT, OutputT, MetadataT]
) -> ScoringContext[InputsT, OutputT, MetadataT]:
    task_run = _TaskRun()
    if _CURRENT_TASK_RUN.get() is not None:
        raise RuntimeError('A task run has already been entered. Task runs should not be nested')
    token = _CURRENT_TASK_RUN.set(task_run)
    try:
        with _logfire.span('execute {task}', task=get_unwrapped_function_name(task)) as task_span:
            with context_subtree_spans() as finished_spans:
                task_output = await task(case.inputs)
    finally:
        _CURRENT_TASK_RUN.reset(token)

    # TODO: Make this metric-attributes functionality user-configurable in some way
    #   Note: this is the main reason why I think we should require at least otel as a dependency, if not logfire;
    #   otherwise, we don't have a great way to get usage data from arbitrary frameworks
    for span in finished_spans:
        attributes = span.attributes
        assert attributes is not None  # this appears to be guaranteed, despite type-hinting to the contrary
        for k, v in attributes.items():
            if not isinstance(v, (int, float)):
                continue
            if k.startswith('gen_ai.usage.details.'):
                task_run.increment_metric(k[21:], v)
            if k.startswith('gen_ai.usage.'):
                task_run.increment_metric(k[13:], v)

    return ScoringContext[InputsT, OutputT, MetadataT](
        name=case.name,
        inputs=case.inputs,
        metadata=case.metadata,
        expected_output=case.expected_output,
        output=task_output,
        duration=_get_span_duration(task_span),
        span_tree=SpanTree(finished_spans),
        attributes=task_run.attributes,
        metrics=task_run.metrics,
    )


async def run_case(
    task: Callable[[InputsT], Awaitable[OutputT]],
    case: EvaluationRow[InputsT, OutputT, MetadataT],
    extra_assessments: list[Assessment[InputsT, OutputT, MetadataT]],
) -> EvalReportCase:
    # dataset_row = case.dataset_row
    # handler = case.handler

    with _logfire.span(
        '{task_name}: {case_name}',
        task_name=get_unwrapped_function_name(task),
        case_name=case.name,
        inputs=case.inputs,
        metadata=case.metadata,
    ) as case_span:
        scoring_context = await _run_task(task, case)

        case_span.set_attribute('output', scoring_context.output)
        case_span.set_attribute('task_duration', scoring_context.duration)
        case_span.set_attribute('metrics', scoring_context.metrics)
        case_span.set_attribute('attributes', scoring_context.attributes)

        assessments = case.assessments + extra_assessments
        assessment_details: list[AssessmentDetail] = []
        if assessments:
            async with asyncio.TaskGroup() as tg:
                tasks: list[asyncio.Task[list[AssessmentDetail]]] = []
                for assessment in assessments:
                    tasks.append(tg.create_task(assessment.execute(scoring_context)))
            for t in tasks:
                assessment_details.extend(t.result())
        case_span.set_attribute('assessments', assessment_details)

        context = case_span.context
        assert context is not None
        trace_id = f'{context.trace_id:032x}'
        span_id = f'{context.span_id:016x}'

    jsonable_inputs = to_jsonable_python(case.inputs)
    report_inputs: dict[str, Any] = (
        jsonable_inputs if isinstance(jsonable_inputs, dict) else {'inputs': jsonable_inputs}
    )
    return EvalReportCase(
        name=case.name,
        inputs=report_inputs,
        metadata=case.metadata,
        expected_output=case.expected_output,
        output=scoring_context.output,
        metrics=scoring_context.metrics,
        attributes=scoring_context.attributes,
        assessments=assessment_details,
        task_duration=scoring_context.duration,
        total_duration=_get_span_duration(case_span),
        trace_id=trace_id,
        span_id=span_id,
    )


_CURRENT_TASK_RUN = ContextVar[_TaskRun | None]('_CURRENT_TASK_RUN', default=None)


def set_eval_attribute(name: str, value: Any) -> None:
    """Set the named attribute for the current eval task run. Do nothing if not in an eval task run."""
    current_case = _CURRENT_TASK_RUN.get()
    if current_case is not None:
        current_case.record_attribute(name, value)


def increment_eval_metric(name: str, amount: int | float) -> None:
    """Increment the named metric for the current eval task run. Do nothing if not in an eval task run."""
    current_case = _CURRENT_TASK_RUN.get()
    if current_case is not None:
        current_case.increment_metric(name, amount)


def _get_span_duration(span: logfire.LogfireSpan) -> float:
    end_time = span.end_time
    start_time = span.start_time
    assert isinstance(start_time, int), 'span is not started'
    assert isinstance(end_time, int), 'span is not finished'
    return (end_time - start_time) / 1_000_000_000
