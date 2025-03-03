from __future__ import annotations as _annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Generic

import logfire_api
from pydantic_core import to_jsonable_python
from rich.console import Console
from typing_extensions import TypeVar

from pydantic_evals._utils import UNSET, Unset
from pydantic_evals.reports import EvalReport, EvalReportCase, RenderNumberConfig, RenderValueConfig

__all__ = ('Eval', 'EvalCase', 'evaluation', 'increment_eval_metric')

# P = ParamSpec('P')
InputsT = TypeVar('InputsT')
OutputT = TypeVar('OutputT')
MetadataT = TypeVar('MetadataT', default=None)


@asynccontextmanager
async def evaluation(
    task: Callable[[InputsT], Awaitable[OutputT]],
    *,
    name: str | None = None,
    handler: Callable[[EvalCase[InputsT, OutputT, MetadataT]], Awaitable[None]] | None = None,
) -> AsyncIterator[Eval[InputsT, OutputT, MetadataT]]:
    """Context manager for starting an evaluation."""
    with logfire_api.span('evaluation', name=name) as eval_span:
        eval = Eval(task, name=name, span=eval_span, handler=handler)

        yield eval

        async def _handle_case(case: EvalCase[InputsT, OutputT, MetadataT]) -> None:
            if not case.has_output:
                async with case:
                    pass

        async with asyncio.TaskGroup() as group:
            for eval_case in eval.cases:
                # TODO: Use a semaphore to enforce some kind of max concurrency.
                #  See https://discuss.python.org/t/adding-a-limit-parameter-to-taskgroup-for-concurrent-coroutines-management/47686/3
                group.create_task(_handle_case(eval_case), name=eval_case.name)


@dataclass(init=False)
class Eval(Generic[InputsT, OutputT, MetadataT]):
    """A container for evaluation cases.

    This should generally not be instantiated directly; instead, use the `evaluation` context manager.
    """

    task: Callable[[InputsT], Awaitable[OutputT]]
    name: str

    handler: Callable[[EvalCase[InputsT, OutputT, MetadataT]], Awaitable[None]] | None = field(repr=False)
    span: logfire_api.LogfireSpan | None = field(repr=False)

    cases: list[EvalCase[InputsT, OutputT, MetadataT]] = field(repr=False)

    def __init__(
        self,
        task: Callable[[InputsT], Awaitable[OutputT]],
        *,
        name: str | None = None,
        handler: Callable[[EvalCase[InputsT, OutputT, MetadataT]], Awaitable[None]] | None = None,
        span: logfire_api.LogfireSpan | None = None,
    ):
        if name is None:
            name = _get_task_name(task)

        self.task = task
        self.name = name
        self.handler = handler
        self.span = span

        self.cases = []

    def as_report(self) -> EvalReport:
        return EvalReport(name=self.name, cases=[c.as_report_case() for c in self.cases])

    def print_report(
        self,
        width: int | None = None,
        include_input: bool = False,
        include_output: bool = False,
        include_total_duration: bool = False,
        include_averages: bool = True,
        input_config: RenderValueConfig | None = None,
        output_config: RenderValueConfig | None = None,
        score_configs: dict[str, RenderNumberConfig] | None = None,
        label_configs: dict[str, RenderValueConfig] | None = None,
        metric_configs: dict[str, RenderNumberConfig] | None = None,
        duration_config: RenderNumberConfig | None = None,
    ) -> None:
        console = Console(width=width)
        console.print(
            self.as_report().console_table(
                include_input=include_input,
                include_output=include_output,
                include_total_duration=include_total_duration,
                include_averages=include_averages,
                input_config=input_config,
                output_config=output_config,
                score_configs=score_configs,
                label_configs=label_configs,
                metric_configs=metric_configs,
                duration_config=duration_config,
            )
        )

    def print_diff(
        self,
        *,
        baseline: Eval[InputsT, OutputT, MetadataT] | EvalReport,
        width: int | None = None,
        include_input: bool = False,
        include_output: bool = False,
        include_total_duration: bool = False,
        include_removed_cases: bool = False,
        include_averages: bool = True,
        input_config: RenderValueConfig | None = None,
        output_config: RenderValueConfig | None = None,
        score_configs: dict[str, RenderNumberConfig] | None = None,
        label_configs: dict[str, RenderValueConfig] | None = None,
        metric_configs: dict[str, RenderNumberConfig] | None = None,
        duration_config: RenderNumberConfig | None = None,
    ) -> None:
        if not isinstance(baseline, EvalReport):
            baseline = baseline.as_report()

        console = Console(width=width)
        console.print(
            self.as_report().console_table(
                baseline=baseline,
                include_input=include_input,
                include_output=include_output,
                include_total_duration=include_total_duration,
                include_removed_cases=include_removed_cases,
                include_averages=include_averages,
                input_config=input_config,
                output_config=output_config,
                score_configs=score_configs,
                label_configs=label_configs,
                metric_configs=metric_configs,
                duration_config=duration_config,
            )
        )

    def case(
        self,
        name: str,
        inputs: InputsT,
        metadata: MetadataT = None,
        handler: Callable[[EvalCase[InputsT, OutputT, MetadataT]], Awaitable[None]] | None = None,
    ) -> EvalCase[InputsT, OutputT, MetadataT]:
        case = EvalCase(name, self.task, inputs=inputs, metadata=metadata, handler=handler or self.handler)
        self.cases.append(case)
        return case


@dataclass
class DatasetItem(Generic[InputsT, MetadataT]):
    """A container for an evaluation case."""

    name: str
    inputs: InputsT
    metadata: MetadataT  # might include expected output, annotations, etc.


# @dataclass
# class Dataset(Generic[InputsT, MetadataT]):
#     items: list[DatasetItem[InputsT, MetadataT]]
#
#     def run(self, task: Callable[[InputsT], Awaitable[OutputT]]) -> None:
#         for item in self.items:
#             case = EvalCase(item.name, task, inputs=item.inputs, metadata=item.metadata)
#             case.run()
#
#
# @dataclass
# class _TaskRunResults(Generic[OutputT]):
#     output: OutputT
#     attributes: dict[str, Any]
#     metrics: dict[str, int | float]
#
#
# @dataclass
# class _OutputScoringResults(Generic[OutputT]):
#     scores: dict[str, int | float]
#     labels: dict[str, bool | str]


# @dataclass(init=False)
class EvalCase(Generic[InputsT, OutputT, MetadataT]):
    """A container for an evaluation case."""

    # dataset_item: DatasetItem[InputsT, MetadataT]
    # scores: dict[str, str | bool | int | float]

    def __init__(
        self,
        name: str,
        task: Callable[[InputsT], Awaitable[OutputT]],
        *,
        inputs: InputsT,
        metadata: MetadataT,
        handler: Callable[[EvalCase[InputsT, OutputT, MetadataT]], Awaitable[None]] | None = None,
    ):
        self.name = name
        self.task = task
        self.inputs = inputs
        self.metadata = metadata
        self._handler = handler

        self.scores: dict[str, int | float] = {}
        self.metrics: dict[str, int | float] = {}
        self.labels: dict[str, bool | str] = {}
        # self.attributes = {}

        self._case_span = logfire_api.span(
            'case: {name}',
            name=name,
            inputs=inputs,
            metadata=metadata,
        )
        self._task_span = logfire_api.span('task execution')
        self._output: OutputT | Unset = UNSET
        self._token: Any = None

    @property
    def has_output(self) -> bool:
        return not isinstance(self._output, Unset)

    @property
    def output(self) -> OutputT:
        if isinstance(self._output, Unset):
            raise RuntimeError('`output` accessed before it was set.')
        return self._output

    @output.setter
    def output(self, case_output: OutputT) -> None:
        self._output = case_output

    def record_score(self, name: str, value: int | float) -> None:
        score_attribute = f'score.{name}'
        self.scores[name] = value

        # If we want to use span links to store scores we can do something like this:
        # with logfire.span('score {name=} {value=}', name=name, value=value, _links=[(self.span.context, None)]):
        #     pass

        # We need to support updating scores via span links, but I'm not sure if we should _only_ support that
        self._case_span.set_attribute(score_attribute, value)

    def record_label(self, name: str, value: bool | str) -> None:
        label_attribute = f'label.{name}'
        self.labels[name] = value

        # If we want to use span links to store labels we can do something like this:
        # with logfire.span('label {name=} {value=}', name=name, value=value, _links=[(self.span.context, None)]):
        #     pass

        # We need to support updating labels via span links, but I'm not sure if we should _only_ support that
        self._case_span.set_attribute(label_attribute, value)

    def record_metric(self, name: str, value: int | float) -> None:
        metric_attribute = f'metric.{name}'
        self.metrics[name] = value

        # If we want to use span links to store metrics we can do something like this:
        # with logfire.span('metric {name=} {value=}', name=name, value=value, _links=[(self.span.context, None)]):
        #     pass

        # We need to support updating metrics via span links, but I'm not sure if we should _only_ support that
        self._case_span.set_attribute(metric_attribute, value)

    def increment_metric(self, name: str, amount: int | float) -> None:
        current_value = self.metrics.get(name, 0)
        incremented_value = current_value + amount
        self.record_metric(name, incremented_value)

    def record_attribute(self, name: str, value: Any) -> None:
        self._case_span.set_attribute(name, value)

    def handler(self, handler: Callable[[EvalCase[InputsT, OutputT, MetadataT]], Awaitable[None]]) -> None:
        self._handler = handler

    def as_report_case(self) -> EvalReportCase:
        jsonable_inputs = to_jsonable_python(self.inputs)
        report_inputs: dict[str, Any] = (
            jsonable_inputs if isinstance(jsonable_inputs, dict) else {'inputs': jsonable_inputs}
        )
        return EvalReportCase(
            name=self.name,
            inputs=report_inputs,
            output=self.output,
            scores=self.scores,
            metrics=self.metrics,
            labels=self.labels,
            task_duration=_get_span_duration(self._task_span),
            total_duration=_get_span_duration(self._case_span),
        )

    async def __aenter__(self):
        if _CURRENT_EVAL_CASE.get() is not None:
            raise RuntimeError('An eval case has already been entered. Evaluation cases should not be nested')
        self._token = _CURRENT_EVAL_CASE.set(self)
        self._case_span.__enter__()
        with self._task_span:
            case_output = await self.task(self.inputs)
        self._case_span.set_attribute('case_output', case_output)
        self.output = case_output
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: Any
    ) -> None:
        if self._handler is not None:
            await self._handler(self)
        self._case_span.__exit__(exc_type, exc_value, traceback)
        _CURRENT_EVAL_CASE.reset(self._token)


_CURRENT_EVAL_CASE = ContextVar[EvalCase[Any, Any, Any] | None]('_CURRENT_EVAL_CASE', default=None)


def increment_eval_metric(name: str, amount: int | float) -> None:
    """Increment the named metric for the current evaluation case."""
    current_case = _CURRENT_EVAL_CASE.get()
    if current_case is not None:
        current_case.increment_metric(name, amount)


def _get_task_name(func: Callable[..., Any]) -> str:
    def _unwrap(f: Callable[..., Any]) -> Callable[..., Any]:
        # Unwraps f, also unwrapping partials, for the sake of getting f's name
        if isinstance(f, partial):
            return _unwrap(f.func)
        return inspect.unwrap(f)

    return _unwrap(func).__qualname__


def _get_span_duration(span: logfire_api.LogfireSpan) -> float:
    end_time = span.end_time
    start_time = span.start_time
    assert isinstance(start_time, int), 'span is not started'
    assert isinstance(end_time, int), 'span is not finished'
    return (end_time - start_time) / 1_000_000_000
