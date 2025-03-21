from __future__ import annotations as _annotations

import asyncio
import functools
import inspect
import sys
from collections.abc import Awaitable, Sequence
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Literal, NotRequired, Self, Union

from pydantic._internal import _typing_extra

from pydantic_graph import _utils

from ._utils import get_unwrapped_function_name
from .evaluators.context import EvaluatorContext
from .evaluators.spec import SourcedEvaluatorOutput
from .otel.context_in_memory_span_exporter import context_subtree
from .reporting.reports import EvaluationReport, ReportCase, ReportCaseAggregate

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup
else:
    ExceptionGroup = ExceptionGroup

import logfire
import yaml
from pydantic import BaseModel, Field, ValidationError
from pydantic_core import to_json, to_jsonable_python
from typing_extensions import TypedDict, TypeVar

from .evaluators.spec import BoundEvaluatorFunction, Evaluator, EvaluatorFunction, EvaluatorSpec

# while waiting for https://github.com/pydantic/logfire/issues/745
try:
    import logfire._internal.stack_info
except ImportError:
    pass
else:
    from pathlib import Path

    logfire._internal.stack_info.NON_USER_CODE_PREFIXES += (str(Path(__file__).parent.absolute()),)

_logfire = logfire.Logfire(otel_scope='pydantic-evals')

InputsT = TypeVar('InputsT', default=dict[str, Any])  # TODO: Should we default to Any?
OutputT = TypeVar('OutputT', default=dict[str, Any] | None)  # TODO: Should we default to Any?
MetadataT = TypeVar('MetadataT', default=dict[str, Any] | None)  # TODO: Should we default to Any?

DEFAULT_DATASET_PATH = './test_cases.yaml'


class _CaseModel(BaseModel, Generic[InputsT, OutputT, MetadataT], extra='forbid'):
    """A single row of a "dataset", consisting of input, expected output, and metadata."""

    name: str
    inputs: InputsT
    metadata: MetadataT
    expected_output: OutputT | None = None
    evaluators: list[EvaluatorSpec] = Field(default_factory=list)


class _DatasetModel(BaseModel, Generic[InputsT, OutputT, MetadataT], extra='forbid'):
    """A dataset of test cases, each consisting of input, expected output, and metadata."""

    cases: list[_CaseModel[InputsT, OutputT, MetadataT]]
    evaluators: list[EvaluatorSpec] = Field(default_factory=list)


@dataclass(init=False)
class Case(Generic[InputsT, OutputT, MetadataT]):
    """A single row of a "dataset", consisting of input, expected output, and metadata."""

    name: str
    inputs: InputsT
    metadata: MetadataT
    expected_output: OutputT | None
    evaluators: list[Evaluator[InputsT, OutputT, MetadataT]]

    UNSET_NAME = '<unset>'

    def __init__(
        self,
        *,
        name: str = UNSET_NAME,
        inputs: InputsT,
        metadata: MetadataT = None,
        expected_output: OutputT | None = None,
        evaluators: Sequence[
            BoundEvaluatorFunction[InputsT, OutputT, MetadataT] | Evaluator[InputsT, OutputT, MetadataT]
        ] = (),
    ):
        self.name = name
        self.inputs = inputs
        self.metadata = metadata
        self.expected_output = expected_output
        self.evaluators = [
            e if isinstance(e, Evaluator) else Evaluator[InputsT, OutputT, MetadataT].from_function(e)
            for e in evaluators
        ]


class Dataset(BaseModel, Generic[InputsT, OutputT, MetadataT], extra='forbid', arbitrary_types_allowed=True):
    """A dataset of test cases, each consisting of input, expected output, and metadata."""

    cases: list[Case[InputsT, OutputT, MetadataT]]
    evaluators: list[Evaluator[InputsT, OutputT, MetadataT]]

    def __init__(
        self,
        *,
        cases: Sequence[Case[InputsT, OutputT, MetadataT]] = (),
        evaluators: Sequence[
            BoundEvaluatorFunction[InputsT, OutputT, MetadataT] | Evaluator[InputsT, OutputT, MetadataT]
        ] = (),
    ):
        for i, case in enumerate(cases):
            if case.name == Case.UNSET_NAME:
                case.name = f'Case {i}'

        super().__init__(cases=cases, evaluators=evaluators)
        self.cases = list(cases)
        self.evaluators = [
            a if isinstance(a, Evaluator) else Evaluator[InputsT, OutputT, MetadataT].from_function(a)
            for a in evaluators
        ]

    def evaluate_sync(
        self, task: Callable[[InputsT], Awaitable[OutputT]], name: str | None = None, max_concurrency: int | None = None
    ) -> EvaluationReport:
        return _utils.get_event_loop().run_until_complete(
            self.evaluate(task, name=name, max_concurrency=max_concurrency)
        )

    async def evaluate(
        self, task: Callable[[InputsT], Awaitable[OutputT]], name: str | None = None, max_concurrency: int | None = None
    ) -> EvaluationReport:
        """Evaluates the test cases in the dataset using the given task."""
        name = name or get_unwrapped_function_name(task)

        limiter = asyncio.Semaphore(max_concurrency) if max_concurrency is not None else nullcontext()
        with _logfire.span('evaluate {name}', name=name) as eval_span:

            async def _handle_case(case: Case[InputsT, OutputT, MetadataT]) -> ReportCase:
                async with limiter:
                    return await _run_task_and_evaluators(task, case, self.evaluators)

            async_tasks: list[asyncio.Task[ReportCase]] = []
            async with asyncio.TaskGroup() as group:
                for case in self.cases:
                    async_tasks.append(group.create_task(_handle_case(case), name=case.name))

            report = EvaluationReport(name=name, cases=[x.result() for x in async_tasks])
            # TODO(DavidM): This attribute will be too big in general; remove it once we can use child spans in details panel:
            eval_span.set_attribute('cases', report.cases)
            # TODO(DavidM): Remove this 'averages' attribute once we compute it in the details panel
            eval_span.set_attribute('averages', ReportCaseAggregate.average(report.cases))
        return report

    def add_case(
        self,
        name: str,
        inputs: InputsT,
        metadata: MetadataT,
        expected_output: OutputT | None = None,
        evaluators: Sequence[
            BoundEvaluatorFunction[InputsT, OutputT, MetadataT] | Evaluator[InputsT, OutputT, MetadataT]
        ] = (),
    ) -> None:
        """Adds a case to the evaluation."""
        case = Case[InputsT, OutputT, MetadataT](
            name=name,
            inputs=inputs,
            metadata=metadata,
            expected_output=expected_output,
            evaluators=evaluators,
        )
        self.cases.append(case)

    def add_evaluator(
        self,
        evaluator: BoundEvaluatorFunction[InputsT, OutputT, MetadataT] | Evaluator[InputsT, OutputT, MetadataT],
        case_name: str | None = None,
    ) -> None:
        evaluator = (
            evaluator
            if isinstance(evaluator, Evaluator)
            else Evaluator[InputsT, OutputT, MetadataT].from_function(evaluator)
        )
        if case_name is None:
            # Add the evaluator to the dataset itself
            self.evaluators.append(evaluator)
            return

        # Add the evaluator to rows with the given name (generally there should only be one), but error if none found
        added = False
        for case in self.cases:
            if case.name == case_name:
                case.evaluators.append(evaluator)
                added = True
        if not added:
            raise ValueError(f'Case {case_name!r} not found in the dataset')

    @classmethod
    @functools.cache
    def _serialization_type(cls) -> type[_DatasetModel[InputsT, OutputT, MetadataT]]:
        return _DatasetModel[cls._params()]  # type: ignore

    @classmethod
    @functools.cache
    def _params(cls) -> tuple[type[InputsT], type[OutputT], type[MetadataT]]:
        for c in cls.__mro__:
            metadata = getattr(c, '__pydantic_generic_metadata__')
            if len(args := (metadata.get('args', ()) or getattr(c, '__args__', ()))) == 3:
                return args
        raise ValueError(f'Could not determine the generic parameters for {cls}')

    # TODO: Task: Always save a schema file when saving the dataset
    def save(self, path: Path | str = DEFAULT_DATASET_PATH, schema_ref: str | None = None) -> None:
        path = self._get_relative_path(path)
        if path.exists():
            first_line = path.read_text().split('\n', 1)[0]
            if first_line.startswith('# yaml-language-server: $schema='):
                schema_ref = first_line.split('=', 1)[1]
        content = yaml.dump(to_jsonable_python(self), sort_keys=False)
        if schema_ref is not None:
            content = _ensure_yaml_language_server_line(content, schema_ref)
        path.write_text(content)

    @classmethod
    def from_yaml(
        cls,
        path: Path | str = DEFAULT_DATASET_PATH,
        custom_evaluators: Sequence[EvaluatorFunction[InputsT, OutputT, MetadataT]] = (),
    ) -> Self:
        path = cls._get_relative_path(path)
        if not path.exists():
            raise FileNotFoundError(f'{cls.__name__} dataset file {path} does not exist')

        dataset_model_type = cls._serialization_type()
        raw = path.read_text()
        loaded = yaml.safe_load(raw)
        try:
            dataset_model: _DatasetModel[InputsT, OutputT, MetadataT] = dataset_model_type.model_validate(loaded)
            # result.rows = [cls.serialized_row_type().model_validate(row.model_dump()) for row in result.rows]
        except ValidationError as e:
            raise ValueError(
                f'{cls.__name__} dataset file {path} contains data that does not match the schema:\n{e}.'
            ) from e

        registry = {get_unwrapped_function_name(f): f for f in custom_evaluators}

        cases: list[Case[InputsT, OutputT, MetadataT]] = []
        errors: list[ValueError] = []
        dataset_evaluators: list[Evaluator[Any, Any, Any]] = []
        for evaluator in dataset_model.evaluators:
            try:
                dataset_evaluator = Evaluator[InputsT, OutputT, MetadataT].from_registry(registry, None, evaluator)
            except ValueError as e:
                errors.append(e)
                continue
            dataset_evaluators.append(dataset_evaluator)

        for row in dataset_model.cases:
            evaluators: list[Evaluator[Any, Any, Any]] = []
            for spec in row.evaluators:
                try:
                    evaluator = Evaluator[InputsT, OutputT, MetadataT].from_registry(registry, row.name, spec)
                except ValueError as e:
                    errors.append(e)
                    continue
                evaluators.append(evaluator)
            row = Case[InputsT, OutputT, MetadataT](
                name=row.name,
                inputs=row.inputs,
                metadata=row.metadata,
                expected_output=row.expected_output,
            )
            row.evaluators = evaluators
            cases.append(row)
        if errors:
            raise ExceptionGroup(f'{len(errors)} error(s) loading evaluators from registry', errors[:3])
        result = cls(cases=cases)
        result.evaluators = dataset_evaluators
        return result

    @classmethod
    def generate_dataset_files(
        cls,
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
        schema_path: Path | str | None = None,
        custom_evaluators: Sequence[EvaluatorFunction[InputsT, OutputT, MetadataT]] = (),
    ) -> str:
        dataset_path = cls._get_relative_path(dataset_path)

        if schema_path is None:
            if dataset_path.exists():
                # Try to infer the schema path from the first line of the existing dataset file
                first_line = dataset_path.read_text().split('\n', 1)[0]
                if first_line.startswith('# yaml-language-server: $schema='):
                    schema_path = (dataset_path.parent / first_line.split('=', 1)[1]).resolve()
            if schema_path is None:
                schema_path = cls._get_schema_path(dataset_path)
        else:
            schema_path = cls._get_relative_path(schema_path)

        schema_content = to_json(cls.model_json_schema_with_evaluators(custom_evaluators), indent=2).decode() + '\n'
        if not schema_path.exists() or schema_path.read_text() != schema_content:
            schema_path.write_text(schema_content)

        schema_ref = str(_get_relative_path_reference(schema_path, dataset_path.parent))
        yaml_language_server_line = f'# yaml-language-server: $schema={schema_ref}'
        if dataset_path.exists():
            try:
                cls.from_yaml(dataset_path, custom_evaluators)
            except ValueError as e:
                if isinstance(e.__cause__, ValidationError):
                    raise ValueError(
                        f'{cls.__name__} dataset file {dataset_path} already exists, but does not contain compatible data.'
                        f' Fix or delete the file before calling this function:\n{e.__cause__}'
                    ) from e.__cause__
                else:
                    raise
            dataset_text = dataset_path.read_text()
            cases_text_with_schema = _ensure_yaml_language_server_line(dataset_text, schema_ref)
            if cases_text_with_schema != dataset_text:
                dataset_path.write_text(cases_text_with_schema)
        else:
            content = yaml.dump(to_jsonable_python(cls(cases=[])), sort_keys=False)
            dataset_path.write_text(f'{yaml_language_server_line}\n{content}')
        return schema_ref

    @classmethod
    def model_json_schema_with_evaluators(
        cls,
        scorers: Sequence[EvaluatorFunction[InputsT, OutputT, MetadataT]] = (),
    ) -> dict[str, Any]:
        registry = {get_unwrapped_function_name(f): f for f in scorers}

        evaluator_types: list[Any] = []
        for name, function in registry.items():
            signature = inspect.signature(function)

            scoring_context_param, *other_params = signature.parameters.values()
            type_hints = _typing_extra.get_function_type_hints(function)
            type_hints.pop(scoring_context_param.name, None)
            type_hints.pop('return', None)
            required_type_hints: dict[str, Any] = {}

            for p in other_params:
                type_hints.setdefault(p.name, Any)
                if p.default is not p.empty:
                    type_hints[p.name] = NotRequired[type_hints[p.name]]
                else:
                    required_type_hints[p.name] = type_hints[p.name]

            if len(type_hints) == 0 or not required_type_hints:
                # Shortest option: just the call name
                evaluator_types.append(Literal[name])
            if len(type_hints) == 1:
                # Short option: only have one parameter, so we can drop the nesting
                [type_hint_type] = type_hints.values()  # pyright: ignore
                td = TypedDict(f'short_evaluator_{name}', {name: type_hint_type})  # pyright: ignore
                td.__pydantic_config__ = {'extra': 'forbid'}  # pyright: ignore
                evaluator_types.append(td)
            if len(type_hints) > 1:
                if len(required_type_hints) == 1:
                    # Short option: only have one required parameter, so we can drop the nesting
                    type_hint_type = next(iter(required_type_hints.values()))  # pyright: ignore
                    td = TypedDict(f'short_evaluator_{name}', {name: type_hint_type})  # pyright: ignore
                    td.__pydantic_config__ = {'extra': 'forbid'}  # pyright: ignore
                    evaluator_types.append(td)

                # Long form: multiple parameters, or multiple required parameters
                params_td = TypedDict(f'evaluator_params_{name}', type_hints)  # pyright: ignore
                params_td.__pydantic_config__ = {'extra': 'forbid'}  # pyright: ignore
                td = TypedDict(f'evaluator_{name}', {name: params_td})  # pyright: ignore
                td.__pydantic_config__ = {'extra': 'forbid'}  # pyright: ignore
                evaluator_types.append(td)
            # Note: We might want to also generate the JSON schema for the format `call: '...', args: [...], kwargs: {...}`.
            #   It would be a bit complex to implement but not impossible.

        in_type, out_type, meta_type = cls._params()

        class ClsDatasetRow(BaseModel, extra='forbid'):
            name: str
            inputs: in_type
            metadata: meta_type
            expected_output: out_type | None = None
            evaluators: list[Union[tuple(evaluator_types)]] = []  # pyright: ignore  # noqa UP007

        ClsDatasetRow.__name__ = cls.__name__ + 'Row'

        class ClsDataset(BaseModel, extra='forbid'):
            cases: list[ClsDatasetRow]
            evaluators: list[Union[tuple(evaluator_types)]] = []  # pyright: ignore  # noqa UP007

        ClsDataset.__name__ = cls.__name__

        return ClsDataset.model_json_schema()

    # TODO: Task: Uncomment and finish implementing function to generate examples for a dataset using an LLM
    # @classmethod
    # def generate_dataset_examples(
    #     cls,
    #     model: models.Model | models.KnownModelName = 'gpt-4o',
    #     min_count: int = 3,
    #     dataset_path: Path | str = DEFAULT_DATASET_PATH,
    # ):
    #     dataset_path = cls._get_relative_path(dataset_path)
    #     schema_ref = cls.generate_dataset_files(dataset_path=dataset_path, schema_path=None)
    #
    #     existing_content: str | None = None
    #
    #     try:
    #         existing_rows = cls.from_yaml(dataset_path).rows
    #         min_count = max(0, min_count - len(existing_rows))
    #         if min_count == 0:
    #             return  # nothing to do, already have enough examples
    #         if existing_rows:
    #             existing_content = dataset_path.read_text()
    #     except FileNotFoundError:
    #         pass  # in this case, we'll generate a new file, so we ignore the error
    #
    #     examples = asyncio.run(_generate_examples(cls, dataset_path, model, min_count))
    #
    #     if existing_content is None:
    #         content = yaml.dump(to_jsonable_python(cls(rows=examples)), sort_keys=False)
    #         content = _ensure_yaml_language_server_line(content, schema_ref)
    #         dataset_path.write_text(content)
    #     else:
    #         new_lines = yaml.dump(to_jsonable_python(cls(rows=examples)), sort_keys=False).splitlines()
    #         new_lines = new_lines[1:]  # drop the first line, which is the document start
    #         new_content = _ensure_yaml_language_server_line(existing_content, schema_ref)
    #         if not new_content.endswith('\n'):
    #             new_content += '\n'
    #         new_content += '\n'.join(new_lines)
    #         dataset_path.write_text(new_content)

    @classmethod
    def _get_schema_path(cls, dataset_path: Path) -> Path:
        return dataset_path.parent / f'./{dataset_path.stem}_schema.json'

    @classmethod
    def _get_relative_path(cls, path: Path | str) -> Path:
        """Resolve relative paths as relative to the module in which the subclass is defined."""
        path = Path(path)

        if path.is_absolute():
            return path

        # TODO: Should we use the cwd instead of the module path? Then it would work for non-proper-subclasses..
        module_path = sys.modules[cls.__module__].__file__
        if module_path == __file__:
            raise ValueError(f'You should only call this method from a proper subclass of `{cls.__name__}`')

        assert module_path is not None, 'Module must be a file-based module'
        root = Path(module_path).parent
        return root / path


def _ensure_yaml_language_server_line(content: str, schema_ref: str) -> str:
    first_line = content.split('\n', 1)[0]
    yaml_language_server_line = f'# yaml-language-server: $schema={schema_ref}'
    if first_line == yaml_language_server_line:
        return content
    elif first_line.startswith('# yaml-language-server: $schema='):
        return '\n'.join([yaml_language_server_line] + content.split('\n')[1:])
    else:
        return f'{yaml_language_server_line}\n{content}'


# TODO: Use or lose this function
# async def _generate_examples(
#     dataset_type: type[Dataset[Any, Any, Any]],
#     path: Path,
#     model: models.Model | models.KnownModelName = 'gpt-4o',
#     n_examples: int = 3,
# ) -> list[SerializedDatasetRow[Any, Any, Any]]:
#     if path.exists():
#         cases_text = path.read_text()
#         try:
#             loaded = yaml.safe_load(cases_text)
#         except yaml.YAMLError:
#             raise ValueError(f'Cases file {path} is not valid YAML')
#
#         try:
#             existing_cases = dataset_type.type_adapter().validate_python(loaded).rows
#         except ValidationError as e:
#             raise ValueError(
#                 f'Cases file {path} contains data that does not match the schema. Delete the file before calling this function.'
#             ) from e
#     else:
#         existing_cases = []
#
#     n_examples = max(0, n_examples - len(existing_cases))
#     if n_examples == 0:
#         return []
#
#     from pydantic_ai import Agent  # import locally to prevent circular dependencies
#
#     agent = Agent(
#         model,
#         system_prompt=dedent('Generate concise example test cases that comply with the provided JSON schema.'),
#         result_type=dataset_type,
#         retries=1,
#     )
#     return (await agent.run(f'Generate {n_examples} examples')).data.rows


def _get_relative_path_reference(target: Path, source: Path, _prefix: str = '') -> Path:
    # Recursively resolve a relative path to target from source, adding '..' as needed.
    # This is useful for creating a relative path reference from a source file to a target file.
    # For example, if source is '/a/b/c.py' and target is '/a/d/e.py', the relative path reference
    # would be '../../d/e.py'.
    if not target.is_absolute():
        target = target.resolve()
    try:
        return Path(f'{_prefix}{Path(target).relative_to(source)}')
    except ValueError:
        return _get_relative_path_reference(target, source.parent, _prefix=f'{_prefix}../')


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
    task: Callable[[InputsT], Awaitable[OutputT]], case: Case[InputsT, OutputT, MetadataT]
) -> EvaluatorContext[InputsT, OutputT, MetadataT]:
    task_run = _TaskRun()
    if _CURRENT_TASK_RUN.get() is not None:
        raise RuntimeError('A task run has already been entered. Task runs should not be nested')
    token = _CURRENT_TASK_RUN.set(task_run)
    try:
        with _logfire.span('execute {task}', task=get_unwrapped_function_name(task)) as task_span:
            with context_subtree() as span_tree:
                task_output = await task(case.inputs)
    finally:
        _CURRENT_TASK_RUN.reset(token)

    # TODO: Question: Should we make this metric-attributes functionality more user-configurable in some way before merging?
    #   Note: the use of otel for collecting these metrics is the main reason why I think we should require at least otel as a dependency, if not logfire;
    #   otherwise, we don't have a great way to get usage data from arbitrary frameworks.
    #   Ideally we wouldn't need to hard-code the specific logic here, but I'm not sure a great way to expose it to
    #   users. Maybe via an argument of type Callable[[SpanTree], dict[str, int | float]] or similar?
    for node in span_tree.flattened():
        if node.attributes.get('gen_ai.operation.name') == 'chat':
            task_run.increment_metric('requests', 1)
        for k, v in node.attributes.items():
            if not isinstance(v, (int, float)):
                continue
            if k.startswith('gen_ai.usage.details.'):
                task_run.increment_metric(k[21:], v)
            if k.startswith('gen_ai.usage.'):
                task_run.increment_metric(k[13:], v)

    return EvaluatorContext[InputsT, OutputT, MetadataT](
        name=case.name,
        inputs=case.inputs,
        metadata=case.metadata,
        expected_output=case.expected_output,
        output=task_output,
        duration=_get_span_duration(task_span),
        span_tree=span_tree,
        attributes=task_run.attributes,
        metrics=task_run.metrics,
    )


async def _run_task_and_evaluators(
    task: Callable[[InputsT], Awaitable[OutputT]],
    case: Case[InputsT, OutputT, MetadataT],
    dataset_evaluators: list[Evaluator[InputsT, OutputT, MetadataT]],
) -> ReportCase:
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

        evaluators = case.evaluators + dataset_evaluators
        evaluator_outputs: list[SourcedEvaluatorOutput] = []
        if evaluators:
            async with asyncio.TaskGroup() as tg:
                tasks: list[asyncio.Task[list[SourcedEvaluatorOutput]]] = []
                for evaluator in evaluators:
                    tasks.append(tg.create_task(evaluator.execute(scoring_context)))
            for t in tasks:
                evaluator_outputs.extend(t.result())

        assertions, scores, labels = _group_evaluator_outputs_by_type(evaluator_outputs)
        case_span.set_attribute('assertions', assertions)
        case_span.set_attribute('scores', scores)
        case_span.set_attribute('labels', labels)

        context = case_span.context
        assert context is not None
        trace_id = f'{context.trace_id:032x}'
        span_id = f'{context.span_id:016x}'

    jsonable_inputs = to_jsonable_python(case.inputs)
    report_inputs: dict[str, Any] = (
        jsonable_inputs if isinstance(jsonable_inputs, dict) else {'inputs': jsonable_inputs}
    )

    return ReportCase(
        name=case.name,
        inputs=report_inputs,
        metadata=case.metadata,
        expected_output=case.expected_output,
        output=scoring_context.output,
        metrics=scoring_context.metrics,
        attributes=scoring_context.attributes,
        evaluator_outputs=evaluator_outputs,
        task_duration=scoring_context.duration,
        total_duration=_get_span_duration(case_span),
        trace_id=trace_id,
        span_id=span_id,
    )


def _group_evaluator_outputs_by_type(
    evaluators: Sequence[SourcedEvaluatorOutput],
) -> tuple[
    dict[str, SourcedEvaluatorOutput[bool]],
    dict[str, SourcedEvaluatorOutput[int | float]],
    dict[str, SourcedEvaluatorOutput[str]],
]:
    """Groups evaluators by value type."""
    assertions: dict[str, SourcedEvaluatorOutput[bool]] = {}
    scores: dict[str, SourcedEvaluatorOutput[int | float]] = {}
    labels: dict[str, SourcedEvaluatorOutput[str]] = {}
    seen_names = set[str]()
    for a in evaluators:
        name = a.name
        # Dedupe repeated names by adding a numeric suffix
        if name in seen_names:
            suffix = 2
            while f'{name}_{suffix}' in seen_names:
                suffix += 1
            name = f'{name}_{suffix}'
        seen_names.add(name)
        if assertion := a.downcast(bool):
            assertions[name] = assertion
        elif score := a.downcast(int, float):
            scores[name] = score
        elif label := a.downcast(str):
            labels[name] = label
    return assertions, scores, labels


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
